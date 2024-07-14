# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Improved diffusion model architecture proposed in the paper
"Analyzing and Improving the Training Dynamics of Diffusion Models"."""

# Modifications under MIT License
#
# Copyright (c) 2023 Christopher Friesen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from dual_diffusion_utils import torch_compile

def patchify(x, h):
    return x.view(x.shape[0], x.shape[1]*h, x.shape[2]//h, x.shape[3])

def unpatchify(x, h):
    return x.view(x.shape[0], x.shape[1]//h, x.shape[2]*h, x.shape[3])

#----------------------------------------------------------------------------
# Normalize given tensor to unit magnitude with respect to the given
# dimensions. Default = all dimensions except the first.

def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)

#----------------------------------------------------------------------------
# Magnitude-preserving SiLU (Equation 81).

def mp_silu(x):
    return torch.nn.functional.silu(x) / 0.596

#----------------------------------------------------------------------------
# Magnitude-preserving sum (Equation 88).

def mp_sum(a, b, t=0.5):
    return a.lerp(b, t) / np.sqrt((1 - t) ** 2 + t ** 2)

#----------------------------------------------------------------------------
# Magnitude-preserving concatenation (Equation 103).

def mp_cat(a, b, dim=1, t=0.5):
    Na = a.shape[dim]
    Nb = b.shape[dim]
    C = np.sqrt((Na + Nb) / ((1 - t) ** 2 + t ** 2))
    wa = C / np.sqrt(Na) * (1 - t)
    wb = C / np.sqrt(Nb) * t
    return torch.cat([wa * a , wb * b], dim=dim)

def mp_cat_interleave(a, b, dim=1, t=0.5):
    Na = a.shape[dim]
    Nb = b.shape[dim]
    C = np.sqrt((Na + Nb) / ((1 - t) ** 2 + t ** 2))
    wa = C / np.sqrt(Na) * (1 - t)
    wb = C / np.sqrt(Nb) * t
    return torch.stack([wa * a , wb * b], dim=dim+1).reshape(*a.shape[:dim], a.shape[dim]*2, *a.shape[dim+1:])

#----------------------------------------------------------------------------
# Magnitude-preserving Fourier features (Equation 75).

class MPFourier(torch.nn.Module):

    def __init__(self, num_channels, bandwidth=1, eps=1e-3):
        super().__init__()
        
        self.register_buffer('freqs', torch.pi * torch.linspace(0, 1-eps, num_channels).erfinv() * bandwidth)
        self.register_buffer('phases', torch.pi/2 * (torch.arange(num_channels) % 2 == 0).float())

    def forward(self, x):
        y = x.float().ger(self.freqs.float()) + self.phases.float()
        return (y.cos() * np.sqrt(2)).to(x.dtype)

#----------------------------------------------------------------------------
# Magnitude-preserving convolution or fully-connected layer (Equation 47)
# with force weight normalization (Equation 66).

class MPConv(torch.nn.Module):

    __constants__ = ["in_channels", "out_channels", "weight_gain", "disable_weight_normalization", "groups"]

    def __init__(self, in_channels, out_channels, kernel, disable_weight_normalization=False, groups=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel))
        self.weight_gain = np.sqrt(self.weight[0].numel())

        self.disable_weight_normalization = disable_weight_normalization
        self.groups = groups

    def forward(self, x, gain=1):

        w = self.weight * (gain / self.weight_gain)
        if w.ndim == 2:
            return x @ w.t()
            
        assert w.ndim == 4
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-2]//2, w.shape[-1]//2), groups=self.groups)

    @torch.no_grad()
    def normalize_weights(self):
        if self.disable_weight_normalization: return
        self.weight.copy_(normalize(self.weight))
#----------------------------------------------------------------------------
# U-Net encoder/decoder block with optional self-attention (Figure 21).

class Block(torch.nn.Module):

    __constants__ = ["out_channels", "flavor", "dropout", "res_balance", "clip_act", "mlp_groups"]

    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        emb_channels,                   # Number of embedding channels.
        flavor              = 'enc',    # Flavor: 'enc' or 'dec'.
        dropout             = 0,        # Dropout probability.
        res_balance         = 0.5,      # Balance between main branch (0) and residual branch (1).
        clip_act            = 256,      # Clip output activations. None = do not clip.
        mlp_multiplier      = 2,        # Multiplier for the number of channels in the MLP.
        mlp_groups          = 8,        # Number of groups for the MLP.
        kernel              = [1,1],     #
    ):
        super().__init__()

        self.out_channels = out_channels
        self.flavor = flavor
        self.dropout = dropout
        self.res_balance = res_balance
        self.clip_act = clip_act
        self.mlp_groups = mlp_groups
        self.emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.conv_res0 = MPConv(out_channels if flavor == 'enc' else in_channels, out_channels * mlp_multiplier, kernel=kernel, groups=mlp_groups)
        self.emb_linear = MPConv(emb_channels, out_channels * mlp_multiplier, kernel=[1,1], groups=mlp_groups) if emb_channels != 0 else None
        self.conv_res1 = MPConv(out_channels * mlp_multiplier, out_channels, kernel=kernel, groups=mlp_groups)
        self.conv_skip = MPConv(in_channels, out_channels, kernel=[1,1], groups=1)

    def forward(self, x, emb):
        # Main branch.

        if self.flavor == 'enc':
            x = self.conv_skip(x)
            x = normalize(x, dim=1) # pixel norm

        # Residual branch.
        y = self.conv_res0(mp_silu(x))

        if self.emb_linear is not None:
            c = self.emb_linear(emb, gain=self.emb_gain) + 1
            y = mp_silu(y * c)

        if self.dropout != 0: # magnitude preserving fix for dropout
            if self.training:
                y = torch.nn.functional.dropout(y, p=self.dropout)
            else:
                y *= 1 - self.dropout

        y = self.conv_res1(y)

        # Connect the branches.
        if self.flavor == 'dec':
            x = self.conv_skip(x)
        x = mp_sum(x, y, t=self.res_balance)
        
        # Clip activations.
        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)
        return x

#----------------------------------------------------------------------------
# EDM2 U-Net model (Figure 21).

class UNet(ModelMixin, ConfigMixin):

    __constants__ = ["label_dim", "training", "dtype", "label_dropout", "label_balance", "sample_height", "sample_len",
                     "dropout", "concat_balance", "sigma_max", "sigma_min", "sigma_data"]

    @register_to_config
    def __init__(self,
        in_channels = 4,                    # Number of input channels.
        out_channels = 4,                   # Number of output channels.
        label_dim = 0,                      # Class label dimensionality. 0 = unconditional.
        label_dropout = 0.1,                # Dropout probability for class labels. 
        dropout = 0,                        # Dropout probability.
        model_channels_h       = 2048,        # Base multiplier for the number of channels.
        model_channels_v       = 2048,        # Base multiplier for the number of channels.
        emb_channels_h = None,
        emb_channels_v = None,
        logvar_channels = 128,              # Number of channels for training uncertainty estimation.
        num_layers_per_block = 4,           # Number of residual blocks per resolution.
        label_balance        = 0.5,         # Balance between noise embedding (0) and class embedding (1).
        concat_balance       = 0.5,         # Balance between skip connections (0) and main path (1).
        sigma_max = 200.,                    # Expected max noise std
        sigma_min = 0.03,                  # Expected min noise std
        sigma_data = 1.,                   # Expected data / input sample std
        mlp_multiplier_v = 2,                 # Multiplier for the number of channels in the MLP.
        mlp_multiplier_h = 2,                 # Multiplier for the number of channels in the MLP.
        mlp_groups_v = 8,                     # Number of groups for the MLP.
        mlp_groups_h = 8,                     # Number of groups for the MLP.
        sample_height = 32,
        sample_len = 512,
        #**block_kwargs,                    # Arguments for Block.
        last_global_step = 0,               # Only used to track training progress in config.
    ):
        super().__init__()

        block_kwargs = {"dropout": dropout }

        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sigma_data = sigma_data

        self.sample_height = sample_height
        self.sample_len = sample_len
        self.label_dim = label_dim
        self.label_dropout = label_dropout
        self.label_balance = label_balance
        self.dropout = dropout
        self.concat_balance = concat_balance
        self.out_gain = torch.nn.Parameter(torch.zeros([]))

        # Embedding.
        emb_channels_v = emb_channels_v or model_channels_v
        self.emb_fourier_v = MPFourier(emb_channels_v)
        self.emb_noise_v = MPConv(emb_channels_v, emb_channels_v, kernel=[])
        self.emb_label_v = MPConv(label_dim, emb_channels_v, kernel=[]) if label_dim != 0 else None
        self.emb_label_unconditional_v = MPConv(1, emb_channels_v, kernel=[]) if label_dim != 0 else None

        emb_channels_h = emb_channels_h or model_channels_h
        self.emb_fourier_h = MPFourier(emb_channels_h)
        self.emb_noise_h = MPConv(emb_channels_h, emb_channels_h, kernel=[])
        self.emb_label_h = MPConv(label_dim, emb_channels_h, kernel=[]) if label_dim != 0 else None
        self.emb_label_unconditional_h = MPConv(1, emb_channels_h, kernel=[]) if label_dim != 0 else None

        # Training uncertainty estimation.
        self.logvar_fourier = MPFourier(logvar_channels)
        self.logvar_linear = MPConv(logvar_channels, 1, kernel=[], disable_weight_normalization=True)

        self.conv_in_v = MPConv(in_channels * sample_height + 1, model_channels_v, kernel=[1,3])
        self.conv_in_h = MPConv(sample_len, model_channels_h, kernel=[1,1])

        # Encoder.
        self.enc = torch.nn.ModuleDict()            
        for idx in range(num_layers_per_block):

            self.enc[f'layer_v{idx}'] = Block(model_channels_v,
                                              model_channels_v,
                                              emb_channels_v,
                                              mlp_multiplier=mlp_multiplier_v,
                                              mlp_groups=mlp_groups_v,
                                              kernel=[1,3],
                                              flavor='enc', **block_kwargs)
            
            self.enc[f'layer_h{idx}'] = Block(model_channels_h,
                                              model_channels_h,
                                              emb_channels_h,
                                              mlp_multiplier=mlp_multiplier_h,
                                              mlp_groups=mlp_groups_h,
                                              flavor='enc', **block_kwargs)

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for idx in range(num_layers_per_block):

            self.dec[f'layer_h{idx}'] = Block(model_channels_h * 2,
                                              model_channels_h,
                                              emb_channels_h,
                                              mlp_multiplier=mlp_multiplier_h,
                                              mlp_groups=mlp_groups_h,
                                              flavor='dec', **block_kwargs)
            
            self.dec[f'layer_v{idx}'] = Block(model_channels_v * 2,
                                              model_channels_v,
                                              emb_channels_v,
                                              mlp_multiplier=mlp_multiplier_v,
                                              mlp_groups=mlp_groups_v,
                                              kernel=[1,3],
                                              flavor='dec', **block_kwargs)

        self.conv_out_h = MPConv(model_channels_h, sample_len, kernel=[1,1])
        self.conv_out_v = MPConv(model_channels_v, out_channels * sample_height, kernel=[1,3])

    @torch_compile(fullgraph=True, dynamic=False)
    def forward(self, x_in, sigma, class_embeddings, t_ranges, format, return_logvar=False):

        with torch.no_grad():
            sigma = sigma.view(-1, 1, 1, 1)

            # Preconditioning weights.
            c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
            c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
            c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
            c_noise = (sigma.flatten().log() / 4).to(self.dtype)

            x = patchify(c_in * x_in, h=self.sample_height).to(self.dtype)

        # Embedding.
        emb_v = self.emb_noise_v(self.emb_fourier_v(c_noise))
        emb_h = self.emb_noise_h(self.emb_fourier_h(c_noise))
        if self.label_dim != 0:
            if class_embeddings is None or (self.training and self.label_dropout != 0):
                unconditional_embedding_v = self.emb_label_unconditional_v(torch.ones(1, device=self.device, dtype=self.dtype))
                unconditional_embedding_h = self.emb_label_unconditional_h(torch.ones(1, device=self.device, dtype=self.dtype))
            if class_embeddings is not None:
                if self.training and self.label_dropout != 0:
                    conditioning_mask = torch.nn.functional.dropout(torch.ones(class_embeddings[0].shape[0],
                                                                               device=self.device,
                                                                               dtype=self.dtype),
                                                                               p=self.label_dropout).unsqueeze(1)
                    class_embeddings_v = class_embeddings[0] * conditioning_mask + unconditional_embedding_v * (1 - conditioning_mask)
                    class_embeddings_h = class_embeddings[1] * conditioning_mask + unconditional_embedding_h * (1 - conditioning_mask)
            else:
                class_embeddings_v = unconditional_embedding_v
                class_embeddings_h = unconditional_embedding_h
            emb_v = mp_sum(emb_v, class_embeddings_v.to(emb_v.dtype), t=self.label_balance)
            emb_h = mp_sum(emb_h, class_embeddings_h.to(emb_h.dtype), t=self.label_balance)
        emb_v = mp_silu(emb_v).unsqueeze(2).unsqueeze(3).to(x.dtype)
        emb_h = mp_silu(emb_h).unsqueeze(2).unsqueeze(3).to(x.dtype)

        # Encoder.
        x = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)
        x = self.conv_in_v(x).transpose(1,-1)
        x = self.conv_in_h(x).transpose(1,-1)

        skips = []
        for name, block in self.enc.items():
            emb = emb_v if "layer_v" in name else emb_h
            x = block(x, emb)
            skips.append(x)
            x = x.transpose(1,-1)

        #x = x.transpose(1,-1)

        # Decoder.
        for name, block in self.dec.items():
            emb = emb_v if "layer_v" in name else emb_h
            x = mp_cat(x, skips.pop(), t=self.concat_balance)
            x = block(x, emb).transpose(1,-1)

        x = self.conv_out_h(x).transpose(1,-1)
        x = self.conv_out_v(x, gain=self.out_gain)

        D_x = c_skip * x_in + c_out * unpatchify(x.float(), h=self.sample_height)

        # Training uncertainty, if requested.
        if return_logvar:
            logvar = self.logvar_linear(self.logvar_fourier(c_noise)).view(-1, 1, 1, 1)
            return D_x, logvar.float()
        
        return D_x
    
    def get_class_embeddings(self, class_labels):
        class_labels = normalize(class_labels).to(device=self.device, dtype=self.dtype)
        class_embeddings_v = self.emb_label_v(class_labels)
        class_embeddings_h = self.emb_label_h(class_labels)
        #return torch.stack((class_embeddings_v, class_embeddings_h), dim=1)
        return class_embeddings_v, class_embeddings_h
    
    @torch.no_grad()
    def normalize_weights(self):
        for module in self.modules():
            if isinstance(module, MPConv):
                module.normalize_weights()

if __name__ == "__main__": # fourier embedding inner product test

    from dual_diffusion_utils import save_raw, save_raw_img
    from dotenv import load_dotenv
    import os

    load_dotenv(override=True)
    
    steps = 200
    cnoise = 192*4
    sigma_max = 80.
    sigma_min = 0.002

    emb_fourier = MPFourier(cnoise)
    noise_label = torch.linspace(np.log(sigma_max), np.log(sigma_min), steps) / 4

    emb = emb_fourier(noise_label)
    inner_products = (emb.view(1, steps, cnoise) * emb.view(steps, 1, cnoise)).sum(dim=2)

    debug_path = os.environ.get("DEBUG_PATH", None)
    if debug_path is not None:    
        save_raw(inner_products / inner_products.amax(), os.path.join(debug_path, "fourier_inner_products.raw"))
        save_raw_img(inner_products, os.path.join(debug_path, "fourier_inner_products.png"))

        coverage = inner_products.sum(dim=0)
        save_raw(coverage / coverage.amax(), os.path.join(debug_path, "fourier_inner_products_coverage.raw"))