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


def patchify(x, h=32):
    return x.view(x.shape[0], x.shape[1]*h, x.shape[2]//h, x.shape[3])

def unpatchify(x, h=32):
    return x.view(x.shape[0], x.shape[1]//h, x.shape[2]*h, x.shape[3])

def apply_rotary_embedding(x, pos_emb):
    real = x[:, ::2] * pos_emb[:,  ::2] - x[:, 1::2] * pos_emb[:, 1::2]
    imag = x[:, ::2] * pos_emb[:, 1::2] + x[:, 1::2] * pos_emb[:,  ::2]
    return torch.stack([real, imag], dim=2).reshape(x.shape)

def apply_pos_embedding(x, pos_emb):
    return torch.stack([x, x * pos_emb], dim=2).reshape(x.shape[0], x.shape[1]*2, *x.shape[2:])

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
# Upsample or downsample the given tensor with the given filter,
# or keep it as is.

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

#----------------------------------------------------------------------------
# Magnitude-preserving Fourier features (Equation 75).

class MPFourier(torch.nn.Module):

    def __init__(self, num_channels, bandwidth=1, eps=1e-3):
        super().__init__()
        
        self.register_buffer('freqs', torch.pi * torch.linspace(0, 1-eps, num_channels).erfinv() * bandwidth)
        self.register_buffer('phases', torch.pi/2 * (torch.arange(num_channels) % 2 == 0).float())

    def forward(self, x):
        if x.ndim == 1:
            y = x.float().ger(self.freqs.float()) + self.phases.float()
        else:
            y = (x.float() * self.freqs.float().view(1,-1, 1, 1) + self.phases.float().view(1,-1, 1, 1))
        return (y.cos() * np.sqrt(2)).to(x.dtype)

#----------------------------------------------------------------------------
# Magnitude-preserving convolution or fully-connected layer (Equation 47)
# with force weight normalization (Equation 66).

class MPConv(torch.nn.Module):

    __constants__ = ["weight_gain", "disable_weight_normalization", "groups"]

    def __init__(self, in_channels, out_channels, kernel, disable_weight_normalization=False, groups=1):
        super().__init__()

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

    __constants__ = ["num_heads", "dropout", "clip_act", "pos_emb_fn"]

    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        emb_channels,                   # Number of embedding channels.
        channels_per_head   = 128,      # Number of channels per attention head.
        dropout             = 0,        # Dropout probability.
        clip_act            = 256,      # Clip output activations. None = do not clip.
        mlp_multiplier      = 2,        # Multiplier for the number of channels in the MLP.
        rotary_pos_embedding = False,   # Use rotary position embedding for attention qk, or concatenated multiplicative if disabled
    ):
        super().__init__()

        self.num_heads = out_channels // channels_per_head
        self.dropout = dropout
        self.clip_act = clip_act
        self.emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.conv_res0 = MPConv(in_channels, out_channels * mlp_multiplier, kernel=[1,1])
        self.conv_depth = MPConv(out_channels * mlp_multiplier, out_channels * mlp_multiplier, kernel=[1,9], groups=8)
        print("HELLO?")
        self.emb_linear = MPConv(emb_channels, out_channels * mlp_multiplier, kernel=[1,1])
        self.conv_res1 = MPConv(out_channels * mlp_multiplier, out_channels, kernel=[1,1])
        self.attn_qk = MPConv(out_channels if rotary_pos_embedding else out_channels * 2, 2 * out_channels, kernel=[1,1])
        self.attn_v = MPConv(out_channels, out_channels, kernel=[1,1])
        self.pos_emb_fn = apply_rotary_embedding if rotary_pos_embedding else apply_pos_embedding

    def forward(self, x, emb, pos_emb):

        x = normalize(x, dim=1) # pixel norm

        # Residual branch.
        y = self.conv_res0(x)
        y = self.conv_depth(y)

        if self.emb_linear is not None:
            c = self.emb_linear(emb, gain=self.emb_gain) + 1
            y = mp_silu(y * c)

        if self.dropout != 0: # magnitude preserving fix for dropout
            if self.training:
                y = torch.nn.functional.dropout(y, p=self.dropout)
            else:
                y *= 1 - self.dropout

        y = self.conv_res1(y)
        x = mp_sum(x, y)

        # Self-attention.
        if self.num_heads != 0:
            
            qk = self.attn_qk(self.pos_emb_fn(x, pos_emb))
            qk = qk.reshape(qk.shape[0], self.num_heads, -1, 2, y.shape[2] * y.shape[3])
            q, k = normalize(qk, dim=2).unbind(3)

            v = self.attn_v(x)
            v = v.reshape(v.shape[0], self.num_heads, -1, y.shape[2] * y.shape[3])
            v = normalize(v, dim=2)

            y = torch.nn.functional.scaled_dot_product_attention(q.transpose(-1, -2),
                                                                 k.transpose(-1, -2),
                                                                 v.transpose(-1, -2)).transpose(-1, -2)
            x = mp_sum(x, y.reshape(x.shape))

        # Clip activations.
        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)
        return x

#----------------------------------------------------------------------------
# EDM2 U-Net model (Figure 21).

class UNet(ModelMixin, ConfigMixin):

    __constants__ = ["label_dim", "training", "dtype", "label_dropout",
                     "dropout", "sigma_max", "sigma_min", "sigma_data"]

    @register_to_config
    def __init__(self,
        in_channels = 4,                    # Number of input channels.
        out_channels = 4,                   # Number of output channels.
        logvar_channels = 128,              # Number of channels for training uncertainty estimation.
        channels_per_head = 192,            # Number of channels per attention head.
        label_dim = 0,                      # Class label dimensionality. 0 = unconditional.
        label_dropout = 0.1,                # Dropout probability for class labels. 
        dropout = 0,                        # Dropout probability.
        model_channels = 1536,              # Base multiplier for the number of channels.
        emb_channels = None,                # Multiplier for final embedding dimensionality. None = select based on channel_mult.
        num_layers_per_block = 8,           # Number of residual blocks per resolution.
        sigma_max = 200.,                   # Expected max noise std
        sigma_min = 0.03,                   # Expected min noise std
        sigma_data = 1.,                    # Expected data / input sample std
        mlp_multiplier = 4,                 # Multiplier for the number of channels in the MLP / conv layers.
        rotary_pos_embedding = False,        # Use rotary position embedding for attention qk, or concatenated multiplicative if disabled
        last_global_step = 0,               # Only used to track training progress in config.
        **kwargs,
    ):
        super().__init__()

        block_kwargs = {"channels_per_head": channels_per_head,
                        "dropout": dropout,
                        "mlp_multiplier": mlp_multiplier,
                        "rotary_pos_embedding": rotary_pos_embedding}

        cnoise = emb_channels or model_channels
        cemb = emb_channels or model_channels
        clogvar = logvar_channels

        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sigma_data = sigma_data

        self.label_dim = label_dim
        self.label_dropout = label_dropout
        self.dropout = dropout
        self.out_gain = torch.nn.Parameter(torch.zeros([]))

        # Embedding.
        self.emb_fourier = MPFourier(cnoise)
        self.emb_noise = MPConv(cnoise, cemb, kernel=[])
        self.emb_label = MPConv(label_dim, cemb, kernel=[]) if label_dim != 0 else None
        self.emb_label_unconditional = MPConv(1, cemb, kernel=[]) if label_dim != 0 else None
        self.emb_pos_fourier = MPFourier(model_channels, bandwidth=100)

        # Training uncertainty estimation.
        self.logvar_fourier = MPFourier(clogvar)
        self.logvar_linear = MPConv(clogvar, 1, kernel=[], disable_weight_normalization=True)

        conv_in_width = ((model_channels // (in_channels * 32)) // 2) * 2 + 1
        self.conv_in = MPConv(in_channels*32 + 1, model_channels, kernel=[1, conv_in_width])

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        for idx in range(num_layers_per_block):
            self.enc[f'block_enc_layer{idx}'] = Block(model_channels, model_channels, cemb, **block_kwargs)

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for idx in range(num_layers_per_block):
            self.dec[f'block_dec_layer{idx}'] = Block(model_channels, model_channels, cemb, **block_kwargs)

        self.conv_out = MPConv(model_channels, out_channels*32, kernel=[1, 3])

    def forward(self, x_in, sigma, class_embeddings, t_ranges, format, return_logvar=False):

        with torch.no_grad():
            sigma = sigma.view(-1, 1, 1, 1)

            # Preconditioning weights.
            c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
            c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
            c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
            c_noise = (sigma.flatten().log() / 4).to(self.dtype)

            x = patchify(c_in * x_in).to(self.dtype)

        # Embedding.
        emb = self.emb_noise(self.emb_fourier(c_noise))
        if self.label_dim != 0:
            if class_embeddings is None or (self.training and self.label_dropout != 0):
                unconditional_embedding = self.emb_label_unconditional(torch.ones(1, device=self.device, dtype=self.dtype))
            if class_embeddings is not None:
                if self.training and self.label_dropout != 0:
                    conditioning_mask = torch.nn.functional.dropout(torch.ones(class_embeddings.shape[0],
                                                                                device=self.device,
                                                                                dtype=self.dtype),
                                                                                p=self.label_dropout).unsqueeze(1)
                    class_embeddings = class_embeddings * conditioning_mask + unconditional_embedding * (1 - conditioning_mask)
            else:
                class_embeddings = unconditional_embedding 
            emb = mp_sum(emb, class_embeddings.to(emb.dtype), t=0.5)
        emb = mp_silu(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)

        pos_t = torch.linspace(-0.5, 0.5, x.shape[3], device=self.device)
        pos_emb = self.emb_pos_fourier(pos_t.view(1, 1, 1,-1)).to(x.dtype)

        # Encoder.
        x = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)
        x = self.conv_in(x)

        for _, block in self.enc.items():
            x = block(x, emb, pos_emb)

        # Decoder.
        for _, block in self.dec.items():
            x = block(x, emb, pos_emb)

        x = self.conv_out(x, gain=self.out_gain)
        D_x = c_skip * x_in + c_out * unpatchify(x.float())

        # Training uncertainty, if requested.
        if return_logvar:
            logvar = self.logvar_linear(self.logvar_fourier(c_noise)).view(-1, 1, 1, 1)
            return D_x, logvar.float()
        
        return D_x
    
    def get_class_embeddings(self, class_labels):
        return self.emb_label(normalize(class_labels).to(device=self.device, dtype=self.dtype))
    
    @torch.no_grad()
    def normalize_weights(self):
        for module in self.modules():
            if isinstance(module, MPConv):
                module.normalize_weights()