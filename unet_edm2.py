# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Improved diffusion model architecture proposed in the paper
"Analyzing and Improving the Training Dynamics of Diffusion Models"."""

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from dual_diffusion_utils import torch_compile

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

"""
def resample(x, f=[1,1], mode='keep'):
    if mode == 'keep':
        return x
    f = np.float32(f)
    assert f.ndim == 1 and len(f) % 2 == 0
    pad = (len(f) - 1) // 2
    f = f / f.sum()
    f = np.outer(f, f)[np.newaxis, np.newaxis, :, :]
    #f = misc.const_like(x, f)
    f = torch.tensor(f, device=x.device, dtype=x.dtype)
    c = x.shape[1]
    if mode == 'down':
        return torch.nn.functional.conv2d(x, f.tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,))
    assert mode == 'up'
    return torch.nn.functional.conv_transpose2d(x, (f * 4).tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,))
"""

def resample(x, mode="keep"):
    if mode == "keep":
        return x
    elif mode == 'down':
        return torch.nn.functional.avg_pool2d(x, 2) # should be multiplied by 2 to be magnitude preserving,
    elif mode == 'up':                              # however pixel norm is applied after downsampling so it doesn't matter
        return torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")

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

    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        
        #self.register_buffer('freqs', 2 * np.pi * torch.randn(num_channels) * bandwidth)
        #self.register_buffer('phases', 2 * np.pi * torch.rand(num_channels))

        # smoother inner product space with less overlap
        self.register_buffer('freqs', (bandwidth * 2**0.5 * 2*torch.pi) * ((torch.arange(num_channels)+0.5) / num_channels).erfinv())
        self.register_buffer('phases', torch.pi/2 * (torch.arange(num_channels) % 2 == 0).float())

    def forward(self, x):
        y = x.float().ger(self.freqs.float()) + self.phases.float()
        return (y.cos() * np.sqrt(2)).to(x.dtype)

#----------------------------------------------------------------------------
# Magnitude-preserving convolution or fully-connected layer (Equation 47)
# with force weight normalization (Equation 66).

class MPConv(torch.nn.Module):

    __constants__ = ["in_channels", "out_channels", "weight_gain"]

    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *kernel))
        self.weight_gain = np.sqrt(self.weight[0].numel())

    """
    def forward(self, x, gain=1):
        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w)) # forced weight normalization
        w = normalize(w) # traditional weight normalization
        w = w * (gain / np.sqrt(w[0].numel())) # magnitude-preserving scaling
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1]//2,))
    """

    def forward(self, x, gain=1):

        w = self.weight * (gain / self.weight_gain)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4

        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1]//2,))

    @torch.no_grad()
    def normalize_weights(self):
        self.weight.copy_(normalize(self.weight))

#----------------------------------------------------------------------------
# U-Net encoder/decoder block with optional self-attention (Figure 21).

class Block(torch.nn.Module):

    __constants__ = ["out_channels", "pos_channels", "flavor", "resample_mode", "num_heads", "dropout", "res_balance", "attn_balance", "clip_act"]

    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        emb_channels,                   # Number of embedding channels.
        pos_channels,                   # Number of positional embedding channels.
        flavor              = 'enc',    # Flavor: 'enc' or 'dec'.
        resample_mode       = 'keep',   # Resampling: 'keep', 'up', or 'down'.
        #resample_filter     = [1,1],    # Resampling filter.
        attention           = False,    # Include self-attention?
        channels_per_head   = 64,       # Number of channels per attention head.
        dropout             = 0,        # Dropout probability.
        res_balance         = 0.3,      # Balance between main branch (0) and residual branch (1).
        attn_balance        = 0.3,      # Balance between main branch (0) and self-attention (1).
        clip_act            = 256,      # Clip output activations. None = do not clip.
    ):
        super().__init__()

        if attention:
            if (out_channels % channels_per_head != 0) or ((out_channels + pos_channels) % channels_per_head != 0):
                raise ValueError(f'Number of output channels {out_channels} must be divisible by the number of channels per head {channels_per_head}.')

        self.out_channels = out_channels
        self.pos_channels = pos_channels
        self.flavor = flavor
        #self.resample_filter = resample_filter
        self.resample_mode = resample_mode
        self.num_heads = out_channels // channels_per_head if attention else 0
        self.dropout = dropout
        self.res_balance = res_balance
        self.attn_balance = attn_balance
        self.clip_act = clip_act
        self.emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.conv_res0 = MPConv(out_channels if flavor == 'enc' else in_channels, out_channels, kernel=[3,3])
        self.emb_linear = MPConv(emb_channels, out_channels, kernel=[]) if emb_channels != 0 else None
        self.conv_res1 = MPConv(out_channels, out_channels, kernel=[3,3])
        self.conv_skip = MPConv(in_channels, out_channels, kernel=[1,1]) if in_channels != out_channels else None

        if self.num_heads != 0:
            #self.attn_qkv = MPConv(out_channels, out_channels * 3, kernel=[1,1])
            #self.attn_proj = MPConv(out_channels, out_channels, kernel=[1,1])
            self.attn_qk = MPConv(out_channels + pos_channels, (out_channels + pos_channels) * 2, kernel=[1,1])
            self.attn_v = MPConv(out_channels, out_channels, kernel=[1,1])
            self.attn_proj = MPConv(out_channels, out_channels, kernel=[1,1])
        else:
            self.attn_qk = None
            self.attn_v = None
            self.attn_proj = None

    def forward(self, x, emb, format, t_ranges):
        # Main branch.
        #x = resample(x, f=self.resample_filter, mode=self.resample_mode)
        x = resample(x, mode=self.resample_mode)

        if self.flavor == 'enc':
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1) # pixel norm

        # Residual branch.
        y = self.conv_res0(mp_silu(x))
        if self.emb_linear is not None:
            c = self.emb_linear(emb, gain=self.emb_gain) + 1
            y = mp_silu(y * c.unsqueeze(2).unsqueeze(3).to(y.dtype))
        #if self.training and self.dropout != 0:
        #    y = torch.nn.functional.dropout(y, p=self.dropout)
        if self.dropout != 0: # magnitude preserving fix for dropout
            if self.training:
                y = torch.nn.functional.dropout(y, p=self.dropout)
            else:
                y *= 1 - self.dropout

        y = self.conv_res1(y)

        # Connect the branches.
        if self.flavor == 'dec' and self.conv_skip is not None:
            x = self.conv_skip(x)
        x = mp_sum(x, y, t=self.res_balance)

        # Self-attention.
        if self.num_heads != 0:
            
            #y = y.reshape(y.shape[0], self.num_heads, -1, 3, y.shape[2] * y.shape[3])
            #q, k, v = normalize(y, dim=2).unbind(3) # pixel norm & split
            #w = torch.einsum('nhcq,nhck->nhqk', q, k / np.sqrt(q.shape[2])).softmax(dim=3)
            #y = torch.einsum('nhqk,nhck->nhcq', w, v)
            #y = self.attn_proj(y.reshape(*x.shape))
            #x = mp_sum(x, y, t=self.attn_balance)

            # faster self-attention with torch sdp, qk separated for positional embedding
            if self.pos_channels > 0:
                qk = torch.cat((x, format.get_positional_embedding(x, t_ranges, mode="fourier", num_fourier_channels=self.pos_channels)), dim=1)
            else:
                qk = x

            qk = self.attn_qk(qk)
            qk = qk.reshape(qk.shape[0], self.num_heads, -1, 2, y.shape[2] * y.shape[3])
            q, k = normalize(qk, dim=2).unbind(3)

            v = self.attn_v(x)
            v = v.reshape(v.shape[0], self.num_heads, -1, y.shape[2] * y.shape[3])
            v = normalize(v, dim=2)

            y = torch.nn.functional.scaled_dot_product_attention(q.transpose(-1, -2),
                                                                 k.transpose(-1, -2),
                                                                 v.transpose(-1, -2)).transpose(-1, -2)
            y = self.attn_proj(y.reshape(*x.shape))
            x = mp_sum(x, y, t=self.attn_balance)

        # Clip activations.
        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)
        return x

#----------------------------------------------------------------------------
# EDM2 U-Net model (Figure 21).

class UNet(ModelMixin, ConfigMixin):

    __constants__ = ["label_dim", "training", "dtype", "label_dropout", "label_balance",
                     "dropout", "concat_balance", "sigma_max", "sigma_min", "sigma_data"]

    @register_to_config
    def __init__(self,
        in_channels = 4,                    # Number of input channels.
        out_channels = 4,                   # Number of output channels.
        pos_channels = 0,                   # Number of positional embedding channels for attention.
        logvar_channels = 128,              # Number of channels for training uncertainty estimation.
        use_t_ranges = False,
        channels_per_head = 64,             # Number of channels per attention head.
        label_dim = 0,                      # Class label dimensionality. 0 = unconditional.
        label_dropout = 0.1,                # Dropout probability for class labels. 
        dropout = 0,                        # Dropout probability.
        model_channels       = 192,         # Base multiplier for the number of channels.
        channel_mult         = [1,2,3,4],   # Per-resolution multipliers for the number of channels.
        channel_mult_noise   = None,        # Multiplier for noise embedding dimensionality. None = select based on channel_mult.
        channel_mult_emb     = None,        # Multiplier for final embedding dimensionality. None = select based on channel_mult.
        num_layers_per_block = 3,           # Number of residual blocks per resolution.
        attn_levels          = [2,3],       # List of resolutions with self-attention.
        label_balance        = 0.5,         # Balance between noise embedding (0) and class embedding (1).
        concat_balance       = 0.5,         # Balance between skip connections (0) and main path (1).
        sigma_max = 80.,                    # Expected max noise std
        sigma_min = 0.002,                  # Expected min noise std
        sigma_data = 0.5,                   # Expected data / input sample std
        #**block_kwargs,                    # Arguments for Block.
        last_global_step = 0,               # Only used to track training progress in config.
    ):
        super().__init__()

        block_kwargs = {"channels_per_head": channels_per_head, "dropout": dropout}

        cblock = [int(model_channels * x) for x in channel_mult]
        cnoise = int(model_channels * channel_mult_noise) if channel_mult_noise is not None else max(cblock)#cblock[0]
        cemb = int(model_channels * channel_mult_emb) if channel_mult_emb is not None else max(cblock)
        clogvar = logvar_channels
        cpos = pos_channels

        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sigma_data = sigma_data

        self.label_dim = label_dim
        self.label_dropout = label_dropout
        self.label_balance = label_balance
        self.dropout = dropout
        self.concat_balance = concat_balance
        self.out_gain = torch.nn.Parameter(torch.zeros([]))

        # Embedding.
        self.emb_fourier = MPFourier(cnoise)
        self.emb_noise = MPConv(cnoise, cemb, kernel=[])
        self.emb_label = MPConv(label_dim, cemb, kernel=[]) if label_dim != 0 else None
        self.emb_label_unconditional = MPConv(1, cemb, kernel=[]) if label_dim != 0 else None

        # Training uncertainty estimation.
        self.logvar_fourier = MPFourier(clogvar)
        self.logvar_linear = MPConv(clogvar, 1, kernel=[])

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels + 2 + int(use_t_ranges) # 1 const channel, 1 positional channels, plus 1 optional position channel
        for level, channels in enumerate(cblock):
            
            if level == 0:
                cin = cout
                cout = channels
                self.enc[f'conv_in'] = MPConv(cin, cout, kernel=[3,3])
            else:
                self.enc[f'block{level}_down'] = Block(cout, cout, cemb, cpos,
                                                       flavor='enc', resample_mode='down', **block_kwargs)
            for idx in range(num_layers_per_block):
                cin = cout
                cout = channels
                self.enc[f'block{level}_layer{idx}'] = Block(cin, cout, cemb, cpos,
                                                             flavor='enc', attention=(level in attn_levels), **block_kwargs)

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        skips = [block.out_channels for block in self.enc.values()]
        for level, channels in reversed(list(enumerate(cblock))):
            
            if level == len(cblock) - 1:
                self.dec[f'block{level}_in0'] = Block(cout, cout, cemb, cpos,
                                                      flavor='dec', attention=True, **block_kwargs)
                self.dec[f'block{level}_in1'] = Block(cout, cout, cemb, cpos,
                                                      flavor='dec', **block_kwargs)
            else:
                self.dec[f'block{level}_up'] = Block(cout, cout, cemb, cpos,
                                                     flavor='dec', resample_mode='up', **block_kwargs)
            for idx in range(num_layers_per_block + 1):
                cin = cout + skips.pop()
                cout = channels
                self.dec[f'block{level}_layer{idx}'] = Block(cin, cout, cemb, cpos,
                                                             flavor='dec', attention=(level in attn_levels), **block_kwargs)
        self.conv_out = MPConv(cout, out_channels, kernel=[3,3])

    @torch_compile(fullgraph=True)
    def forward(self, x_in, sigma, class_embeddings, t_ranges, format, return_logvar=False):

        sigma = sigma.view(-1, 1, 1, 1)

        # Preconditioning weights.
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = (sigma.flatten().log() / 4).to(self.dtype)

        # Run the model.
        x = (c_in * x_in).to(self.dtype)

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
            emb = mp_sum(emb, class_embeddings.to(self.dtype), t=self.label_balance)
        emb = mp_silu(emb)

        # Encoder.
        x = torch.cat((x, torch.ones_like(x[:, :1]),
                       format.get_positional_embedding(x, t_ranges, mode="linear")), dim=1)
        skips = []
        for name, block in self.enc.items():
            x = block(x) if 'conv' in name else block(x, emb, format, t_ranges)
            skips.append(x)

        # Decoder.
        for name, block in self.dec.items():
            if 'layer' in name:
                x = mp_cat(x, skips.pop(), t=self.concat_balance)
            x = block(x, emb, format, t_ranges)
        x = self.conv_out(x, gain=self.out_gain)

        D_x = c_skip * x_in + c_out * x.float()

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

if __name__ == "__main__": # fourier embedding inner product test

    from dual_diffusion_utils import save_raw, save_raw_img
    from dotenv import load_dotenv
    import os

    load_dotenv(override=True)
    
    steps = 200
    cnoise = 192*4
    sigma_max = 80.
    sigma_min = 0.002
    mpfourier_mode = "gaussian"

    emb_fourier = MPFourier(cnoise)
    noise_label = torch.linspace(sigma_max, sigma_min, steps).log() / 4

    emb = emb_fourier(noise_label)
    inner_products = (emb.view(1, steps, cnoise) * emb.view(steps, 1, cnoise)).sum(dim=2)

    debug_path = os.environ.get("DEBUG_PATH", None)
    if debug_path is not None:    
        save_raw(inner_products / inner_products.amax(), os.path.join(debug_path, "fourier_inner_products.raw"))
        save_raw_img(inner_products, os.path.join(debug_path, "fourier_inner_products.png"))

        coverage = inner_products.sum(dim=0)
        save_raw(coverage / coverage.amax(), os.path.join(debug_path, "fourier_inner_products_coverage.raw"))