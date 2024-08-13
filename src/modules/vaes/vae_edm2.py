# MIT License
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

from typing import Optional
from dataclasses import dataclass

import torch

from modules.vaes.vae import DualDiffusionVAEConfig, DualDiffusionVAE
from modules.mp_tools import MPConv

@dataclass
class DualDiffusionVAE_EDM2Config(DualDiffusionVAEConfig):
    pass

class Block(torch.nn.Module):

    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        emb_channels,                   # Number of embedding channels.
        pos_channels,                   # Number of positional embedding channels.
        flavor              = 'enc',    # Flavor: 'enc' or 'dec'.
        resample_mode       = 'keep',   # Resampling: 'keep', 'up', or 'down'.
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

class AutoencoderKL_EDM2(DualDiffusionVAE):

    def __init__(self,
        channels_per_head = 64,             # Number of channels per attention head.
        model_channels       = 64,          # Base multiplier for the number of channels.
        channel_mult         = [1,2,3,5],   # Per-resolution multipliers for the number of channels.
        channel_mult_emb     = None,        # Multiplier for final embedding dimensionality. None = select based on channel_mult.
        num_layers_per_block = 3,           # Number of residual blocks per resolution.
        attn_levels          = [],          # List of resolutions with self-attention.
        midblock_decoder_attn = False,      # Whether to use attention in the first layer of the decoder.
        #**block_kwargs,                    # Arguments for Block.
        last_global_step = 0,               # Only used to track training progress in config.
    ):
        super().__init__()

        block_kwargs = {"channels_per_head": channels_per_head, "dropout": dropout}

        cblock = [int(model_channels * x) for x in channel_mult]
        if label_dim != 0:
            cemb = int(model_channels * channel_mult_emb) if channel_mult_emb is not None else max(cblock)
        else:
            cemb = 0

        self.label_dim = label_dim
        self.dropout = dropout
        self.target_snr = target_snr
        self.num_levels = len(channel_mult)

        target_noise_std = (1 / (self.target_snr**2 + 1))**0.5
        target_sample_std = (1 - target_noise_std**2)**0.5
        self.latents_out_gain = torch.nn.Parameter(torch.tensor(target_sample_std))
        self.out_gain = torch.nn.Parameter(torch.ones([]))
        
        # Embedding.
        self.emb_label = MPConv(label_dim, cemb, kernel=[]) if label_dim != 0 else None

        # Training uncertainty estimation.
        self.recon_loss_logvar = torch.nn.Parameter(torch.zeros(1))
        self.latents_logvar = torch.nn.Parameter(torch.zeros(1)) # currently unused
        
        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels + 2 # 1 extra const channel, 1 pos embedding channel
        for level, channels in enumerate(cblock):
            
            if level == 0:
                cin = cout
                cout = channels
                self.enc[f'conv_in'] = MPConv(cin, cout, kernel=[3,3])
            else:
                self.enc[f'block{level}_down'] = Block(cout, cout, cemb, 0,
                                                       flavor='enc', resample_mode='down', **block_kwargs)
            for idx in range(num_layers_per_block):
                cin = cout
                cout = channels
                self.enc[f'block{level}_layer{idx}'] = Block(cin, cout, cemb, 0,
                                                             flavor='enc', attention=(level in attn_levels), **block_kwargs)

        self.conv_latents_out = MPConv(cout, latent_channels, kernel=[3,3])
        self.conv_latents_in = MPConv(latent_channels + 2, cout, kernel=[3,3]) # 1 extra const channel, 1 pos embedding channel

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, channels in reversed(list(enumerate(cblock))):
            
            if level == len(cblock) - 1:
                self.dec[f'block{level}_in0'] = Block(cout, cout, cemb, 0,
                                                      flavor='dec',attention=midblock_decoder_attn, **block_kwargs)
                self.dec[f'block{level}_in1'] = Block(cout, cout, cemb, 0,
                                                      flavor='dec', **block_kwargs)
            else:
                self.dec[f'block{level}_up'] = Block(cout, cout, cemb, 0,
                                                     flavor='dec', resample_mode='up', **block_kwargs)
            for idx in range(num_layers_per_block + 1):
                cin = cout
                cout = channels
                self.dec[f'block{level}_layer{idx}'] = Block(cin, cout, cemb, 0,
                                                             flavor='dec', attention=(level in attn_levels), **block_kwargs)
        self.conv_out = MPConv(cout, out_channels, kernel=[3,3])