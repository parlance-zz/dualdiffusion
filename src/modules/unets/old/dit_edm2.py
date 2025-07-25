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

import torch

from modules.mp_tools import MPConv, MPFourier, mp_silu, mp_sum, mp_cat, normalize

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
        channels_per_head   = 128,      # Number of channels per attention head.
        headroom = 1,
        dropout             = 0,        # Dropout probability.
        res_balance         = 0.5,      # Balance between main branch (0) and residual branch (1).
        attn_balance        = 0.5,      # Balance between main branch (0) and self-attention (1).
        clip_act            = 256,      # Clip output activations. None = do not clip.
    ):
        super().__init__()

        if out_channels % channels_per_head != 0:
            raise ValueError(f'Number of output channels {out_channels} must be divisible by the number of channels per head {channels_per_head}.')

        self.out_channels = out_channels
        self.pos_channels = pos_channels
        self.flavor = flavor
        self.num_heads = (out_channels * headroom) // channels_per_head
        self.dropout = dropout
        self.res_balance = res_balance
        self.attn_balance = attn_balance
        self.clip_act = clip_act
        self.emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.conv_res0 = MPConv(out_channels if flavor == 'enc' else in_channels, out_channels, kernel=[1,3])
        self.emb_linear = MPConv(emb_channels, out_channels, kernel=[]) if emb_channels != 0 else None
        self.conv_res1 = MPConv(out_channels, out_channels, kernel=[1,3])
        self.conv_skip = MPConv(in_channels, out_channels, kernel=[1,1]) if in_channels != out_channels else None

        #self.attn_qk = MPConv(out_channels + pos_channels, out_channels * 2 * headroom, kernel=[1,1])
        self.attn_qk = MPConv(out_channels * 2, out_channels * 2 * headroom, kernel=[1,1])
        self.attn_v = MPConv(out_channels, out_channels, kernel=[1,1])
        self.attn_proj = MPConv(out_channels, out_channels, kernel=[1,1])

    def forward(self, x, emb, pos_emb):
        # Main branch.

        if self.flavor == 'enc':
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1) # pixel norm

        # Residual branch.
        y = self.conv_res0(mp_silu(x))
        if self.emb_linear is not None:
            c = self.emb_linear(emb, gain=self.emb_gain) + 1
            y = mp_silu(y * c.unsqueeze(2).unsqueeze(3).to(y.dtype))

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

        #pos_emb = x * pos_emb
        qk_x = mp_cat(x, x * pos_emb) #torch.cat((x, pos_emb), dim=1)
        #qk_x = torch.cat((x * pos_emb[:, ::2], x * pos_emb[:, 1::2]), dim=1)
        #qk_x = x * 0.5**0.5 + pos_emb * 0.5**0.5
        qk = self.attn_qk(qk_x)
        qk = qk.reshape(qk.shape[0], self.num_heads, -1, 2, y.shape[2] * y.shape[3])
        q, k = normalize(qk, dim=2).unbind(3)

        #self.w = torch.einsum('nhcq,nhck->nhqk', q, k / np.sqrt(q.shape[2])).softmax(dim=3)[0]

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
                     "dropout", "concat_balance", "sigma_max", "sigma_min", "sigma_data", "in_channels", "out_channels", "pos_channels"]

    @register_to_config
    def __init__(self,
        in_channels = 4,                    # Number of input channels.
        out_channels = 4,                   # Number of output channels.
        pos_channels = 768,                 # Number of positional embedding channels for attention.
        logvar_channels = 128,              # Number of channels for training uncertainty estimation.
        use_t_ranges = True,
        channels_per_head = 128,            # Number of channels per attention head.
        label_dim = 0,                      # Class label dimensionality. 0 = unconditional.
        label_dropout = 0.1,                # Dropout probability for class labels. 
        dropout = 0,                        # Dropout probability.
        model_channels       = 1024,        # Base multiplier for the number of channels.
        channel_mult         = [1,1,1,1],   # Per-resolution multipliers for the number of channels.
        channel_mult_noise   = None,        # Multiplier for noise embedding dimensionality. None = select based on channel_mult.
        channel_mult_emb     = None,        # Multiplier for final embedding dimensionality. None = select based on channel_mult.
        num_layers_per_block = 1,           # Number of residual blocks per resolution.
        label_balance        = 0.5,         # Balance between noise embedding (0) and class embedding (1).
        concat_balance       = 0.5,         # Balance between skip connections (0) and main path (1).
        sigma_max = 200.,                   # Expected max noise std
        sigma_min = 0.03,                   # Expected min noise std
        sigma_data = 1.,                    # Expected data / input sample std
        #**block_kwargs,                    # Arguments for Block.
        last_global_step = 0,               # Only used to track training progress in config.
    ):
        super().__init__()

        assert use_t_ranges and (model_channels + pos_channels) % channels_per_head == 0 and pos_channels > 0 and pos_channels % 2 == 0
        block_kwargs = {"channels_per_head": channels_per_head, "dropout": dropout}

        cblock = [int(model_channels * x) for x in channel_mult]
        cnoise = int(model_channels * channel_mult_noise) if channel_mult_noise is not None else max(cblock)#cblock[0]
        cemb = int(model_channels * channel_mult_emb) if channel_mult_emb is not None else max(cblock)
        clogvar = logvar_channels
        cpos = pos_channels

        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sigma_data = sigma_data

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pos_channels = pos_channels
        self.label_dim = label_dim
        self.label_dropout = label_dropout
        self.label_balance = label_balance
        self.dropout = dropout
        self.concat_balance = concat_balance
        self.out_gain = torch.nn.Parameter(torch.zeros([]))

        # Embedding.
        self.emb_fourier = MPFourier(cnoise, bandwidth=1.414)
        self.emb_noise = MPConv(cnoise, cemb, kernel=[])
        self.emb_label = MPConv(label_dim, cemb, kernel=[]) if label_dim != 0 else None
        self.emb_label_unconditional = MPConv(1, cemb, kernel=[]) if label_dim != 0 else None
        self.emb_pos_fourier = MPFourier(cpos, bandwidth=100)

        # Training uncertainty estimation.
        self.logvar_fourier = MPFourier(clogvar)
        self.logvar_linear = MPConv(clogvar, 1, kernel=[], disable_weight_normalization=True)

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        #cout = ((32 * 3) // 2 + 1) * 2 * in_channels + 1
        cout = 32*in_channels + 1
        for level, channels in enumerate(cblock):
            
            if level == 0:
                cin = cout
                cout = channels
                self.enc[f'conv_in'] = MPConv(cin, cout, kernel=[1,3])
            else:
                self.enc[f'block{level}_in'] = Block(cout, cout, cemb, cpos,
                                                       flavor='enc', **block_kwargs)
            for idx in range(num_layers_per_block):
                cin = cout
                cout = channels
                self.enc[f'block{level}_layer{idx}'] = Block(cin, cout, cemb, cpos,
                                                             flavor='enc', **block_kwargs)

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        skips = [block.out_channels for block in self.enc.values()]
        for level, channels in reversed(list(enumerate(cblock))):
            
            self.dec[f'block{level}_in'] = Block(cout, cout, cemb, cpos,
                                                    flavor='dec', **block_kwargs)
            for idx in range(num_layers_per_block + 1):
                cin = cout + skips.pop()
                cout = channels
                self.dec[f'block{level}_layer{idx}'] = Block(cin, cout, cemb, cpos,
                                                             flavor='dec', **block_kwargs)
                
        #self.conv_out = MPConv(cout, ((32 * 3) // 2 + 1) * 2 * out_channels, kernel=[1,3])
        self.conv_out = MPConv(cout, 32 * out_channels, kernel=[1,3])

    def forward(self, x_in, sigma, class_embeddings, t_ranges, format, return_logvar=False):

        with torch.no_grad():
            sigma = sigma.view(-1, 1, 1, 1)

            # Preconditioning weights.
            c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
            c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
            c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
            c_noise = (sigma.flatten().log() / 4).to(self.dtype)
            
            x = self.patchify(c_in * x_in).to(self.dtype)
            
            pos_t = torch.linspace(0, 1, x.shape[3], device=self.device).view(1, -1) * (t_ranges[:, 1:2] - t_ranges[:, 0:1]) + t_ranges[:, 0:1]
            #pos_t = torch.linspace(-3.5714285714/2, 3.5714285714/2, x.shape[3], device=self.device).view(1, -1)
            pos_emb = self.emb_pos_fourier(pos_t.view(pos_t.shape[0], 1, 1,-1)).to(self.dtype)

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
            emb = mp_sum(emb, class_embeddings.to(emb.dtype), t=self.label_balance)
        emb = mp_silu(emb)

        # Encoder.
        x = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)

        skips = []
        for name, block in self.enc.items():
            x = block(x) if 'conv' in name else block(x, emb, pos_emb)
            skips.append(x)

        # Decoder.
        for name, block in self.dec.items():
            if 'layer' in name:
                x = mp_cat(x, skips.pop(), t=self.concat_balance)
            x = block(x, emb, pos_emb)
        x = self.conv_out(x, gain=self.out_gain)

        D_x = c_skip * x_in + c_out * self.unpatchify(x.float())

        # Training uncertainty, if requested.
        if return_logvar:
            logvar = self.logvar_linear(self.logvar_fourier(c_noise)).view(-1, 1, 1, 1)
            return D_x, logvar.float()
        
        return D_x
    
    @torch.no_grad()
    def patchify(self, x):
        #x = torch.nn.functional.pad(x, (0, 0, 32, 32)).float()
        #x = torch.view_as_real(torch.fft.rfft(x, dim=2, norm="ortho")).permute(0, 1, 2, 4, 3)
        #return x.reshape(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3], 1, x.shape[4])
        return x.reshape(x.shape[0], x.shape[1]*x.shape[2], 1, x.shape[3])

    #@torch.no_grad()
    def unpatchify(self, x):
        #x = x.view(x.shape[0], self.out_channels, (32*3)//2+1, 2, x.shape[3]).permute(0, 1, 2, 4, 3).contiguous()
        #return torch.fft.irfft(torch.view_as_complex(x), dim=2, norm="ortho")[:, :, 32:-32]
        return x.reshape(x.shape[0], self.out_channels, 32, x.shape[3])

    def get_class_embeddings(self, class_labels):
        return self.emb_label(normalize(class_labels).to(device=self.device, dtype=self.dtype))
    
    @torch.no_grad()
    def normalize_weights(self):
        for module in self.modules():
            if isinstance(module, MPConv):
                module.normalize_weights()