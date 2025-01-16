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

from typing import Optional, Union, Literal
from dataclasses import dataclass

import torch

from modules.formats.format import DualDiffusionFormat
from modules.vaes.vae import DualDiffusionVAEConfig, DualDiffusionVAE, DegenerateDistribution
from modules.mp_tools import normalize, resample_3d, mp_silu, mp_sum


@dataclass
class DualDiffusionVAE_EDM2_D1_Config(DualDiffusionVAEConfig):

    model_channels: int       = 32           # Base multiplier for the number of channels.
    latent_channels: int      = 4
    channel_mult: list[int]   = (1,2,3,5)    # Per-resolution multipliers for the number of channels.
    channel_mult_emb: Optional[int] = 5      # Multiplier for final embedding dimensionality.
    num_layers_per_block: int = 3            # Number of resnet blocks per resolution.
    res_balance: float        = 0.3          # Balance between main branch (0) and residual branch (1).
    mlp_multiplier: int   = 2                # Multiplier for the number of channels in the MLP.
    mlp_groups: int       = 1                # Number of groups for the MLPs.
    noise_multiplier: int = 2                # augments the amount of noise for the diffusion decoder

class MPConv(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                 kernel: tuple[int, int], groups: int = 1, stride: int = 1,
                 disable_weight_norm: bool = False) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.stride = stride
        self.disable_weight_norm = disable_weight_norm
        
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel))

    def forward(self, x: torch.Tensor, gain: Union[float, torch.Tensor] = 1.) -> torch.Tensor:
        
        w = self.weight.float()
        if self.training == True and self.disable_weight_norm == False:
            w = normalize(w) # traditional weight normalization
            
        w = w * (gain / w[0].numel()**0.5) # magnitude-preserving scaling
        w = w.to(x.dtype)

        if w.ndim == 2:
            return x @ w.t()
        
        if w.ndim == 5:
            if w.shape[-3] == 2:
                x = torch.cat((x, x[:, :, 0:1]), dim=2)
            return torch.nn.functional.conv3d(x, w, padding=(0, w.shape[-2]//2, w.shape[-1]//2), groups=self.groups, stride=self.stride)
        else:
            return torch.nn.functional.conv2d(x, w, padding=(w.shape[-2]//2, w.shape[-1]//2), groups=self.groups, stride=self.stride)

    @torch.no_grad()
    def normalize_weights(self):
        if self.disable_weight_norm == False:
            self.weight.copy_(normalize(self.weight))

class Block(torch.nn.Module):

    def __init__(self,
        level: int,                             # Resolution level.
        in_channels: int,                       # Number of input channels.
        out_channels: int,                      # Number of output channels.
        emb_channels: int,                      # Number of embedding channels.
        flavor: Literal["enc", "dec"] = "enc",
        resample_mode: Literal["keep", "up", "down"] = "keep",
        dropout: float         = 0.,       # Dropout probability.
        res_balance: float     = 0.5,      # Balance between main branch (0) and residual branch (1).
        clip_act: float        = 256,      # Clip output activations. None = do not clip.
        mlp_multiplier: int    = 2,        # Multiplier for the number of channels in the MLP.
        mlp_groups: int        = 1,        # Number of groups for the MLP.
        noise_multiplier: float = 1

    ) -> None:
        super().__init__()

        self.level = level
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_mode = resample_mode
        self.dropout = dropout
        self.res_balance = res_balance
        self.clip_act = clip_act
        self.noise_multiplier = noise_multiplier
        
        self.conv_res0 = MPConv(out_channels if flavor == "enc" else in_channels,
                                out_channels * mlp_multiplier, kernel=(2,3,3), groups=mlp_groups)
        self.conv_res1 = MPConv(out_channels * mlp_multiplier, out_channels, kernel=(2,3,3), groups=mlp_groups)
        self.conv_skip = MPConv(in_channels, out_channels, kernel=(2,3,3), groups=1) if in_channels != out_channels else None

        self.emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.emb_linear = MPConv(emb_channels, out_channels * mlp_multiplier,
                                 kernel=(1,1,1), groups=1) if emb_channels != 0 else None

        self.error_logvar = torch.nn.Parameter(torch.zeros([])) if flavor == "dec" else None

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        
        x = resample_3d(x, mode=self.resample_mode)

        #if self.flavor == "dec":
        #    noise_std = 0.25#(self.error_logvar/2).exp().detach() * self.noise_multiplier
        #    noise = torch.randn_like(x)
        #    x = x + noise * noise_std

        if self.flavor == "enc":
            if self.conv_skip is not None:
                x = self.conv_skip(x)

        y = self.conv_res0(mp_silu(x))

        c = self.emb_linear(emb, gain=self.emb_gain) + 1.
        y = mp_silu(y * c)

        if self.dropout != 0 and self.training == True: # magnitude preserving fix for dropout
            y = torch.nn.functional.dropout(y, p=self.dropout) * (1. - self.dropout)**0.5

        y = self.conv_res1(y)

        if self.flavor == "dec" and self.conv_skip is not None:
            x = self.conv_skip(x)

        #if self.flavor == "dec":
        #    x = x - y * noise_std
        #else:
        #    x = mp_sum(x, y, t=self.res_balance)
        x = mp_sum(x, y, t=self.res_balance)

        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)
        return x

class AutoencoderKL_EDM2_D1(DualDiffusionVAE):

    def __init__(self, config: DualDiffusionVAE_EDM2_D1_Config) -> None:
        super().__init__()
        self.config = config

        block_kwargs = {"dropout": config.dropout,
                        "mlp_multiplier": config.mlp_multiplier,
                        "mlp_groups": config.mlp_groups,
                        "res_balance": config.res_balance,
                        "noise_multiplier": config.noise_multiplier} 
        
        cblock = [config.model_channels * x for x in config.channel_mult]
        cemb = config.model_channels * config.channel_mult_emb if config.channel_mult_emb is not None else max(cblock)
        cemb *= self.config.mlp_multiplier

        self.num_levels = len(config.channel_mult)
        
        # Embedding.
        self.emb_label_enc = MPConv(config.in_channels_emb, cemb, kernel=())
        self.emb_label_dec = MPConv(config.in_channels_emb, cemb, kernel=())
        self.emb_dim = cemb
        
        # Encoder.
        self.enc = torch.nn.ModuleDict()
        self.dec = {}
        cout = 1

        for level, channels in enumerate(cblock):
            
            if level == 0:
                cin = cout
                cout = channels
                self.enc[f"conv_in"] = Block(level, cin, cout, cemb,
                                                flavor="enc", **block_kwargs)
                self.dec[f"conv_out"] = Block(level, cout, cin, cemb,
                                                flavor="dec", **block_kwargs)
            else:
                self.enc[f"block{level}_down"] = Block(level, cout, cout, cemb,
                                                flavor="enc", resample_mode="down", **block_kwargs)
                self.dec[f"block{level}_up"] = Block(level, cout, cout, cemb,
                                                flavor="dec", resample_mode="up", **block_kwargs)
                
            for idx in range(config.num_layers_per_block):
                cin = cout
                cout = channels
                self.enc[f"block{level}_layer{idx}"] = Block(level, cin, cout, cemb,
                                                            flavor="enc", **block_kwargs)
                self.dec[f"block{level}_layer{idx}"] = Block(level, cout, cin, cemb,
                                                            flavor="dec", **block_kwargs)

        self.enc["conv_latents_out"] = Block(level, cout, config.latent_channels, cemb, flavor="enc", **block_kwargs)
        self.dec["conv_latents_in"] = Block(level, config.latent_channels, cout, cemb, flavor="dec", **block_kwargs)

        self.dec = torch.nn.ModuleDict({k:v for k,v in reversed(self.dec.items())})

    def get_embeddings(self, emb_in: torch.Tensor) -> torch.Tensor:
        enc_embeddings = self.emb_label_enc(normalize(emb_in).to(device=self.device, dtype=self.dtype))
        dec_embeddings  = self.emb_label_dec(normalize(emb_in).to(device=self.device, dtype=self.dtype))
        return enc_embeddings, dec_embeddings
    
    def get_recon_loss_logvar(self) -> torch.Tensor:
        raise NotImplementedError()
    
    def get_latent_shape(self, sample_shape: Union[torch.Size, tuple[int, int, int, int]]) -> torch.Size:
        if len(sample_shape) == 4:
            return (sample_shape[0], self.config.latent_channels, self.config.in_channels,
                    sample_shape[2] // 2 ** (self.num_levels-1),
                    sample_shape[3] // 2 ** (self.num_levels-1))
        else:
            raise ValueError(f"Invalid sample shape: {sample_shape}")
        
    def get_sample_shape(self, latent_shape: Union[torch.Size, tuple[int, int, int, int]]) -> torch.Size:
        if len(latent_shape) == 5:
            return (latent_shape[0], self.config.out_channels, self.config.in_num_freqs,
                    latent_shape[4] * 2 ** (self.num_levels-1))
        else:
            raise ValueError(f"Invalid latent shape: {latent_shape}")
        
    def encode(self, x: torch.Tensor, embeddings: torch.Tensor,
               format: DualDiffusionFormat, return_hidden_states: bool = False) -> DegenerateDistribution:
        raise NotImplementedError()
    
        x = x.unsqueeze(1)
        embeddings = embeddings[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # keep record of encoder inputs
        if return_hidden_states == True:
            hidden_states: list[torch.Tensor] = []

        for name, block in self.enc.items():
            
            if return_hidden_states == True:
                hidden_states.append(x)
                
            if "conv" in name:
                x = block(torch.cat((x, torch.ones_like(x[:, :1])), dim=1), embeddings)
            else:
                x = block(x, embeddings)
        
        if return_hidden_states == True:
            hidden_states.append(x)

        latents = self.conv_latents_out(x, embeddings) * self.latents_out_gain

        if return_hidden_states == True:
            return DegenerateDistribution(latents), hidden_states
        else:
            return DegenerateDistribution(latents)
    
    def decode(self, x: torch.Tensor, embeddings: torch.Tensor, format: DualDiffusionFormat,
               return_hidden_states: bool = False, enc_states: list[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError()
        diff_embeddings = embeddings[1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # keep record of decoder outputs
        if return_hidden_states == True:
            hidden_states: list[torch.Tensor] = []

        x = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)
        x = self.conv_latents_in(x, diff_embeddings)

        if return_hidden_states == True:
            hidden_states.append((x, self.conv_latents_in.error_logvar))

        for name, block in self.dec.items():
            
            x = block(x, diff_embeddings)
            if return_hidden_states == True:
                hidden_states.append((x, block.error_logvar))

        output: torch.Tensor = self.conv_out(x, diff_embeddings) * self.out_gain

        if return_hidden_states == True:
            hidden_states.append((output, self.conv_out.error_logvar))
            return output.squeeze(1), hidden_states
        else:
            return output.squeeze(1)

    def encode_train(self, x: torch.Tensor, embeddings: torch.Tensor) -> list[torch.Tensor]:
    
        x_in = x.unsqueeze(1)
        enc_embeddings = embeddings[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        enc_states: list[torch.Tensor] = []

        for name, block in self.enc.items():
            x_out = block(x_in, enc_embeddings)
            enc_states.append((x_in, x_out))
            x_in = x_out

        return enc_states
    
    def decode_train(self, enc_states: list[torch.Tensor], embeddings: torch.Tensor) -> list[torch.Tensor]:
        
        dec_embeddings = embeddings[1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        dec_states: list[torch.Tensor] = []

        for (name, block), (x_in, x_out) in zip(self.dec.items(), reversed(enc_states)):
            x_in = block(x_out, dec_embeddings)
            dec_states.append((x_out, x_in, block.error_logvar))

        return dec_states
    
    def forward(self, samples: torch.Tensor, embeddings: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:

        embeddings = [embedding.to(dtype=torch.bfloat16) for embedding in embeddings]

        enc_states = self.encode_train(samples, embeddings)
        dec_states = self.decode_train(enc_states, embeddings)

        return enc_states, dec_states