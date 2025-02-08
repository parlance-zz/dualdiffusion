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
from modules.mp_tools import MPConv3D, MPFourier, normalize, resample_3d, mp_silu, mp_sum, mp_cat
from utils.dual_diffusion_utils import tensor_4d_to_5d, tensor_5d_to_4d


@dataclass
class DiffusionAutoencoder_Config(DualDiffusionVAEConfig):

    # vae encoder/decoder params
    model_channels: int       = 32           # Base multiplier for the number of channels.
    latent_channels: int      = 4
    channel_mult: list[int]   = (1,2,4,4)    # Per-resolution multipliers for the number of channels.
    channel_mult_emb: Optional[int] = 4      # Multiplier for final embedding dimensionality.
    num_layers_per_block: int = 2            # Number of resnet blocks per resolution.
    res_balance: float        = 0.4          # Balance between main branch (0) and residual branch (1).
    mlp_multiplier: int   = 1                # Multiplier for the number of channels in the MLP.
    mlp_groups: int       = 1                # Number of groups for the MLPs.

    # diffusion decoder params
    ddec_model_channels: int  = 32                # Base multiplier for the number of channels.
    ddec_logvar_channels: int = 128               # Number of channels for training uncertainty estimation.
    ddec_channel_mult: list[int]    = (1,2,4,4)   # Per-resolution multipliers for the number of channels.
    ddec_channel_mult_noise: Optional[int] = None # Multiplier for noise embedding dimensionality.
    ddec_channel_mult_emb: Optional[int]   = None # Multiplier for final embedding dimensionality.
    ddec_num_layers_per_block: int = 2            # Number of resnet blocks per resolution.
    ddec_label_balance: float      = 0.5          # Balance between noise embedding (0) and class embedding (1).
    ddec_concat_balance: float     = 0.5          # Balance between skip connections (0) and main path (1).
    ddec_res_balance: float        = 0.3          # Balance between main branch (0) and residual branch (1).
    ddec_mlp_multiplier: int = 1                  # Multiplier for the number of channels in the MLP.
    ddec_mlp_groups: int     = 1                  # Number of groups for the MLPs.
    ddec_sigma_max:  float = 200.
    ddec_sigma_min:  float = 0.03
    ddec_sigma_data: float = 1.


class VAE_Block(torch.nn.Module):

    def __init__(self,
        level: int,                             # Resolution level.
        in_channels: int,                       # Number of input channels.
        out_channels: int,                      # Number of output channels.
        emb_channels: int,                      # Number of embedding channels.
        flavor: Literal["enc", "dec"] = "enc",
        resample_mode: Literal["keep", "up", "down"] = "keep",
        dropout: float         = 0.,       # Dropout probability.
        res_balance: float     = 0.3,      # Balance between main branch (0) and residual branch (1).
        clip_act: float        = 256,      # Clip output activations. None = do not clip.
        mlp_multiplier: int    = 1,        # Multiplier for the number of channels in the MLP.
        mlp_groups: int        = 1,        # Number of groups for the MLP.

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
        
        self.conv_res0 = MPConv3D(out_channels if flavor == "enc" else in_channels,
                                out_channels * mlp_multiplier, kernel=(2,3,3), groups=mlp_groups)
        self.conv_res1 = MPConv3D(out_channels * mlp_multiplier, out_channels, kernel=(2,3,3), groups=mlp_groups)
        self.conv_skip = MPConv3D(in_channels, out_channels, kernel=(1,1,1), groups=1) if in_channels != out_channels else None

        self.emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.emb_linear = MPConv3D(emb_channels, out_channels * mlp_multiplier,
                                 kernel=(1,1,1), groups=1) if emb_channels != 0 else None

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        
        x = resample_3d(x, mode=self.resample_mode)

        if self.flavor == "enc" and self.conv_skip is not None:
            x = self.conv_skip(x)

        y = self.conv_res0(mp_silu(x))

        c = self.emb_linear(emb, gain=self.emb_gain) + 1.
        y = mp_silu(y * c)

        if self.dropout != 0 and self.training == True: # magnitude preserving fix for dropout
            y = torch.nn.functional.dropout(y, p=self.dropout) * (1. - self.dropout)**0.5

        y = self.conv_res1(y)

        if self.flavor == "dec" and self.conv_skip is not None:
            x = self.conv_skip(x)

        x = mp_sum(x, y, t=self.res_balance)

        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)
        return x

class DDec_Block(torch.nn.Module):

    def __init__(self,
        level: int,                             # Resolution level.
        in_channels: int,                       # Number of input channels.
        out_channels: int,                      # Number of output channels.
        emb_channels: int,                      # Number of embedding channels.
        flavor: Literal["enc", "dec"] = "enc",
        resample_mode: Literal["keep", "up", "down"] = "keep",
        dropout: float         = 0.,       # Dropout probability.
        res_balance: float     = 0.3,      # Balance between main branch (0) and residual branch (1).
        clip_act: float        = 256,      # Clip output activations. None = do not clip.
        mlp_multiplier: int    = 1,        # Multiplier for the number of channels in the MLP.
        mlp_groups: int        = 1,        # Number of groups for the MLP.
    ) -> None:
        super().__init__()

        self.level = level
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_mode = resample_mode
        self.dropout = dropout
        self.res_balance = res_balance
        self.clip_act = clip_act
        
        self.conv_res0 = MPConv3D(out_channels if flavor == "enc" else in_channels,
                                out_channels * mlp_multiplier, kernel=(2,3,3), groups=mlp_groups)
        self.conv_res1 = MPConv3D(out_channels * mlp_multiplier, out_channels, kernel=(2,3,3), groups=mlp_groups)
        self.conv_skip = MPConv3D(in_channels, out_channels, kernel=(1,1,1), groups=1)

        self.emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.emb_linear = MPConv3D(emb_channels, out_channels * mlp_multiplier,
                                 kernel=(1,1,1), groups=1) if emb_channels != 0 else None

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        
        x = resample_3d(x, mode=self.resample_mode)

        if self.flavor == "enc":
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1) # pixel norm

        y = self.conv_res0(mp_silu(x))

        c = self.emb_linear(emb, gain=self.emb_gain) + 1.
        y = mp_silu(y * c)

        if self.dropout != 0 and self.training == True: # magnitude preserving fix for dropout
            y = torch.nn.functional.dropout(y, p=self.dropout) * (1. - self.dropout)**0.5

        y = self.conv_res1(y)

        if self.flavor == "dec" and self.conv_skip is not None:
            x = self.conv_skip(x)
        x = mp_sum(x, y, t=self.res_balance)
        
        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)
        return x

class DiffusionAutoencoder_EDM2_D1(DualDiffusionVAE):

    supports_channels_last: Union[bool, Literal["3d"]] = "3d"

    def __init__(self, config: DiffusionAutoencoder_Config) -> None:
        super().__init__()
        self.config = config

        # ************************ VAE init ************************
        block_kwargs = {"dropout": config.dropout,
                        "mlp_multiplier": config.mlp_multiplier,
                        "mlp_groups": config.mlp_groups,
                        "res_balance": config.res_balance} 
        
        cblock = [config.model_channels * x for x in config.channel_mult]
        cemb = config.model_channels * config.channel_mult_emb if config.channel_mult_emb is not None else max(cblock)
        cemb *= self.config.mlp_multiplier

        self.num_levels = len(config.channel_mult)
        
        # Embedding.
        self.emb_label_enc = MPConv3D(config.in_channels_emb, cemb, kernel=())
        self.emb_label_dec = MPConv3D(config.in_channels_emb, cemb, kernel=())
        self.emb_dim = cemb

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        self.dec = {}
        cout = 1

        for level, channels in enumerate(cblock):
            
            if level == 0:
                cin = cout
                cout = channels
                self.enc[f"conv_in"] = VAE_Block(level, cin, cout, cemb,
                                                flavor="enc", **block_kwargs)
                self.dec[f"conv_out"] = VAE_Block(level, cout, cin, cemb,
                                                flavor="dec", **block_kwargs)
            else:
                self.enc[f"block{level}_down"] = VAE_Block(level, cout, cout, cemb,
                                                flavor="enc", resample_mode="down", **block_kwargs)
                self.dec[f"block{level}_up"] = VAE_Block(level, cout, cout, cemb,
                                                flavor="dec", resample_mode="up", **block_kwargs)
                
            for idx in range(config.num_layers_per_block):
                cin = cout
                cout = channels
                self.enc[f"block{level}_layer{idx}"] = VAE_Block(level, cin, cout, cemb,
                                                            flavor="enc", **block_kwargs)
                self.dec[f"block{level}_layer{idx}"] = VAE_Block(level, cout, cin, cemb,
                                                            flavor="dec", **block_kwargs)

        self.enc["conv_latents_out"] = VAE_Block(level, cout, config.latent_channels, cemb, flavor="enc", **block_kwargs)
        self.dec["conv_latents_in"] = VAE_Block(level, config.latent_channels, cout, cemb, flavor="dec", **block_kwargs)

        self.dec = torch.nn.ModuleDict({k:v for k,v in reversed(self.dec.items())})

        # ************************ DDEC init ************************
        block_kwargs = {"dropout": config.dropout,
                        "mlp_multiplier": config.ddec_mlp_multiplier,
                        "mlp_groups": config.ddec_mlp_groups,
                        "res_balance": config.ddec_res_balance}

        cblock = [config.ddec_model_channels * x for x in config.ddec_channel_mult]
        cnoise = config.ddec_model_channels * config.ddec_channel_mult_noise if config.ddec_channel_mult_noise is not None else max(cblock)
        cemb = config.ddec_model_channels * config.ddec_channel_mult_emb if config.ddec_channel_mult_emb is not None else max(cblock)
        cemb *= self.config.ddec_mlp_multiplier

        self.ddec_num_levels = len(config.ddec_channel_mult)

        # Embedding.
        self.ddec_emb_fourier = MPFourier(cnoise)
        self.ddec_emb_noise = MPConv3D(cnoise, cemb, kernel=())
        self.ddec_emb_label = MPConv3D(config.in_channels_emb, cemb, kernel=())
        self.ddec_emb_label_unconditional = MPConv3D(1, cemb, kernel=())

        # Training uncertainty estimation.
        self.ddec_logvar_fourier = MPFourier(config.ddec_logvar_channels)
        self.ddec_logvar_linear = MPConv3D(config.ddec_logvar_channels, 1, kernel=(), disable_weight_norm=True)

        # Encoder.
        self.ddec_enc = torch.nn.ModuleDict()
        cout = 2

        for level, channels in enumerate(cblock):
            
            if level == 0:
                cin = cout
                cout = channels
                self.ddec_enc[f"conv_in"] = MPConv3D(cin, cout, kernel=(2,3,3))
            else:
                self.ddec_enc[f"block{level}_down"] = DDec_Block(level, cout, cout, cemb, flavor="enc", resample_mode="down", **block_kwargs)
            
            for idx in range(config.ddec_num_layers_per_block):
                cin = cout
                cout = channels
                self.ddec_enc[f"block{level}_layer{idx}"] = DDec_Block(level, cin, cout, cemb, flavor="enc", **block_kwargs)

        # Decoder.
        self.ddec_dec = torch.nn.ModuleDict()
        skips = [block.out_channels for block in self.ddec_enc.values()]
        
        for level, channels in reversed(list(enumerate(cblock))):
            
            if level == len(cblock) - 1:
                self.ddec_dec[f"block{level}_in0"] = DDec_Block(level, cout, cout, cemb, flavor="dec", **block_kwargs)
                #self.ddec_dec[f"block{level}_in1"] = Block(level, cout, cout, cemb, flavor="dec", **block_kwargs)
            else:
                self.ddec_dec[f"block{level}_up"] = DDec_Block(level, cout, cout, cemb, flavor="dec", resample_mode="up", **block_kwargs)

            for idx in range(config.ddec_num_layers_per_block + 1):
                cin = cout + skips.pop()
                cout = channels
                self.ddec_dec[f"block{level}_layer{idx}"] = DDec_Block(level, cin, cout, cemb, flavor="dec", **block_kwargs)
                
        self.ddec_out_gain = torch.nn.Parameter(torch.zeros([]))
        self.ddec_conv_out = MPConv3D(cout, 1, kernel=(2,3,3))

            
    def get_embeddings(self, emb_in: torch.Tensor) -> torch.Tensor:
        enc_embeddings = self.emb_label_enc(normalize(emb_in).to(device=self.device, dtype=self.dtype))
        dec_embeddings  = self.emb_label_dec(normalize(emb_in).to(device=self.device, dtype=self.dtype))
        return enc_embeddings, dec_embeddings
    
    def get_ddec_embeddings(self, emb_in: torch.Tensor, conditioning_mask: torch.Tensor) -> torch.Tensor:
        u_embedding = self.ddec_emb_label_unconditional(torch.ones(1, device=self.device, dtype=self.dtype))
        c_embedding = self.ddec_emb_label(normalize(emb_in).to(device=self.device, dtype=self.dtype))
        return mp_sum(u_embedding, c_embedding, t=conditioning_mask.unsqueeze(1).to(self.device, self.dtype))
    
    def get_sigma_loss_logvar(self, sigma: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.ddec_logvar_linear(self.ddec_logvar_fourier(sigma.flatten().log() / 4)).view(-1, 1, 1, 1).float()
    
    def get_latent_shape(self, sample_shape: Union[torch.Size, tuple[int, int, int, int]]) -> torch.Size:
        if len(sample_shape) == 4:
            return (sample_shape[0], self.config.latent_channels * self.config.in_channels,
                    sample_shape[2] // 2 ** (self.num_levels-1),
                    sample_shape[3] // 2 ** (self.num_levels-1))
        else:
            raise ValueError(f"Invalid sample shape: {sample_shape}")
        
    def get_sample_shape(self, latent_shape: Union[torch.Size, tuple[int, int, int, int]]) -> torch.Size:
        if len(latent_shape) == 4:
            return (latent_shape[0],
                    self.config.out_channels,
                    latent_shape[2] * 2 ** (self.num_levels-1),
                    latent_shape[3] * 2 ** (self.num_levels-1))
        else:
            raise ValueError(f"Invalid latent shape: {latent_shape}")
        
    def encode(self, x: torch.Tensor, embeddings: torch.Tensor,
            format: DualDiffusionFormat) -> DegenerateDistribution:
        
        enc_embeddings = embeddings[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        x_in = tensor_4d_to_5d(x, num_channels=1)
        for block in self.enc.values():
            x_out = block(x_in, enc_embeddings)
            x_in = x_out

        x_out = x_out / x_out.std(dim=(1,2,3,4), keepdim=True)
        return DegenerateDistribution(tensor_5d_to_4d(x_out))
    
    def decode(self, x: torch.Tensor, embeddings: torch.Tensor,
                format: DualDiffusionFormat) -> torch.Tensor:
        
        dec_embeddings = embeddings[1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        x_in = tensor_4d_to_5d(x, num_channels=self.config.latent_channels)
        for block in self.dec.values():
            x_out: torch.Tensor = block(x_in, dec_embeddings)
            x_in = x_out

        return tensor_5d_to_4d(x_out)

    def encode_train(self, x: torch.Tensor, embeddings: torch.Tensor) -> list[torch.Tensor]:
        enc_embeddings = embeddings[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        enc_states: list[torch.Tensor] = []

        x_in = tensor_4d_to_5d(x, num_channels=1)
        for block in self.enc.values():
            x_out = block(x_in, enc_embeddings)
            enc_states.append((x_in, x_out))
            x_in = x_out

        return enc_states
    
    def decode_train(self, enc_states: list[torch.Tensor], embeddings: torch.Tensor) -> list[torch.Tensor]:
        
        dec_embeddings = embeddings[1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        dec_states: list[torch.Tensor] = []

        x_in = enc_states[-1][1]
        x_in = x_in / x_in.std(dim=(1,2,3,4), keepdim=True).detach()

        for block in self.dec.values():
            x_out = block(x_in, dec_embeddings)
            dec_states.append((x_in, x_out))
            x_in = x_out

        return dec_states
    
    def ddec_forward(self, x_in: torch.Tensor,
                sigma: torch.Tensor,
                format: DualDiffusionFormat,
                embeddings: torch.Tensor,
                x_ref: Optional[torch.Tensor] = None) -> torch.Tensor:

        with torch.no_grad():
            sigma = sigma.view(-1, 1, 1, 1, 1)
            
            # Preconditioning weights.
            c_skip = self.config.ddec_sigma_data ** 2 / (sigma ** 2 + self.config.ddec_sigma_data ** 2)
            c_out = sigma * self.config.ddec_sigma_data / (sigma ** 2 + self.config.ddec_sigma_data ** 2).sqrt()
            c_in = 1 / (self.config.ddec_sigma_data ** 2 + sigma ** 2).sqrt()
            c_noise = (sigma.flatten().log() / 4).to(self.dtype)

            x_in = tensor_4d_to_5d(x_in, num_channels=1)
            x = (c_in * x_in).to(self.dtype)
 
        # Embedding.
        emb = self.ddec_emb_noise(self.ddec_emb_fourier(c_noise))
        emb = mp_sum(emb, embeddings.to(emb.dtype), t=self.config.ddec_label_balance)
        emb = mp_silu(emb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(x.dtype)

        # Encoder.
        x = torch.cat((x, tensor_4d_to_5d(x_ref, num_channels=1)), dim=1)

        skips = []
        for name, block in self.ddec_enc.items():
            x = block(x) if "conv" in name else block(x, emb)
            skips.append(x)

        # Decoder.
        for name, block in self.ddec_dec.items():
            if "layer" in name:
                x = mp_cat(x, skips.pop(), t=self.config.ddec_concat_balance)
            x = block(x, emb)

        x: torch.Tensor = self.ddec_conv_out(x, gain=self.ddec_out_gain)
        D_x: torch.Tensor = c_skip * x_in.float() + c_out * x.float()

        return tensor_5d_to_4d(D_x)
    
    def forward(self, samples: torch.Tensor, vae_embeddings: torch.Tensor,
                noised_samples: torch.Tensor, sigma: torch.Tensor, format: DualDiffusionFormat,
                ddec_embeddings: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        
        enc_states = self.encode_train(samples, vae_embeddings)
        dec_states = self.decode_train(enc_states, vae_embeddings)

        ref_samples: torch.Tensor = tensor_5d_to_4d(dec_states[-1][1])
        denoised = self.ddec_forward(noised_samples, sigma, format, ddec_embeddings, ref_samples)

        return enc_states, dec_states, denoised