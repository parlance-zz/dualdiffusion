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

from dataclasses import dataclass
from typing import Union, Literal, Optional

import torch
from numpy import ndarray

from modules.daes.dae import DualDiffusionDAE, DualDiffusionDAEConfig
from modules.formats.ms_mdct_dual_9 import patch_ms_psd, unpatch_ms_psd
from modules.mp_tools import (
    LatentStatsTracker,
    MPConv,
    AdaptiveGroupBalance,
    mp_silu,
    mp_sum,
    normalize,
    resample_2d,
    patchify_2d,
    unpatchify_2d,
    normalize_groups,
)
from modules.unets.unet_edm2_p6 import UNet, UNetConfig


def residual_space_to_channel_avg(x: torch.Tensor, out_channels: int, factor: int = 2) -> torch.Tensor:
    in_channels = x.shape[1]
    assert in_channels * factor**2 % out_channels == 0
    group_size = in_channels * factor**2 // out_channels

    x = torch.nn.functional.pixel_unshuffle(x, factor)
    b, _, h, w = x.shape
    x = x.view(b, out_channels, group_size, h, w)
    return x.mean(dim=2)


def residual_channel_to_space_dup(x: torch.Tensor, out_channels: int, factor: int = 2) -> torch.Tensor:
    in_channels = x.shape[1]
    assert out_channels * factor**2 % in_channels == 0
    repeats = out_channels * factor**2 // in_channels

    x = x.repeat_interleave(repeats, dim=1)
    return torch.nn.functional.pixel_shuffle(x, factor)


@dataclass
class DAE_Config(DualDiffusionDAEConfig):

    in_channels: int     = 3
    in_channels_emb: int = 0
    out_channels: int    = 3
    latent_channels: int = 16
    use_1d_latents: bool = False
    use_latents_pixel_norm: bool = True

    in_num_freqs: int = 64
    in_psd_num_freqs: list[int] = (64, 128, 256, 512)

    adg_min_balance: Optional[float] = 0.1
    adg_max_balance: Optional[float] = 0.9
    adg_weight_decay: Optional[float] = None
    balance_logits_offset: float = -4

    model_channels: int         = 512        # Base multiplier for the number of channels.
    channel_mult_enc: int       = (1,2,4)
    channel_mult_dec: list[int] = (1,2,4)
    channel_mult_emb: int     = 0            # Multiplier for final embedding dimensionality.
    channels_per_head: int    = 128          # Number of channels per attention head.
    num_enc_layers_per_block: int = 2        # Number of resnet blocks per resolution.
    num_dec_layers_per_block: int = 2        # Number of resnet blocks per resolution.
    mlp_multiplier: int    = 1               # Multiplier for the number of channels in the MLP.
    mlp_groups: int        = 4               # Number of groups for the MLPs.

    emb_linear_groups: int = 1
    add_pixel_norm: bool   = False

    add_recon_logvar: bool = True

    unet: Optional[UNetConfig] = None


class Block(torch.nn.Module):

    def __init__(self,
        level: int,                             # Resolution level.
        in_channels: int,                       # Number of input channels.
        out_channels: int,                      # Number of output channels.
        emb_channels: int,                      # Number of embedding channels.
        flavor: Literal["enc", "dec"] = "enc",
        resample_mode: Literal["keep", "up", "down"] = "keep",
        dropout: float         = 0.,       # Dropout probability.
        balance_logits_offset: float = -4,
        clip_act: float        = 256,      # Clip output activations. None = do not clip.
        mlp_multiplier: int    = 1,        # Multiplier for the number of channels in the MLP.
        mlp_groups: int        = 1,        # Number of groups for the MLP.
        emb_linear_groups: int = 1,
        channels_per_head: int = 64,       # Number of channels per attention head.
        adg_min_balance: Optional[float] = 0.1,
        adg_max_balance: Optional[float] = 0.9,
        adg_weight_decay: Optional[float] = None,
        use_attention: bool    = False,    # Use self-attention in this block.
        use_pixel_norm: bool   = False,
    ) -> None:
        super().__init__()

        assert out_channels % channels_per_head == 0

        self.level = level
        self.use_attention = use_attention
        self.use_pixel_norm = use_pixel_norm
        self.num_heads = out_channels // mlp_groups // channels_per_head
        self.channels_per_head = channels_per_head
        self.mlp_groups = mlp_groups
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_mode = resample_mode
        self.dropout = dropout
        self.balance_logits_offset = balance_logits_offset
        self.clip_act = clip_act

        inner_channels = out_channels * mlp_multiplier

        assert emb_channels % emb_linear_groups == 0
        assert inner_channels % mlp_groups == 0
        assert inner_channels % emb_linear_groups == 0
        assert out_channels % mlp_groups == 0
        assert in_channels % mlp_groups == 0

        kernel = (3, 3)

        self.conv_res0 = MPConv(out_channels, inner_channels, kernel=kernel, groups=mlp_groups)
        self.conv_res1 = MPConv(inner_channels, out_channels, kernel=kernel, groups=mlp_groups)

        if in_channels != out_channels:
            skip_kernel = (3, 3) if resample_mode != "keep" else (1, 1)
            self.conv_skip = MPConv(in_channels, out_channels, kernel=skip_kernel, groups=1)
        else:
            self.conv_skip = None

        if emb_channels > 0:
            self.emb_gain = torch.nn.Parameter(torch.zeros([]))
            self.emb_linear = MPConv(emb_channels, inner_channels,
                kernel=(1, 1), groups=emb_linear_groups) if emb_channels != 0 else None
        else:
            self.emb_gain = self.emb_linear = None

        self.emb_res_balance = AdaptiveGroupBalance(
            emb_channels,
            mlp_groups,
            balance_logits_offset,
            min_balance=adg_min_balance,
            max_balance=adg_max_balance,
            weight_decay=adg_weight_decay,
        )

        if self.use_attention == True:
            self.attn_q = MPConv(out_channels, out_channels, kernel=(1, 1), groups=mlp_groups)
            self.attn_k = MPConv(out_channels, out_channels, kernel=(1, 1), groups=mlp_groups)
            self.attn_v = MPConv(out_channels, out_channels, kernel=(1, 1), groups=mlp_groups)
            self.attn_proj = MPConv(out_channels, out_channels, kernel=(1, 1), groups=mlp_groups)
            self.emb_attn_balance = AdaptiveGroupBalance(
                emb_channels,
                mlp_groups,
                balance_logits_offset,
                min_balance=adg_min_balance,
                max_balance=adg_max_balance,
                weight_decay=adg_weight_decay,
            )

            if emb_channels > 0:
                self.emb_gain_qkv = torch.nn.Parameter(torch.zeros([]))
                self.emb_linear_qkv = MPConv(emb_channels, out_channels, kernel=(1, 1), groups=emb_linear_groups)
            else:
                self.emb_gain_qkv = self.emb_linear_qkv = None

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:

        if self.flavor == "enc":
            if self.resample_mode != "keep":
                y = residual_space_to_channel_avg(x, self.out_channels, factor=2)

            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = resample_2d(x, self.resample_mode)

            if self.resample_mode != "keep":
                x = mp_sum(x, y, t=0.5)

        if self.flavor == "dec":
            if self.resample_mode != "keep":
                y = residual_channel_to_space_dup(x, self.out_channels, factor=2)

            x = resample_2d(x, self.resample_mode)
            if self.conv_skip is not None:
                x = self.conv_skip(x)

            if self.resample_mode != "keep":
                x = mp_sum(x, y, t=0.5)

        if self.use_pixel_norm == True and self.flavor == "enc":
            x = normalize(x, dim=1)

        y = self.conv_res0(x)

        if self.emb_linear is not None:
            c: torch.Tensor = self.emb_linear(emb, gain=self.emb_gain) + 1.
            y = y * c

        y = mp_silu(normalize_groups(y, groups=self.mlp_groups))

        if self.dropout != 0 and self.training == True:  # magnitude preserving fix for dropout
            y = torch.nn.functional.dropout(y, p=self.dropout) * (1. - self.dropout)**0.5

        y = self.conv_res1(y)
        x = self.emb_res_balance(x, y, emb)

        if self.use_attention == True:
            if self.emb_linear_qkv is not None:
                c = self.emb_linear_qkv(emb, gain=self.emb_gain_qkv) + 1.
                y = x * c
            else:
                y = x

            B, C, H, W = y.shape

            q: torch.Tensor = self.attn_q(y).permute(0, 3, 2, 1)
            k: torch.Tensor = self.attn_k(y).permute(0, 3, 2, 1)
            v: torch.Tensor = self.attn_v(y).permute(0, 3, 2, 1)
            q = q.reshape(B, W, H, self.mlp_groups, self.channels_per_head)
            k = k.reshape(B, W, H, self.mlp_groups, self.channels_per_head)
            v = v.reshape(B, W, H, self.mlp_groups, self.channels_per_head)
            q = normalize(q, dim=4)
            k = normalize(k, dim=4)
            v = normalize(v, dim=4)

            y = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            y = y.permute(0, 3, 4, 2, 1).reshape(B, C, H, W)

            y = self.attn_proj(y)
            x = self.emb_attn_balance(x, y, emb)

        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)

        return x


class DAE(DualDiffusionDAE):

    def __init__(self, config: DAE_Config) -> None:
        super().__init__()
        self.config = config

        block_kwargs = {"mlp_multiplier": config.mlp_multiplier,
                        "emb_linear_groups": config.emb_linear_groups,
                        "balance_logits_offset": config.balance_logits_offset,
                        "channels_per_head": config.channels_per_head,
                        "adg_min_balance": config.adg_min_balance,
                        "adg_max_balance": config.adg_max_balance,
                        "adg_weight_decay": config.adg_weight_decay,
                        "use_pixel_norm": config.add_pixel_norm}

        cemb = config.model_channels * config.channel_mult_emb if config.in_channels_emb > 0 else 0
        assert config.model_channels % config.mlp_groups == 0

        self.num_levels = len(config.channel_mult_dec)
        self.num_psd_levels = len(config.in_psd_num_freqs)
        self.downsample_ratio = 2 ** (self.num_levels - 1)
        assert config.in_num_freqs % self.downsample_ratio == 0
        self.num_latent_freqs = config.in_num_freqs // self.downsample_ratio 

        self.psd_freqs_per_freq = 2 ** (self.num_psd_levels - 1)
        for i in range(self.num_psd_levels):
            assert config.in_psd_num_freqs[i] % self.psd_freqs_per_freq == 0

        # embedding
        if config.in_channels_emb > 0:
            self.emb_label = MPConv(config.in_channels_emb, cemb, kernel=())
            self.emb_dim = cemb
        else:
            cemb = 0
            self.emb_label = None
            self.emb_dim = 0

        enc_channels = [config.model_channels * m for m in config.channel_mult_enc]
        dec_channels = [config.model_channels * m for m in config.channel_mult_dec]

        if config.add_recon_logvar == True:
            self.recon_logvar = torch.nn.Parameter(torch.zeros([]))

        self.latents_stats_tracker = LatentStatsTracker(config.latent_channels)

        # encoder
        self.conv_psd_in = MPConv(config.in_channels * self.psd_freqs_per_freq * self.num_psd_levels, enc_channels[0], kernel=(3,3), bias=True)

        self.enc = torch.nn.ModuleDict()
        cin = enc_channels[0]

        for level in range(self.num_levels):

            cout = enc_channels[level]

            if level > 0:
                self.enc[f"block{level}_down"] = Block(level, cin, cout, cemb,
                    use_attention=True, flavor="enc", resample_mode="down", mlp_groups=cout // config.channels_per_head, **block_kwargs)

            for idx in range(config.num_enc_layers_per_block):
                cin = cout
                cout = enc_channels[level]
                self.enc[f"block{level}_layer{idx}"] = Block(level, cout, cout, cemb,
                    use_attention=True, flavor="enc", mlp_groups=cout // config.channels_per_head, **block_kwargs)

        self.latents_out_gain = torch.nn.Parameter(torch.ones([]))

        if config.use_1d_latents == True:
            self.conv_latents_out = MPConv(enc_channels[-1] * self.num_latent_freqs, config.latent_channels, kernel=(1,1))
            self.conv_latents_in = MPConv(config.latent_channels, dec_channels[-1] * self.num_latent_freqs, kernel=(1,1), bias=True)
            with torch.no_grad():
                self.conv_latents_in.weight.copy_(torch.linalg.pinv(self.conv_latents_out.weight.data[:, :, 0, 0])[:, :, None, None])
                self.conv_latents_in.bias.zero_()
        else:
            self.conv_latents_out = MPConv(enc_channels[-1], config.latent_channels, kernel=(3,3))
            self.conv_latents_in = MPConv(config.latent_channels, dec_channels[-1], kernel=(3,3), bias=True)

        # decoder
        self.dec = torch.nn.ModuleDict()
        cin = dec_channels[-1]

        for level in reversed(range(0, self.num_levels)):

            cout = dec_channels[level]

            if level == self.num_levels - 1:
                self.dec[f"block{level}_in0"] = Block(level, cin, cout, cemb,
                    use_attention=True, flavor="dec", mlp_groups=cout // config.channels_per_head, **block_kwargs)
            else:
                self.dec[f"block{level}_up"] = Block(level, cin, cout, cemb,
                    use_attention=True, flavor="dec", resample_mode="up", mlp_groups=cout // config.channels_per_head, **block_kwargs)

            for idx in range(config.num_dec_layers_per_block):
                cin = cout
                cout = dec_channels[level]
                use_attention = not (level == 0 and idx == config.num_dec_layers_per_block - 1)
                self.dec[f"block{level}_layer{idx}"] = Block(level, cout, cout, cemb,
                    use_attention=use_attention, flavor="dec", mlp_groups=cout // config.channels_per_head, **block_kwargs)

        self.conv_psd_out = MPConv(cout, config.out_channels * self.psd_freqs_per_freq * self.num_psd_levels, kernel=(3,3))
        self.psd_out_gain = torch.nn.Parameter(torch.ones([]))

        if config.unet is not None:
            self.unet = UNet(config.unet)
        else:
            self.unet = None

    def get_embeddings(self, emb_in: torch.Tensor) -> torch.Tensor:
        if self.emb_label is not None:
            return mp_silu(self.emb_label(normalize(emb_in).to(device=self.device, dtype=self.dtype)))
        else:
            return None

    def get_recon_loss_logvar(self) -> torch.Tensor:
        return getattr(self, "recon_logvar", None)

    def get_latent_shape(self, mel_spec_shape: Union[torch.Size, tuple[int, int, int, int]]) -> torch.Size:
        if len(mel_spec_shape) == 4:
            if self.config.use_1d_latents == True:
                return (mel_spec_shape[0], self.config.latent_channels, 1,
                        mel_spec_shape[3] // 2 ** (self.num_levels - 1) // 2 ** (self.num_psd_levels - 1))
            else:
                return (mel_spec_shape[0], self.config.latent_channels,
                        mel_spec_shape[2] // 2 ** (self.num_levels - 1),
                        mel_spec_shape[3] // 2 ** (self.num_levels - 1) // 2 ** (self.num_psd_levels - 1))
        else:
            raise ValueError(f"Invalid sample shape: {mel_spec_shape}")

    def get_mel_spec_shape(self, latent_shape: Union[torch.Size, tuple[int, int, int, int]]) -> torch.Size:
        if len(latent_shape) == 4:
            return (latent_shape[0], self.config.out_channels,
                self.config.in_num_freqs, latent_shape[3] * 2 ** (self.num_levels-1) * 2 ** (self.num_psd_levels-1))
        else:
            raise ValueError(f"Invalid latent shape: {latent_shape}")

    def encode(self, x: list[torch.Tensor], embeddings: torch.Tensor, training: bool = False) -> torch.Tensor:

        if embeddings is not None:
            embeddings = embeddings[:, :, None, None]

        x: torch.Tensor = self.conv_psd_in(patch_ms_psd(x, self.num_psd_levels).to(dtype=torch.bfloat16))

        for name, block in self.enc.items():
            x = block(x, embeddings)

        if self.config.use_1d_latents == True:
            assert x.shape[2] == self.num_latent_freqs
            x = patchify_2d(x, self.num_latent_freqs, 1)

        latents: torch.Tensor = self.conv_latents_out(x, gain=self.latents_out_gain)
        latents = normalize(latents.float(), dim=1 if self.config.use_latents_pixel_norm == True else None)

        if training == False:
            assert self.training == False

        return latents

    def decode(self, x: torch.Tensor, embeddings: torch.Tensor, training: bool = False) -> list[torch.Tensor]:

        if training == False:
            assert self.training == False
            x = x.float()
            x = normalize(x, dim=1 if self.config.use_latents_pixel_norm == True else None)

        x = self.conv_latents_in(x.to(dtype=torch.bfloat16))

        if self.config.use_1d_latents == True:
            assert x.shape[2] == 1
            x = unpatchify_2d(x, self.num_latent_freqs, 1)

        if embeddings is not None:
            embeddings = embeddings[:, :, None, None]

        for name, block in self.dec.items():
            x = block(x, embeddings)

        psd_output: torch.Tensor = self.conv_psd_out(x, gain=self.psd_out_gain).float()
        return unpatch_ms_psd(psd_output, self.num_psd_levels)

    def forward(self, samples: torch.Tensor, audio_embeddings: torch.Tensor, batch_sigma: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:

        dae_embeddings = self.get_embeddings(audio_embeddings)
        latents = self.encode(samples, dae_embeddings, training=True).float()

        if batch_sigma is not None:

            assert self.unet is not None

            conditioning_mask = torch.ones(latents.shape[0], device=latents.device)

            noise = torch.randn(latents.shape, device=latents.device, dtype=torch.float32)
            noise = (noise * batch_sigma.view(-1, 1, 1, 1)).detach()

            denoised, error_logvar = self.unet(latents + noise, batch_sigma, None, audio_embeddings, conditioning_mask=conditioning_mask)

            batch_loss_weight = (batch_sigma**2 + 1) / batch_sigma**2
            batch_weighted_loss = torch.nn.functional.mse_loss(denoised, latents, reduction="none").mean(dim=(1,2,3)) * batch_loss_weight

            batch_loss = batch_weighted_loss / error_logvar.exp() + error_logvar
            bucket_log_loss = batch_weighted_loss
            decode_latents = denoised
        else:
            assert self.unet is None
            batch_loss = bucket_log_loss = None
            decode_latents = latents

        ddec_cond = self.decode(decode_latents, dae_embeddings, training=True)

        return latents, ddec_cond, self.get_recon_loss_logvar(), batch_loss, bucket_log_loss

    def latents_to_img(self, latents: torch.Tensor, **kwargs) -> ndarray:

        if self.config.use_1d_latents == True:

            latents = latents.reshape(latents.shape[0], latents.shape[1] // 4, 4, latents.shape[3])
            latents = latents.permute(0, 2, 1, 3).contiguous()

        elif self.config.latent_channels > 8:
            latents = unpatchify_2d(latents, self.downsample_ratio, 1)

        return super().latents_to_img(latents, img_split_stereo=False, **kwargs)

    def tiled_encode(self, x: torch.Tensor, embeddings: torch.Tensor, max_chunk: int = 6144, overlap: int = 256) -> torch.Tensor:

        raise NotImplementedError()

        x_w = x.shape[-1]
        ds = self.downsample_ratio

        assert max_chunk % ds == 0, "max_chunk must be divisible by downsample ratio"
        assert overlap % ds == 0, "overlap must be divisible by downsample ratio"
        assert x_w % ds == 0, "sample length must be divisible by downsample ratio"

        if x_w <= max_chunk:
            return self.encode(x, embeddings)

        min_chunk_len = overlap * 3
        out_overlap = overlap // ds

        latents_shape = (x.shape[0], self.config.latent_channels*2, x.shape[-2] // ds, x.shape[-1] // ds)
        latents = torch.zeros(latents_shape, device=x.device, dtype=x.dtype)

        # encode latents in overlapping chunks
        for w_start in range(0, x_w, max_chunk - overlap*2):

            if w_start >= x_w:
                break

            # sample boundaries including overlap
            chunk_start = max(0, w_start)
            chunk_end = min(x_w, w_start + max_chunk)

            # if last chunk is too small, extend it to the left
            if chunk_end - chunk_start < min_chunk_len:
                chunk_start -= min_chunk_len - (chunk_end - chunk_start)

            chunk = x[:, :, :, chunk_start:chunk_end]
            latents_chunk = self.encode(chunk, embeddings, normalize_latents=False)

            # latent boundaries including overlap
            out_start = chunk_start // ds
            out_end = chunk_end // ds

            # first chunk: keep left edge, other chunks: discard left overlap
            is_first_chunk = (w_start == 0)
            valid_start = 0 if is_first_chunk else out_overlap

            # last chunk: keep right edge, other chunks: discard right overlap
            is_last_chunk = (chunk_end == x_w)
            valid_end = latents_chunk.shape[3] if is_last_chunk else latents_chunk.shape[3] - out_overlap

            # latent boundaries excluding overlap
            dest_start = out_start if is_first_chunk else out_start + out_overlap
            dest_end = out_end if is_last_chunk else out_end - out_overlap

            latents[:, :, :, dest_start:dest_end] = latents_chunk[:, :, :, valid_start:valid_end]

        return latents
