# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.vae import DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.autoencoder_kl import AutoencoderKLOutput

from unet2d_dual_blocks import SeparableAttnDownBlock2D, SeparableAttnUpBlock2D, SeparableMidBlock2D, get_activation

class EncoderDual(nn.Module):
    def __init__(
        self,
        in_channels=2,
        out_channels=4,
        act_fn="silu",
        double_z=True,
        block_out_channels=(128, 256, 512, 512,),
        layers_per_block=2,
        layers_per_mid_block: int = 1,
        norm_num_groups: Union[int, Tuple[int]] = 32,
        norm_eps: float = 1e-6,
        downsample_type: str = "conv",
        add_mid_attention: bool = True,
        attention_num_heads: Union[int, Tuple[int]] = (8,8,16,16),
        separate_attn_dim_down: Tuple[int] = (2,3),
        separate_attn_dim_mid: Tuple[int] = (0,),
        double_attention: Union[bool, Tuple[bool]] = False,
        pre_attention: Union[bool, Tuple[bool]] = False,
        conv_size = (3,3),
        freq_embedding_dim: Union[int, Tuple[int]] = 0,
        time_embedding_dim: Union[int, Tuple[int]] = 0,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = torch.nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.mid_block = None
        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i in range(len(block_out_channels)):
            input_channel = output_channel
            output_channel = block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1
            if is_final_block is True: _downsample_type = None
            else: _downsample_type = downsample_type or "conv"
            _norm_num_groups = norm_num_groups[i]
            _attention_num_heads = attention_num_heads[i]
            _double_attention = double_attention[i]
            _pre_attention = pre_attention[i]
            _freq_embedding_dim = freq_embedding_dim[i]
            _time_embedding_dim = time_embedding_dim[i]

            down_block = SeparableAttnDownBlock2D(
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=_norm_num_groups,
                attention_num_heads=_attention_num_heads,
                downsample_padding=0, #?
                downsample_type=_downsample_type,
                separate_attn_dim=separate_attn_dim_down,
                double_attention=_double_attention,
                pre_attention=_pre_attention,
                conv_size=conv_size,
                freq_embedding_dim=_freq_embedding_dim,
                time_embedding_dim=_time_embedding_dim,
                return_skip_samples=False,
                temb_channels=None,
            )

            self.down_blocks.append(down_block)

        # mid
        _norm_num_groups = norm_num_groups[-1]
        _attention_num_heads = attention_num_heads[-1]
        _double_attention = double_attention[-1]
        _pre_attention = pre_attention[-1]
        _freq_embedding_dim = freq_embedding_dim[-1]
        _time_embedding_dim = time_embedding_dim[-1]
        
        self.mid_block = SeparableMidBlock2D(
            in_channels=block_out_channels[-1],
            temb_channels=None,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            resnet_time_scale_shift="default",
            attention_num_heads=_attention_num_heads,
            resnet_groups=_norm_num_groups,
            add_attention=add_mid_attention,
            pre_attention=_pre_attention,
            separate_attn_dim=separate_attn_dim_mid,
            double_attention=_double_attention,
            conv_size=conv_size,
            num_layers=layers_per_mid_block,
            freq_embedding_dim=_freq_embedding_dim,
            time_embedding_dim=_time_embedding_dim,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups[-1], eps=norm_eps)
        self.conv_act = get_activation(act_fn)

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(self, x):
        sample = x
        sample = self.conv_in(sample)

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # down
            for down_block in self.down_blocks:
                sample, _ = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(down_block), sample, use_reentrant=False
                )
            # middle
            sample = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.mid_block), sample, use_reentrant=False
            )

        else:
            # down
            for down_block in self.down_blocks:
                sample, _ = down_block(sample)

            # middle
            sample = self.mid_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class DecoderDual(nn.Module):
    def __init__(
        self,
        in_channels=4,
        out_channels=2,
        act_fn="silu",
        block_out_channels=(128, 256, 512, 512,),
        layers_per_block=2,
        layers_per_mid_block: int = 1,
        norm_num_groups: Union[int, Tuple[int]] = 32,
        norm_eps: float = 1e-6,
        upsample_type: str = "conv",
        add_mid_attention: bool = True,
        attention_num_heads: Union[int, Tuple[int]] = (8,8,16,16),
        separate_attn_dim_up: Tuple[int] = (3,2,3),
        separate_attn_dim_mid: Tuple[int] = (0,),
        double_attention: Union[bool, Tuple[bool]] = False,
        pre_attention: Union[bool, Tuple[bool]] = False,
        conv_size = (3,3),
        freq_embedding_dim: Union[int, Tuple[int]] = 0,
        time_embedding_dim: Union[int, Tuple[int]] = 0,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # mid
        _norm_num_groups = norm_num_groups[-1]
        _attention_num_heads = attention_num_heads[-1]
        _double_attention = double_attention[-1]
        _pre_attention = pre_attention[-1]
        _freq_embedding_dim = freq_embedding_dim[-1]
        _time_embedding_dim = time_embedding_dim[-1]
        
        self.mid_block = SeparableMidBlock2D(
            in_channels=block_out_channels[-1],
            temb_channels=None,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_num_heads=_attention_num_heads,
            resnet_groups=_norm_num_groups,
            add_attention=add_mid_attention,
            pre_attention=_pre_attention,
            separate_attn_dim=separate_attn_dim_mid,
            double_attention=_double_attention,
            conv_size=conv_size,
            num_layers=layers_per_mid_block,
            freq_embedding_dim=_freq_embedding_dim,
            time_embedding_dim=_time_embedding_dim,
        )

        reversed_attention_num_heads = list(reversed(attention_num_heads))
        reversed_norm_num_groups = list(reversed(norm_num_groups))
        reversed_double_attention = list(reversed(double_attention))
        reversed_pre_attention = list(reversed(pre_attention))
        reversed_freq_embedding_dim = list(reversed(freq_embedding_dim))
        reversed_time_embedding_dim = list(reversed(time_embedding_dim))

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i in range(len(block_out_channels)):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1
            if is_final_block is True: _upsample_type = None
            else: _upsample_type = upsample_type or "conv"  # default to 'conv'
            _norm_num_groups = reversed_norm_num_groups[i]
            _attention_num_heads = reversed_attention_num_heads[i]
            _double_attention = reversed_double_attention[i]
            _pre_attention = reversed_pre_attention[i]
            _freq_embedding_dim = reversed_freq_embedding_dim[i]
            _time_embedding_dim = reversed_time_embedding_dim[i]

            up_block = SeparableAttnUpBlock2D(
                num_layers=layers_per_block + 1,
                in_channels=0,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=None,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=_norm_num_groups,
                attention_num_heads=_attention_num_heads,
                resnet_time_scale_shift="group",
                upsample_type=_upsample_type,
                separate_attn_dim=separate_attn_dim_up,
                double_attention=_double_attention,
                pre_attention=_pre_attention,
                conv_size=conv_size,
                freq_embedding_dim=_freq_embedding_dim,
                time_embedding_dim=_time_embedding_dim,
                use_skip_samples=False,
            )

            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups[-1], eps=norm_eps)
        self.conv_act = get_activation(act_fn)
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(self, z, latent_embeds=None):
        sample = z
        sample = self.conv_in(sample)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # middle
            sample = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.mid_block), sample, latent_embeds, use_reentrant=False
            )
            sample = sample.to(upscale_dtype)

            # up
            for up_block in self.up_blocks:
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(up_block), sample, latent_embeds, use_reentrant=False
                )

        else:
            # middle
            sample = self.mid_block(sample, latent_embeds)
            sample = sample.to(upscale_dtype)

            # up
            for up_block in self.up_blocks:
                sample = up_block(sample, latent_embeds)

        # post-process
        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class AutoencoderKLDual(ModelMixin, ConfigMixin):

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        act_fn: str = "silu",
        block_out_channels: Tuple[int] = (128, 256, 512, 512,),
        latent_channels: int = 4,
        sample_size: Tuple[int, int] = (512, 512,),
        scaling_factor: float = 0.18215,
        layers_per_block=2,
        layers_per_mid_block: int = 1,
        norm_num_groups: Union[int, Tuple[int]] = 32,
        norm_eps: float = 1e-6,
        downsample_type: str = "conv",
        upsample_type: str = "conv",
        add_mid_attention: bool = True,
        attention_num_heads: Union[int, Tuple[int]] = (8,8,16,16),
        separate_attn_dim_down: Tuple[int] = (2,3),
        separate_attn_dim_up: Tuple[int] = (3,2,3),    
        separate_attn_dim_mid: Tuple[int] = (0,),
        double_attention: Union[bool, Tuple[bool]] = False,
        pre_attention: Union[bool, Tuple[bool]] = False,
        conv_size = (3,3),
        freq_embedding_dim: Union[int, Tuple[int]] = 0,
        time_embedding_dim: Union[int, Tuple[int]] = 0,
        last_global_step: int = 0,
    ):
        super().__init__()

        if not isinstance(attention_num_heads, int) and len(attention_num_heads) != len(block_out_channels):
            raise ValueError(
                f"Must provide the same number of `attention_num_heads` as `block_out_channels`. `attention_num_heads`: {attention_num_heads}. `block_out_channels`: {block_out_channels}."
            )   
        
        if not isinstance(norm_num_groups, int) and len(norm_num_groups) != len(block_out_channels):
            raise ValueError(
                f"Must provide the same number of `norm_num_groups` as `block_out_channels`. `norm_num_groups`: {norm_num_groups}. `block_out_channels`: {block_out_channels}."
            ) 
        
        if not isinstance(double_attention, bool) and len(double_attention) != len(block_out_channels):
            raise ValueError(
                f"Must provide the same number of `double_attention` as `block_out_channels`. `double_attention`: {double_attention}. `block_out_channels`: {block_out_channels}."
            )
        
        if not isinstance(pre_attention, bool) and len(pre_attention) != len(block_out_channels):
            raise ValueError(
                f"Must provide the same number of `pre_attention` as `block_out_channels`. `pre_attention`: {pre_attention}. `block_out_channels`: {block_out_channels}."
            )
        
        if not isinstance(freq_embedding_dim, int) and len(freq_embedding_dim) != len(block_out_channels):
            raise ValueError(
                f"Must provide the same number of `freq_embedding_dim` as `block_out_channels`. `freq_embedding_dim`: {freq_embedding_dim}. `block_out_channels`: {block_out_channels}."
            ) 
        
        if not isinstance(time_embedding_dim, int) and len(time_embedding_dim) != len(block_out_channels):
            raise ValueError(
                f"Must provide the same number of `time_embedding_dim` as `block_out_channels`. `time_embedding_dim`: {time_embedding_dim}. `block_out_channels`: {block_out_channels}."
            ) 
        
        if isinstance(attention_num_heads, int):
            attention_num_heads = (attention_num_heads,) * len(block_out_channels)
        if isinstance(norm_num_groups, int):
            norm_num_groups = (norm_num_groups,) * len(block_out_channels)
        if isinstance(separate_attn_dim_down, int):
            separate_attn_dim_down = (separate_attn_dim_down,)
        if isinstance(separate_attn_dim_up, int):
            separate_attn_dim_up = (separate_attn_dim_up,)
        if isinstance(double_attention, bool):
            double_attention = (double_attention,) * len(block_out_channels)
        if isinstance(pre_attention, bool):
            pre_attention = (pre_attention,) * len(block_out_channels)
        if isinstance(freq_embedding_dim, int):
            freq_embedding_dim = (freq_embedding_dim,) * len(block_out_channels)
        if isinstance(time_embedding_dim, int):
            time_embedding_dim = (time_embedding_dim,) * len(block_out_channels)

        # pass init params to Encoder
        self.encoder = EncoderDual(
            in_channels=in_channels,
            out_channels=latent_channels,
            act_fn=act_fn,
            double_z=True,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            layers_per_mid_block=layers_per_mid_block,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            downsample_type=downsample_type,
            add_mid_attention=add_mid_attention,
            attention_num_heads=attention_num_heads,
            separate_attn_dim_down=separate_attn_dim_down,
            separate_attn_dim_mid=separate_attn_dim_mid,
            double_attention=double_attention,
            pre_attention=pre_attention,
            conv_size = conv_size,
            freq_embedding_dim=freq_embedding_dim,
            time_embedding_dim=time_embedding_dim,
        )

        # pass init params to Decoder
        self.decoder = DecoderDual(
            in_channels=latent_channels,
            out_channels=out_channels,
            act_fn=act_fn,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            layers_per_mid_block=layers_per_mid_block,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            upsample_type=upsample_type,
            add_mid_attention=add_mid_attention,
            attention_num_heads=attention_num_heads,
            separate_attn_dim_up=separate_attn_dim_up,
            separate_attn_dim_mid=separate_attn_dim_mid,
            double_attention=double_attention,
            pre_attention=pre_attention,
            conv_size = conv_size,
            freq_embedding_dim=freq_embedding_dim,
            time_embedding_dim=time_embedding_dim,
        )

        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)

        self.use_slicing = False
        self.use_tiling = False

        # only relevant if vae tiling is enabled
        self.tile_sample_min_size = self.config.sample_size
        sample_size = (
            self.config.sample_size[0]
            if isinstance(self.config.sample_size, (list, tuple))
            else self.config.sample_size
        )
        self.tile_latent_min_size = int(sample_size / (2 ** (len(self.config.block_out_channels) - 1)))
        self.tile_overlap_factor = 0.25

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (EncoderDual, DecoderDual)):
            module.gradient_checkpointing = value

    def enable_tiling(self, use_tiling: bool = True):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.use_tiling = use_tiling

    def disable_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_tiling` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.enable_tiling(False)

    def enable_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True

    def disable_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    @apply_forward_hook
    def encode(self, x: torch.FloatTensor, return_dict: bool = True) -> AutoencoderKLOutput:
        if self.use_tiling and (x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size):
            return self.tiled_encode(x, return_dict=return_dict)

        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self.encoder(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self.encoder(x)

        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(self, z: torch.FloatTensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        if self.use_tiling and (z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size):
            return self.tiled_decode(z, return_dict=return_dict)

        z = self.post_quant_conv(z)
        dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    @apply_forward_hook
    def decode(self, z: torch.FloatTensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def blend_v(self, a, b, blend_extent):
        blend_extent = min(a.shape[2], b.shape[2], blend_extent)
        for y in range(blend_extent):
            b[:, :, y, :] = a[:, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, y, :] * (y / blend_extent)
        return b

    def blend_h(self, a, b, blend_extent):
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, x] = a[:, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, x] * (x / blend_extent)
        return b

    def tiled_encode(self, x: torch.FloatTensor, return_dict: bool = True) -> AutoencoderKLOutput:
        r"""Encode a batch of images using a tiled encoder.

        When this option is enabled, the VAE will split the input tensor into tiles to compute encoding in several
        steps. This is useful to keep memory use constant regardless of image size. The end result of tiled encoding is
        different from non-tiled encoding because each tile uses a different encoder. To avoid tiling artifacts, the
        tiles overlap and are blended together to form a smooth output. You may still see tile-sized changes in the
        output, but they should be much less noticeable.

        Args:
            x (`torch.FloatTensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
            [`~models.autoencoder_kl.AutoencoderKLOutput`] or `tuple`:
                If return_dict is True, a [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain
                `tuple` is returned.
        """
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        # Split the image into 512x512 tiles and encode them separately.
        rows = []
        for i in range(0, x.shape[2], overlap_size):
            row = []
            for j in range(0, x.shape[3], overlap_size):
                tile = x[:, :, i : i + self.tile_sample_min_size, j : j + self.tile_sample_min_size]
                tile = self.encoder(tile)
                tile = self.quant_conv(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=3))

        moments = torch.cat(result_rows, dim=2)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def tiled_decode(self, z: torch.FloatTensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        r"""
        Decode a batch of images using a tiled decoder.

        Args:
            z (`torch.FloatTensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        # Split z into overlapping 64x64 tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, z.shape[2], overlap_size):
            row = []
            for j in range(0, z.shape[3], overlap_size):
                tile = z[:, :, i : i + self.tile_latent_min_size, j : j + self.tile_latent_min_size]
                tile = self.post_quant_conv(tile)
                decoded = self.decoder(tile)
                row.append(decoded)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=3))

        dec = torch.cat(result_rows, dim=2)
        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def forward(
        self,
        sample: torch.FloatTensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        r"""
        Args:
            sample (`torch.FloatTensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z).sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)