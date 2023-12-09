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

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unet_2d_blocks import UNetMidBlock2D, get_down_block, get_up_block
from diffusers.models.unet_2d import UNet2DOutput

from unet2d_dual_blocks import SeparableAttnDownBlock2D, SeparableMidBlock2D, SeparableAttnUpBlock2D

class UNet2DDualModel(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,
        in_channels: int = 2,
        out_channels: int = 2,
        time_embedding_type: str = "positional",
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        block_out_channels: Tuple[int] = (128, 192, 320, 512),
        layers_per_block: int = 2,
        add_mid_attention: bool = True,
        layers_per_mid_block: int = 1,
        mid_block_scale_factor: float = 1,
        mid_block_bottleneck_channels: int = 0,
        downsample_padding: int = 1,
        downsample_type: str = "conv",
        upsample_type: str = "conv",
        act_fn: str = "silu",
        attention_num_heads: Union[int, Tuple[int]] = (8,12,20,32),
        separate_attn_dim_down: Tuple[int] = (2,3),
        separate_attn_dim_up: Tuple[int] = (3,2,3),
        separate_attn_dim_mid: Tuple[int] = (0,),
        double_attention: Union[bool, Tuple[bool]] = False,
        pre_attention: Union[bool, Tuple[bool]] = False,
        norm_num_groups: Union[int, Tuple[int]] = 32,
        norm_eps: float = 1e-5,
        resnet_time_scale_shift: str = "default",
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        dropout: Union[float, Tuple[float]] = 0.0,
        conv_size = (3,3),
        no_conv_in: bool = False,
        freq_embedding_dim: Union[int, Tuple[int]] = 0,
        time_embedding_dim: Union[int, Tuple[int]] = 0,
        add_attention: Union[bool, Tuple[bool]] = True,
        use_skip_samples: bool = True,
        last_global_step: int = 0,
    ):
        super().__init__()

        self.sample_size = sample_size
        self.freq_embedding_dim = freq_embedding_dim
        self.time_embedding_dim = time_embedding_dim
        self.use_skip_samples = use_skip_samples

        time_embed_dim = block_out_channels[0] * 4

        # Check inputs
        if not isinstance(attention_num_heads, int) and len(attention_num_heads) != len(block_out_channels):
            raise ValueError(
                f"Must provide the same number of `attention_num_heads` as `block_out_channels`. `attention_num_heads`: {attention_num_heads}. `block_out_channels`: {block_out_channels}."
            )   

        if not isinstance(norm_num_groups, int) and len(norm_num_groups) != len(block_out_channels):
            raise ValueError(
                f"Must provide the same number of `norm_num_groups` as `block_out_channels`. `norm_num_groups`: {norm_num_groups}. `block_out_channels`: {block_out_channels}."
            )
        
        if not isinstance(dropout, float) and len(dropout) != len(block_out_channels):
            raise ValueError(
                f"Must provide the same number of `dropout` as `block_out_channels`. `dropout`: {dropout}. `block_out_channels`: {block_out_channels}."
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
        
        if not isinstance(add_attention, bool) and len(add_attention) != len(block_out_channels):
            raise ValueError(
                f"Must provide the same number of `add_attention` as `block_out_channels`. `add_attention`: {add_attention}. `block_out_channels`: {block_out_channels}."
            )
        
        # input
        if no_conv_in:
            self.conv_in = nn.Identity()
        else:
            #conv_in_kernel_size = conv_size
            conv_in_kernel_size = (3,3)
            self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=conv_in_kernel_size, padding=(conv_in_kernel_size[0]//2, conv_in_kernel_size[1]//2))

        # time
        if time_embedding_type == "fourier":
            self.time_proj = GaussianFourierProjection(embedding_size=block_out_channels[0], scale=16)
            timestep_input_dim = 2 * block_out_channels[0]
        elif time_embedding_type == "positional":
            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # class embedding
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        else:
            self.class_embedding = None

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        if isinstance(attention_num_heads, int):
            attention_num_heads = (attention_num_heads,) * len(block_out_channels)
        if isinstance(norm_num_groups, int):
            norm_num_groups = (norm_num_groups,) * len(block_out_channels)
        if isinstance(dropout, float):
            dropout = (dropout,) * len(block_out_channels)
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
        if isinstance(add_attention, bool):
            add_attention = (add_attention,) * len(block_out_channels)

        # down
        output_channel = block_out_channels[0]
        for i in range(len(block_out_channels)):
            input_channel = output_channel
            output_channel = block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1
            if is_final_block is True: _downsample_type = None
            else: _downsample_type = downsample_type or "conv"  # default to 'conv'
            _norm_num_groups = norm_num_groups[i]
            _attention_num_heads = attention_num_heads[i]
            _dropout = dropout[i]
            _double_attention = double_attention[i]
            _pre_attention = pre_attention[i]
            _freq_embedding_dim = freq_embedding_dim[i]
            _time_embedding_dim = time_embedding_dim[i]
            _add_attention = add_attention[i]

            down_block = SeparableAttnDownBlock2D(
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=_norm_num_groups,
                attention_num_heads=_attention_num_heads,
                downsample_padding=downsample_padding,
                resnet_time_scale_shift=resnet_time_scale_shift,
                downsample_type=_downsample_type,
                separate_attn_dim=separate_attn_dim_down,
                double_attention=_double_attention,
                pre_attention=_pre_attention,
                dropout=_dropout,
                conv_size=conv_size,
                freq_embedding_dim=_freq_embedding_dim,
                time_embedding_dim=_time_embedding_dim,
                add_attention=_add_attention,
                return_skip_samples=use_skip_samples,
            )
            self.down_blocks.append(down_block)

        # mid
        _norm_num_groups = norm_num_groups[-1]
        _attention_num_heads = attention_num_heads[-1]
        _dropout = dropout[-1]
        _double_attention = double_attention[-1]
        _pre_attention = pre_attention[-1]
        _freq_embedding_dim = freq_embedding_dim[-1]
        _time_embedding_dim = time_embedding_dim[-1]

        self.mid_block = SeparableMidBlock2D(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attention_num_heads=_attention_num_heads,
            resnet_groups=_norm_num_groups,
            add_attention=add_mid_attention,
            pre_attention=_pre_attention,
            separate_attn_dim=separate_attn_dim_mid,
            double_attention=_double_attention,
            dropout=_dropout,
            conv_size=conv_size,
            num_layers=layers_per_mid_block,
            freq_embedding_dim=_freq_embedding_dim,
            time_embedding_dim=_time_embedding_dim,
            mid_block_bottleneck_channels=mid_block_bottleneck_channels,
        )

        reversed_attention_num_heads = list(reversed(attention_num_heads))
        reversed_norm_num_groups = list(reversed(norm_num_groups))
        reversed_dropout = list(reversed(dropout))
        reversed_double_attention = list(reversed(double_attention))
        reversed_pre_attention = list(reversed(pre_attention))
        reversed_freq_embedding_dim = list(reversed(freq_embedding_dim))
        reversed_time_embedding_dim = list(reversed(time_embedding_dim))
        reversed_add_attention = list(reversed(add_attention))

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i in range(len(block_out_channels)):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]
            if mid_block_bottleneck_channels > 0 and i == 0:
                prev_output_channel = mid_block_bottleneck_channels

            is_final_block = i == len(block_out_channels) - 1
            if is_final_block is True: _upsample_type = None
            else: _upsample_type = upsample_type or "conv"  # default to 'conv'
            _norm_num_groups = reversed_norm_num_groups[i]
            _attention_num_heads = reversed_attention_num_heads[i]
            _dropout = reversed_dropout[i]
            _double_attention = reversed_double_attention[i]
            _pre_attention = reversed_pre_attention[i]
            _freq_embedding_dim = reversed_freq_embedding_dim[i]
            _time_embedding_dim = reversed_time_embedding_dim[i]
            _add_attention = reversed_add_attention[i]

            up_block = SeparableAttnUpBlock2D(
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=_norm_num_groups,
                attention_num_heads=_attention_num_heads,
                resnet_time_scale_shift=resnet_time_scale_shift,
                upsample_type=_upsample_type,
                separate_attn_dim=separate_attn_dim_up,
                double_attention=_double_attention,
                pre_attention=_pre_attention,
                dropout=_dropout,
                conv_size=conv_size,
                freq_embedding_dim=_freq_embedding_dim,
                time_embedding_dim=_time_embedding_dim,
                add_attention=_add_attention,
                use_skip_samples=use_skip_samples,
            )
            self.up_blocks.append(up_block)

            prev_output_channel = output_channel

        # out
        _num_groups_out = norm_num_groups[0]
        if _num_groups_out < 0: _num_groups_out = block_out_channels[0] // abs(_num_groups_out)
        if _num_groups_out != 0:
            self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=_num_groups_out, eps=norm_eps)
        else:
            self.conv_norm_out = nn.Identity()
        self.conv_act = nn.SiLU()

        #conv_out_kernel_size = conv_size
        conv_out_kernel_size = (3,3)
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=conv_out_kernel_size, padding=(conv_out_kernel_size[0]//2, conv_out_kernel_size[1]//2))

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DOutput, Tuple]:
        r"""
        The [`UNet2DModel`] forward method.

        Args:
            sample (`torch.FloatTensor`):
                The noisy input tensor with the following shape `(batch, channel, height, width)`.
            timestep (`torch.FloatTensor` or `float` or `int`): The number of timesteps to denoise an input.
            class_labels (`torch.FloatTensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_2d.UNet2DOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_2d.UNet2DOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unet_2d.UNet2DOutput`] is returned, otherwise a `tuple` is
                returned where the first element is the sample tensor.
        """

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when doing class conditioning")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # 2. pre-process
        if self.use_skip_samples:
            skip_sample = sample
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)

        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(hidden_states=sample, temb=emb, skip_sample=skip_sample)
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(sample, emb)

        # 5. up
        skip_sample = None
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(sample, res_samples, emb, skip_sample)
            else:
                sample = upsample_block(sample, res_samples, emb)

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if skip_sample is not None:
            sample += skip_sample

        if self.config.time_embedding_type == "fourier":
            timesteps = timesteps.reshape((sample.shape[0], *([1] * len(sample.shape[1:]))))
            sample = sample / timesteps

        if not return_dict:
            return (sample,)

        return UNet2DOutput(sample=sample)
