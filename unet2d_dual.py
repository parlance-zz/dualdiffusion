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
from diffusers.models.embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unet_2d_blocks import UNetMidBlock2D, get_down_block, get_up_block
from diffusers.models.unet_2d import UNet2DOutput

from unet2d_dual_blocks import SeparableAttnDownBlock2D, SeparableMidBlock2D, SeparableAttnUpBlock2D

class UNet2DDualModel(ModelMixin, ConfigMixin):
    r"""
    A 2D UNet model that takes a noisy sample and a timestep and returns a sample shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample. Dimensions must be a multiple of `2 ** (len(block_out_channels) -
            1)`.
        in_channels (`int`, *optional*, defaults to 3): Number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 3): Number of channels in the output.
        center_input_sample (`bool`, *optional*, defaults to `False`): Whether to center the input sample.
        time_embedding_type (`str`, *optional*, defaults to `"positional"`): Type of time embedding to use.
        freq_shift (`int`, *optional*, defaults to 0): Frequency shift for Fourier time embedding.
        flip_sin_to_cos (`bool`, *optional*, defaults to `True`):
            Whether to flip sin to cos for Fourier time embedding.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D")`):
            Tuple of downsample block types.
        mid_block_type (`str`, *optional*, defaults to `"UNetMidBlock2D"`):
            Block type for middle of UNet, it can be either `UNetMidBlock2D` or `UnCLIPUNetMidBlock2D`.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D")`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(224, 448, 672, 896)`):
            Tuple of block output channels.
        layers_per_block (`int`, *optional*, defaults to `2`): The number of layers per block.
        mid_block_scale_factor (`float`, *optional*, defaults to `1`): The scale factor for the mid block.
        downsample_padding (`int`, *optional*, defaults to `1`): The padding for the downsample convolution.
        downsample_type (`str`, *optional*, defaults to `conv`):
            The downsample type for downsampling layers. Choose between "conv" and "resnet"
        upsample_type (`str`, *optional*, defaults to `conv`):
            The upsample type for upsampling layers. Choose between "conv" and "resnet"
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        attention_head_dim (`int`, *optional*, defaults to `8`): The attention head dimension.
        norm_num_groups (`int`, *optional*, defaults to `32`): The number of groups for normalization.
        norm_eps (`float`, *optional*, defaults to `1e-5`): The epsilon for normalization.
        resnet_time_scale_shift (`str`, *optional*, defaults to `"default"`): Time scale shift config
            for ResNet blocks (see [`~models.resnet.ResnetBlock2D`]). Choose from `default` or `scale_shift`.
        class_embed_type (`str`, *optional*, defaults to `None`):
            The type of class embedding to use which is ultimately summed with the time embeddings. Choose from `None`,
            `"timestep"`, or `"identity"`.
        num_class_embeds (`int`, *optional*, defaults to `None`):
            Input dimension of the learnable embedding matrix to be projected to `time_embed_dim` when performing class
            conditioning with `class_embed_type` equal to `None`.
    """

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,
        in_channels: int = 2,
        out_channels: int = 2,
        time_embedding_type: str = "positional",
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        down_block_types: Tuple[str] = ("SeparableAttnDownBlock2D", "SeparableAttnDownBlock2D", "SeparableAttnDownBlock2D", "SeparableAttnDownBlock2D"),
        up_block_types: Tuple[str] = ("SeparableAttnUpBlock2D", "SeparableAttnUpBlock2D", "SeparableAttnUpBlock2D", "SeparableAttnUpBlock2D"),
        block_out_channels: Tuple[int] = (128, 192, 320, 512),
        layers_per_block: int = 2,
        layers_per_mid_block: int = 1,
        mid_block_scale_factor: float = 1,
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
        add_mid_attention: bool = True,
        use_separable_mid_block: bool = False,
        norm_num_groups: Union[int, Tuple[int]] = 32,
        norm_eps: float = 1e-5,
        resnet_time_scale_shift: str = "default",
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        dropout: Union[float, Tuple[float]] = 0.0,
        conv_size = (3,3),
        no_conv_in: bool = False,
    ):
        super().__init__()

        self.sample_size = sample_size

        time_embed_dim = block_out_channels[0] * 4

        # Check inputs
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(attention_num_heads, int) and len(attention_num_heads) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `attention_num_heads` as `down_block_types`. `attention_num_heads`: {attention_num_heads}. `down_block_types`: {down_block_types}."
            )   

        if not isinstance(norm_num_groups, int) and len(norm_num_groups) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `norm_num_groups` as `down_block_types`. `norm_num_groups`: {norm_num_groups}. `down_block_types`: {down_block_types}."
            )
        
        if not isinstance(dropout, float) and len(dropout) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `dropout` as `down_block_types`. `dropout`: {dropout}. `down_block_types`: {down_block_types}."
            )
        
        if not isinstance(double_attention, bool) and len(double_attention) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `double_attention` as `down_block_types`. `double_attention`: {double_attention}. `down_block_types`: {down_block_types}."
            )
        
        if not isinstance(pre_attention, bool) and len(pre_attention) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `pre_attention` as `down_block_types`. `pre_attention`: {pre_attention}. `down_block_types`: {down_block_types}."
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
            attention_num_heads = (attention_num_heads,) * len(down_block_types)
        if isinstance(norm_num_groups, int):
            norm_num_groups = (norm_num_groups,) * len(down_block_types)
        if isinstance(dropout, float):
            dropout = (dropout,) * len(down_block_types)
        if isinstance(separate_attn_dim_down, int):
            separate_attn_dim_down = (separate_attn_dim_down,)
        if isinstance(separate_attn_dim_up, int):
            separate_attn_dim_up = (separate_attn_dim_up,)
        if isinstance(double_attention, bool):
            double_attention = (double_attention,) * len(down_block_types)
        if isinstance(pre_attention, bool):
            pre_attention = (pre_attention,) * len(down_block_types)

        def set_dropout_p(model, p_value):
            for module in model.children():
                if isinstance(module, torch.nn.Dropout):
                    module.p = p_value
                else:
                    set_dropout_p(module, p_value)

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
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

            if down_block_type == "SeparableAttnDownBlock2D":
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
                )
            else:
                down_block = get_down_block(
                    down_block_type,
                    num_layers=layers_per_block,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    add_downsample=not is_final_block,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=_norm_num_groups,
                    attention_head_dim=output_channel // _attention_num_heads,
                    downsample_padding=downsample_padding,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    downsample_type=_downsample_type,
                )
                set_dropout_p(down_block, _dropout) # default blocks don't apply dropout to all modules

            self.down_blocks.append(down_block)

        # mid
        _norm_num_groups = norm_num_groups[-1]
        _attention_num_heads = attention_num_heads[-1]
        _dropout = dropout[-1]
        _double_attention = double_attention[-1]
        _pre_attention = pre_attention[-1]
        if use_separable_mid_block:
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
            )
        else:
            self.mid_block = UNetMidBlock2D(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                attention_head_dim=block_out_channels[-1] // _attention_num_heads,
                resnet_groups=_norm_num_groups,
                add_attention=add_mid_attention,
                dropout=_dropout,
                num_layers=layers_per_mid_block,
            )
            set_dropout_p(self.mid_block, _dropout) # default blocks don't apply dropout to all modules

        reversed_attention_num_heads = list(reversed(attention_num_heads))
        reversed_norm_num_groups = list(reversed(norm_num_groups))
        reversed_dropout = list(reversed(dropout))
        reversed_double_attention = list(reversed(double_attention))
        reversed_pre_attention = list(reversed(pre_attention))

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            is_final_block = i == len(block_out_channels) - 1
            if is_final_block is True: _upsample_type = None
            else: _upsample_type = upsample_type or "conv"  # default to 'conv'
            _norm_num_groups = reversed_norm_num_groups[i]
            _attention_num_heads = reversed_attention_num_heads[i]
            _dropout = reversed_dropout[i]
            _double_attention = reversed_double_attention[i]
            _pre_attention = reversed_pre_attention[i]

            if up_block_type == "SeparableAttnUpBlock2D":
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
                )
            else:
                up_block = get_up_block(
                    up_block_type,
                    num_layers=layers_per_block + 1,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    temb_channels=time_embed_dim,
                    add_upsample=not is_final_block,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=_norm_num_groups,
                    attention_head_dim=output_channel // _attention_num_heads,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    upsample_type=_upsample_type,
                )
                set_dropout_p(up_block, _dropout) # default blocks don't apply dropout to all modules

            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        _num_groups_out = norm_num_groups[0]
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=_num_groups_out, eps=norm_eps)
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
