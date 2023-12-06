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

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from typing import Optional
from functools import partial

from diffusers.utils import logging
from attention_processor_dual import SeparableAttention
from diffusers.models.resnet import Upsample2D, Downsample2D, FirDownsample2D, FirUpsample2D

from diffusers.models.attention import AdaGroupNorm
from diffusers.models.attention_processor import SpatialNorm
from diffusers.models.resnet import upsample_2d, downsample_2d

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def get_activation(act_fn):
    if act_fn in ["swish", "silu"]:
        return nn.SiLU()
    elif act_fn == "mish":
        return nn.Mish()
    elif act_fn == "gelu":
        return nn.GELU()
    elif act_fn == "relu":
        return nn.ReLU()
    elif act_fn == "sin":
        return torch.sin
    elif act_fn == "sinc":
        return torch.sinc
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")

class DualResnetBlock2D(nn.Module):
    r"""
    A Resnet block.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer. If None, same as `in_channels`.
        dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
        temb_channels (`int`, *optional*, default to `512`): the number of channels in timestep embedding.
        groups (`int`, *optional*, default to `32`): The number of groups to use for the first normalization layer.
        groups_out (`int`, *optional*, default to None):
            The number of groups to use for the second normalization layer. if set to None, same as `groups`.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
        non_linearity (`str`, *optional*, default to `"swish"`): the activation function to use.
        time_embedding_norm (`str`, *optional*, default to `"default"` ): Time scale shift config.
            By default, apply timestep embedding conditioning with a simple shift mechanism. Choose "scale_shift" or
            "ada_group" for a stronger conditioning with scale and shift.
        kernel (`torch.FloatTensor`, optional, default to None): FIR filter, see
            [`~models.resnet.FirUpsample2D`] and [`~models.resnet.FirDownsample2D`].
        output_scale_factor (`float`, *optional*, default to be `1.0`): the scale factor to use for the output.
        use_in_shortcut (`bool`, *optional*, default to `True`):
            If `True`, add a 1x1 nn.conv2d layer for skip-connection.
        up (`bool`, *optional*, default to `False`): If `True`, add an upsample layer.
        down (`bool`, *optional*, default to `False`): If `True`, add a downsample layer.
        conv_shortcut_bias (`bool`, *optional*, default to `True`):  If `True`, adds a learnable bias to the
            `conv_shortcut` output.
        conv_2d_out_channels (`int`, *optional*, default to `None`): the number of channels in the output.
            If None, same as `out_channels`.
    """

    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=512,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-6,
        non_linearity="swish",
        skip_time_act=False,
        time_embedding_norm="default",  # default, scale_shift, ada_group, spatial
        kernel=None,
        output_scale_factor=1.0,
        use_in_shortcut=None,
        up=False,
        down=False,
        conv_shortcut_bias: bool = True,
        conv_2d_out_channels: Optional[int] = None,
        conv_size = (3,3),
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm
        self.skip_time_act = skip_time_act

        if groups < 0:
            groups = in_channels // abs(groups)
            groups_out = out_channels // abs(groups)
        else:
            if groups_out is None:
                groups_out = groups

        if self.time_embedding_norm == "ada_group":
            self.norm1 = AdaGroupNorm(temb_channels, in_channels, groups, eps=eps)
        elif self.time_embedding_norm == "spatial":
            self.norm1 = SpatialNorm(in_channels, temb_channels)
        else:
            if groups > 0:
                self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
            else:
                self.norm1 = nn.Identity()

        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=conv_size, stride=1, padding=(conv_size[0]//2,conv_size[1]//2))

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                self.time_emb_proj = torch.nn.Linear(temb_channels, out_channels)
            elif self.time_embedding_norm == "scale_shift":
                self.time_emb_proj = torch.nn.Linear(temb_channels, 2 * out_channels)
            elif self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
                self.time_emb_proj = None
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")
        else:
            self.time_emb_proj = None

        if self.time_embedding_norm == "ada_group":
            self.norm2 = AdaGroupNorm(temb_channels, out_channels, groups_out, eps=eps)
        elif self.time_embedding_norm == "spatial":
            self.norm2 = SpatialNorm(out_channels, temb_channels)
        else:
            if groups_out > 0:
                self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
            else:
                self.norm2 = nn.Identity()

        self.dropout = torch.nn.Dropout(dropout)
        conv_2d_out_channels = conv_2d_out_channels or out_channels
        self.conv2 = torch.nn.Conv2d(out_channels, conv_2d_out_channels, kernel_size=conv_size, stride=1, padding=(conv_size[0]//2,conv_size[1]//2))

        self.nonlinearity = get_activation(non_linearity)

        self.upsample = self.downsample = None
        if self.up:
            if kernel == "fir":
                fir_kernel = (1, 3, 3, 1)
                self.upsample = lambda x: upsample_2d(x, kernel=fir_kernel)
            elif kernel == "sde_vp":
                self.upsample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
            else:
                self.upsample = Upsample2D(in_channels, use_conv=False)
        elif self.down:
            if kernel == "fir":
                fir_kernel = (1, 3, 3, 1)
                self.downsample = lambda x: downsample_2d(x, kernel=fir_kernel)
            elif kernel == "sde_vp":
                self.downsample = partial(F.avg_pool2d, kernel_size=2, stride=2)
            else:
                self.downsample = Downsample2D(in_channels, use_conv=False, padding=1, name="op")

        self.use_in_shortcut = self.in_channels != conv_2d_out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = torch.nn.Conv2d(
                in_channels, conv_2d_out_channels, kernel_size=1, stride=1, padding=0, bias=conv_shortcut_bias
            )

    def forward(self, input_tensor, temb):
        hidden_states = input_tensor

        if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
            hidden_states = self.norm1(hidden_states, temb)
        else:
            hidden_states = self.norm1(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
            hidden_states = self.norm2(hidden_states, temb)
        else:
            hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor

class SeparableAttnDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 2,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "silu",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attention_num_heads=8,
        output_scale_factor=1.0,
        downsample_padding=1,
        downsample_type="conv",
        separate_attn_dim=(2,3,),
        double_attention=False,
        pre_attention=False,
        conv_size=(3,3),
        return_skip_samples=True,
        freq_embedding_dim=0,
        time_embedding_dim=0,
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.downsample_type = downsample_type
        self.separate_attn_dim = separate_attn_dim
        self.conv_size = conv_size
        self.double_attention = double_attention
        self.pre_attention = pre_attention
        self.return_skip_samples = return_skip_samples
        self.freq_embedding_dim = freq_embedding_dim
        self.time_embedding_dim = time_embedding_dim

        for i in range(num_layers):
            _in_channels = in_channels if i == 0 else out_channels
            resnets.append(
            DualResnetBlock2D(
                    in_channels=_in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    conv_size=conv_size,
                )
            )
        
        attn_block_count = 0
        for i in range(num_layers+int(self.pre_attention)):
            _channels = in_channels if i == 0 and self.pre_attention else out_channels
            input_channels = _channels
            for _ in range(2 if double_attention else 1):
                attn_dim = self.separate_attn_dim[attn_block_count]
                if attn_dim == 0:
                    _freq_embedding_dim = self.freq_embedding_dim
                    _time_embedding_dim = self.time_embedding_dim
                elif attn_dim == 2:
                    _freq_embedding_dim = 0
                    _time_embedding_dim = self.time_embedding_dim
                elif attn_dim == 3:
                    _freq_embedding_dim = self.freq_embedding_dim
                    _time_embedding_dim = 0
                else:
                    raise ValueError(f"attn_dim must be 2, 3, or 0. got {attn_dim}")
                attentions.append(
                    SeparableAttention(
                        input_channels,
                        freq_embedding_dim=_freq_embedding_dim,
                        time_embedding_dim=_time_embedding_dim,
                        heads=attention_num_heads,
                        dim_head=input_channels // attention_num_heads,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=resnet_groups,
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        dropout=dropout,
                        separate_attn_dim=attn_dim,
                    )
                )
                attn_block_count += 1

        if len(attentions) != len(separate_attn_dim):
            raise ValueError(f"separate_attn_dim must have the same length as attentions. got {len(separate_attn_dim)} and {len(attentions)}")
        
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if downsample_type == "conv":
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        elif downsample_type == "resnet":
            self.downsamplers = nn.ModuleList(
                [
                    DualResnetBlock2D(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        temb_channels=temb_channels,
                        eps=resnet_eps,
                        groups=resnet_groups,
                        dropout=dropout,
                        time_embedding_norm=resnet_time_scale_shift,
                        non_linearity=resnet_act_fn,
                        output_scale_factor=output_scale_factor,
                        pre_norm=resnet_pre_norm,
                        down=True,
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(self, hidden_states, temb=None, upsample_size=None):
        output_states = ()

        attn_block_count = 0
        if self.pre_attention:
            for _ in range(2 if self.double_attention else 1):
                hidden_states = self.attentions[attn_block_count](hidden_states)
                attn_block_count += 1

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)

            for _ in range(2 if self.double_attention else 1):
                hidden_states = self.attentions[attn_block_count](hidden_states)
                attn_block_count += 1

            if self.return_skip_samples:
                output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:      
                if self.downsample_type == "resnet":
                    hidden_states = downsampler(hidden_states, temb=temb)
                else:
                    hidden_states = downsampler(hidden_states)

            if self.return_skip_samples:
                output_states += (hidden_states,)

        return hidden_states, output_states


class SeparableAttnUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 2,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "silu",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attention_num_heads=8,
        output_scale_factor=1.0,
        upsample_type="conv",
        separate_attn_dim=(2,3,),
        double_attention=False,
        pre_attention=False,
        conv_size=(3,3),
        use_skip_samples=True,
        freq_embedding_dim=0,
        time_embedding_dim=0,
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.upsample_type = upsample_type
        self.separate_attn_dim = separate_attn_dim
        self.conv_size = conv_size
        self.double_attention = double_attention
        self.pre_attention = pre_attention
        self.use_skip_samples = use_skip_samples
        self.freq_embedding_dim = freq_embedding_dim
        self.time_embedding_dim = time_embedding_dim

        for i in range(num_layers):
            if self.use_skip_samples:
                res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            else:
                res_skip_channels = 0
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                DualResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    conv_size=conv_size,
                )
            )
        
        attn_block_count = 0
        for i in range(num_layers+int(self.pre_attention)):
            _channels = prev_output_channel if i == 0 and self.pre_attention else out_channels
            input_channels = _channels
            for _ in range(2 if double_attention else 1):
                attn_dim = self.separate_attn_dim[attn_block_count]
                if attn_dim == 0:
                    _freq_embedding_dim = self.freq_embedding_dim
                    _time_embedding_dim = self.time_embedding_dim
                elif attn_dim == 2:
                    _freq_embedding_dim = 0
                    _time_embedding_dim = self.time_embedding_dim
                elif attn_dim == 3:
                    _freq_embedding_dim = self.freq_embedding_dim
                    _time_embedding_dim = 0
                else:
                    raise ValueError(f"attn_dim must be 2, 3, or 0. got {attn_dim}")
                attentions.append(
                    SeparableAttention(
                        input_channels,
                        freq_embedding_dim=_freq_embedding_dim,
                        time_embedding_dim=_time_embedding_dim,
                        heads=attention_num_heads,
                        dim_head=input_channels // attention_num_heads,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=resnet_groups,
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        dropout=dropout,
                        separate_attn_dim=attn_dim,
                    )
                )
                attn_block_count += 1

        if len(attentions) != len(separate_attn_dim):
            raise ValueError(f"separate_attn_dim must have the same length as attentions. got {len(separate_attn_dim)} and {len(attentions)}")
        
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if upsample_type == "conv":
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        elif upsample_type == "resnet":
            self.upsamplers = nn.ModuleList(
                [
                    DualResnetBlock2D(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        temb_channels=temb_channels,
                        eps=resnet_eps,
                        groups=resnet_groups,
                        dropout=dropout,
                        time_embedding_norm=resnet_time_scale_shift,
                        non_linearity=resnet_act_fn,
                        output_scale_factor=output_scale_factor,
                        pre_norm=resnet_pre_norm,
                        up=True,
                    )
                ]
            )
        else:
            self.upsamplers = None

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None):
        
        attn_block_count = 0
        if self.pre_attention:
            for _ in range(2 if self.double_attention else 1):
                hidden_states = self.attentions[attn_block_count](hidden_states)
                attn_block_count += 1

        for resnet in self.resnets:
            if self.use_skip_samples:
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            
            hidden_states = resnet(hidden_states, temb)

            for _ in range(2 if self.double_attention else 1):
                hidden_states = self.attentions[attn_block_count](hidden_states)
                attn_block_count += 1

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                if self.upsample_type == "resnet":
                    hidden_states = upsampler(hidden_states, temb=temb)
                else:                
                    hidden_states = upsampler(hidden_states)

        return hidden_states

class SeparableMidBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_attention: bool = True,
        attention_num_heads=8,
        separate_attn_dim=(0,0),
        double_attention=False,
        pre_attention=False,
        conv_size=(3,3),
        freq_embedding_dim=0,
        time_embedding_dim=0,
        mid_block_bottleneck_channels=0,
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.add_attention = add_attention
        self.separate_attn_dim = separate_attn_dim
        self.double_attention = double_attention
        self.pre_attention = pre_attention
        self.freq_embedding_dim = freq_embedding_dim
        self.time_embedding_dim = time_embedding_dim
        self.mid_block_bottleneck_channels = mid_block_bottleneck_channels

        for i in range(num_layers+1):
            resnets.append(
                DualResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    conv_size=conv_size,
                )
            )

        if self.add_attention:
            attn_block_count = 0
            for _ in range(num_layers):
                _channels = in_channels
                input_channels = _channels
                for _ in range(2 if double_attention else 1):
                    attn_dim = self.separate_attn_dim[attn_block_count]
                    if attn_dim == 0:
                        _freq_embedding_dim = self.freq_embedding_dim
                        _time_embedding_dim = self.time_embedding_dim
                    elif attn_dim == 2:
                        _freq_embedding_dim = 0
                        _time_embedding_dim = self.time_embedding_dim
                    elif attn_dim == 3:
                        _freq_embedding_dim = self.freq_embedding_dim
                        _time_embedding_dim = 0
                    else:
                        raise ValueError(f"attn_dim must be 2, 3, or 0. got {attn_dim}")
                    attentions.append(
                        SeparableAttention(
                            input_channels,
                            freq_embedding_dim=_freq_embedding_dim,
                            time_embedding_dim=_time_embedding_dim,
                            heads=attention_num_heads,
                            dim_head=input_channels // attention_num_heads,
                            rescale_output_factor=output_scale_factor,
                            eps=resnet_eps,
                            norm_num_groups=resnet_groups,
                            residual_connection=True,
                            bias=True,
                            upcast_softmax=True,
                            dropout=dropout,
                            separate_attn_dim=attn_dim,
                        )
                    )
                    attn_block_count += 1

            if len(attentions) != len(separate_attn_dim):
                raise ValueError(f"separate_attn_dim must have the same length as attentions. got {len(separate_attn_dim)} and {len(attentions)}")
        
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if mid_block_bottleneck_channels > 0:
            self.conv_bottleneck = nn.Conv2d(in_channels, mid_block_bottleneck_channels, kernel_size=(3,3), padding=(1, 1))
        else:
            self.conv_bottleneck = None

    def forward(self, hidden_states, temb=None):
        
        attn_block_count = 0
        hidden_states = self.resnets[0](hidden_states, temb)
        resnets = self.resnets[1:]

        for resnet in resnets:

            if self.add_attention:
                for _ in range(2 if self.double_attention else 1):
                    hidden_states = self.attentions[attn_block_count](hidden_states)
                    attn_block_count += 1

            hidden_states = resnet(hidden_states, temb)

        if self.conv_bottleneck is not None:
            hidden_states = self.conv_bottleneck(hidden_states)
            
        return hidden_states