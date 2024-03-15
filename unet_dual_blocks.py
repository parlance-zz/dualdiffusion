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

import torch
from torch import nn
import torch.nn.functional as F

from diffusers.utils import logging

from diffusers.models.normalization import AdaGroupNorm
from diffusers.models.attention_processor import SpatialNorm

from attention_processor_dual import SeparableAttention
from dual_diffusion_utils import get_activation

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class Upsample(nn.Module):

    def __init__(self, channels, upsample_type="conv", out_channels=None, upsample_ratio=(2,2), interpolation="nearest"):
        super().__init__()

        if isinstance(upsample_ratio, int):
            upsample_ratio = (upsample_ratio,)
        if len(upsample_ratio) > 2:
            raise ValueError(f"upsample_ratio must be an int or a tuple of length 2. got {upsample_ratio}")
        
        self.channels = channels
        self.out_channels = out_channels or channels
        self.upsample_ratio = upsample_ratio
        self.upsample_type = upsample_type
        self.interpolation = interpolation

        if upsample_type == "conv_transpose":
            if len(upsample_ratio) == 1:
                conv_class = nn.ConvTranspose1d
            else:
                conv_class = nn.ConvTranspose2d

            conv_kernel_size = tuple(x * 2 if x > 1 else 1 for x in upsample_ratio)
            conv_stride = upsample_ratio
            conv_padding = tuple(x // 2 for x in upsample_ratio)
            self.conv = conv_class(channels, self.out_channels, conv_kernel_size, conv_stride, conv_padding)

        elif upsample_type == "conv":
            if len(upsample_ratio) == 1:
                conv_class = nn.Conv1d
            else:
                conv_class = nn.Conv2d

            conv_kernel_size = tuple(x*2 - 1 for x in upsample_ratio)
            conv_stride = 1
            conv_padding = tuple(x // 2 for x in conv_kernel_size)
            self.conv = conv_class(channels, self.out_channels, conv_kernel_size, conv_stride, conv_padding)

        elif upsample_type == "interpolate":
            self.conv = None
        else:
            raise ValueError(f"upsample_type must be conv, conv_transpose, or interpolate. got {upsample_type}")

    def forward(self, inputs):
        assert inputs.shape[1] == self.channels
        
        if self.upsample_type == "conv_transpose":
            return self.conv(inputs)

        outputs = F.interpolate(inputs, scale_factor=self.upsample_ratio, mode=self.interpolation)

        if self.upsample_type == "conv":
            outputs = self.conv(outputs)

        return outputs

class Downsample(nn.Module):

    def __init__(self, channels, downsample_type="conv", out_channels=None, downsample_ratio=(2,2)):
        super().__init__()

        if isinstance(downsample_ratio, int):
            downsample_ratio = (downsample_ratio,)
        if len(downsample_ratio) > 2:
            raise ValueError(f"downsample_ratio must be an int or a tuple of length 2. got {downsample_ratio}")
        
        self.channels = channels
        self.out_channels = out_channels or channels
        self.downsample_type = downsample_type
        self.downsample_ratio = downsample_ratio

        if downsample_type == "conv":
            if len(downsample_ratio) == 1:
                conv_class = nn.Conv1d
            else:
                conv_class = nn.Conv2d

            conv_kernel_size = tuple(x * 2 - 1 for x in downsample_ratio)
            conv_stride = downsample_ratio
            conv_padding = tuple(x // 2 for x in conv_kernel_size)
            self.conv = conv_class(channels, self.out_channels, conv_kernel_size, conv_stride, conv_padding)

        elif downsample_type == "avg":
            if len(downsample_ratio) == 1:
                avg_pool_class = nn.AvgPool1d
            else:
                avg_pool_class = nn.AvgPool2d

            self.avg_pool = avg_pool_class(kernel_size=downsample_ratio, stride=downsample_ratio)
        else:
            raise ValueError(f"downsample_type must be conv or avg. got {downsample_type}")

    def forward(self, inputs):
        assert inputs.shape[1] == self.channels

        if self.downsample_type == "conv":
            return self.conv(inputs)
        else:
            return self.avg_pool(inputs)
    
class DualResnetBlock(nn.Module):

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
        output_scale_factor=1.0,
        use_in_shortcut=None,
        conv_shortcut_bias: bool = True,
        conv_out_channels: Optional[int] = None,
        conv_size = (3,3),
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm
        self.skip_time_act = skip_time_act

        if isinstance(conv_size, int):
            conv_class = nn.Conv1d
            conv_padding = conv_size // 2
        else:
            if len(conv_size) != 2:
                raise ValueError(f"conv_size must be an int or a tuple of length 2. got {conv_size}")
            conv_class = nn.Conv2d
            conv_padding = (conv_size[0] // 2, conv_size[1] // 2)

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

        self.conv1 = conv_class(in_channels, out_channels, kernel_size=conv_size, stride=1, padding=conv_padding)

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
        conv_out_channels = conv_out_channels or out_channels
        self.conv2 = conv_class(out_channels, conv_out_channels, kernel_size=conv_size, stride=1, padding=conv_padding)

        self.nonlinearity = get_activation(non_linearity)

        self.use_in_shortcut = self.in_channels != conv_out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = conv_class(
                in_channels, conv_out_channels, kernel_size=1, stride=1, padding=0, bias=conv_shortcut_bias
            )

    def forward(self, input_tensor, temb):
        hidden_states = input_tensor

        if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
            hidden_states = self.norm1(hidden_states, temb)
        else:
            hidden_states = self.norm1(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
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

class SeparableAttnDownBlock(nn.Module):
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
        downsample_type="conv",
        separate_attn_dim=(2,3,),
        double_attention=False,
        pre_attention=False,
        conv_size=(3,3),
        freq_embedding_dim=0,
        time_embedding_dim=0,
        add_attention=True,
        downsample_ratio=(2,2),
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.downsample_type = downsample_type
        self.separate_attn_dim = separate_attn_dim
        self.conv_size = conv_size
        self.double_attention = double_attention
        self.pre_attention = pre_attention
        self.freq_embedding_dim = freq_embedding_dim
        self.time_embedding_dim = time_embedding_dim
        self.add_attention = add_attention

        for i in range(num_layers):
            _in_channels = in_channels if i == 0 else out_channels
            resnets.append(
            DualResnetBlock(
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
        
        self.resnets = nn.ModuleList(resnets)

        if self.add_attention:
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
        

        if downsample_type is not None:
            self.downsamplers = nn.ModuleList([Downsample(out_channels,
                                                          downsample_type=downsample_type,
                                                          out_channels=out_channels,
                                                          downsample_ratio=downsample_ratio)])
        else:
            self.downsamplers = None

    def forward(self, hidden_states, temb=None):
        output_states = ()

        if self.add_attention:
            attn_block_count = 0
            if self.pre_attention:
                for _ in range(2 if self.double_attention else 1):
                    hidden_states = self.attentions[attn_block_count](hidden_states)
                    attn_block_count += 1

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)

            if self.add_attention:
                for _ in range(2 if self.double_attention else 1):
                    hidden_states = self.attentions[attn_block_count](hidden_states)
                    attn_block_count += 1

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states

class SeparableAttnUpBlock(nn.Module):
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
        freq_embedding_dim=0,
        time_embedding_dim=0,
        add_attention=True,
        upsample_ratio=(2,2),
        use_skip_samples=True,
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
        self.add_attention = add_attention

        for i in range(num_layers):
            if self.use_skip_samples:
                res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            else:
                res_skip_channels = 0
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                DualResnetBlock(
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
        
        self.resnets = nn.ModuleList(resnets)

        if self.add_attention:
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
        
        if upsample_type is not None:
            self.upsamplers = nn.ModuleList([Upsample(out_channels,
                                                      upsample_type=upsample_type,
                                                      out_channels=out_channels,
                                                      upsample_ratio=upsample_ratio)])
        else:
            self.upsamplers = None

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None):
        
        if self.add_attention:
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

            if self.add_attention:
                for _ in range(2 if self.double_attention else 1):
                    hidden_states = self.attentions[attn_block_count](hidden_states)
                    attn_block_count += 1

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states

class SeparableMidBlock(nn.Module):
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

        for i in range(num_layers+1):
            resnets.append(
                DualResnetBlock(
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

        self.resnets = nn.ModuleList(resnets)
        
        if self.add_attention:
            attn_block_count = 0
            for _ in range(num_layers+int(self.pre_attention)):
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

    def forward(self, hidden_states, temb=None):
        
        attn_block_count = 0
        hidden_states = self.resnets[0](hidden_states, temb)
        resnets = self.resnets[1:]

        if self.pre_attention:
            for _ in range(2 if self.double_attention else 1):
                hidden_states = self.attentions[attn_block_count](hidden_states)
                attn_block_count += 1
                
        for resnet in resnets:

            hidden_states = resnet(hidden_states, temb)
            
            if self.add_attention:
                for _ in range(2 if self.double_attention else 1):
                    hidden_states = self.attentions[attn_block_count](hidden_states)
                    attn_block_count += 1
                    
        return hidden_states