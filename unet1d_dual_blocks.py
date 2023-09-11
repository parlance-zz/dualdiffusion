from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from diffusers.utils import logging
from diffusers.models.activations import get_activation
from diffusers.models.attention import AdaGroupNorm
from diffusers.models.attention_processor import Attention, SpatialNorm
from diffusers.models.unet_1d_blocks import ResConvBlock, Downsample1d, Upsample1d

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

"""
def raw_to_log_scale(samples, u=255.):
    return torch.sgn(samples) * torch.log(1. + u * samples.abs()) / np.log(1. + u)

def log_scale_to_raw(samples, u=255.):
    return torch.sgn(samples) * ((1. + u) ** samples.abs() - 1.) / u
"""

def to_freq(x):
    #return x

    #"""
    original_shape = x.shape
    #x = x.view(-1, 2, x.shape[2])
    x = x.reshape(-1, 2, x.shape[2])
    x = x.permute(0, 2, 1).contiguous()
    x = torch.fft.fft(torch.view_as_complex(x), norm="ortho")
    x = torch.view_as_real(x)
    x = x.permute(0, 2, 1).contiguous()
    #"""

    """
    original_shape = x.shape
    x = x.view(-1, x.shape[1]//2, 2, x.shape[2])
    x = x.permute(0, 1, 3, 2).contiguous()
    x = torch.fft.fft2(torch.view_as_complex(x), norm="ortho")
    x = torch.view_as_real(x)
    x = x.permute(0, 1, 3, 2).contiguous()
    """

    return x.view(original_shape)

def to_spatial(x):
    #return x

    #"""
    original_shape = x.shape
    #x = x.view(-1, 2, x.shape[2])
    x = x.reshape(-1, 2, x.shape[2])
    x = x.permute(0, 2, 1).contiguous()
    x = torch.fft.ifft(torch.view_as_complex(x), norm="ortho")
    x = torch.view_as_real(x)
    x = x.permute(0, 2, 1).contiguous()
    #"""

    """
    original_shape = x.shape
    x = x.view(-1, x.shape[1]//2, 2, x.shape[2])
    x = x.permute(0, 1, 3, 2).contiguous()
    x = torch.fft.ifft2(torch.view_as_complex(x), norm="ortho")
    x = torch.view_as_real(x)
    x = x.permute(0, 1, 3, 2).contiguous()
    """

    return x.view(original_shape)

class ResnetBlock1D(nn.Module):
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
        kernel="lanczos3",
        output_scale_factor=1.0,
        use_in_shortcut=None,
        up=False,
        down=False,
        conv_shortcut_bias: bool = True,
        conv_1d_out_channels: Optional[int] = None,
        conv_size: int = 3,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.kernel = kernel
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm
        self.skip_time_act = skip_time_act

        if groups_out is None:
            groups_out = groups

        if self.time_embedding_norm == "ada_group":
            self.norm1 = AdaGroupNorm(temb_channels, in_channels, groups, eps=eps)
        elif self.time_embedding_norm == "spatial":
            self.norm1 = SpatialNorm(in_channels, temb_channels)
        else:
            self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=conv_size, stride=1, padding=conv_size//2)

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
            self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)

        self.dropout = torch.nn.Dropout(dropout)
        conv_1d_out_channels = conv_1d_out_channels or out_channels
        self.conv2 = torch.nn.Conv1d(out_channels, conv_1d_out_channels, kernel_size=conv_size, stride=1, padding=conv_size//2)

        self.nonlinearity = get_activation(non_linearity)

        self.upsample = self.downsample = None
        if self.up:
            self.upsample = Upsample1d(self.kernel, pad_mode="constant")
        elif self.down:
            self.downsample = Downsample1d(self.kernel, pad_mode="constant")

        self.use_in_shortcut = self.in_channels != conv_1d_out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = torch.nn.Conv1d(
                in_channels, conv_1d_out_channels, kernel_size=1, stride=1, padding=0, bias=conv_shortcut_bias
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
            temb = self.time_emb_proj(temb)[:, :, None]

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

class DualMidBlock1D(nn.Module):
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
        add_attention: bool = True,
        attention_head_dim=1,
        output_scale_factor=1.0,
        conv_size: int = 3,
        use_fft: bool = False,
    ):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        self.add_attention = add_attention
        self.conv_size = conv_size
        self.num_layers = num_layers
        self.use_fft = use_fft
        
        # there is always at least one resnet
        resnets = [
            ResnetBlock1D(
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
        ]
        attentions = []

        if attention_head_dim is None:
            logger.warn(
                f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {in_channels}."
            )
            attention_head_dim = in_channels

        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    Attention(
                        in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=resnet_groups if resnet_time_scale_shift == "default" else None,
                        spatial_norm_dim=temb_channels if resnet_time_scale_shift == "spatial" else None,
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                        dropout=dropout,
                    )
                )
            else:
                attentions.append(None)

            resnets.append(
                ResnetBlock1D(
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

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb=None, io=0):

        if ((io % 2) == 0) and self.use_fft:
            hidden_states = to_freq(hidden_states)
            
        hidden_states = self.resnets[0](hidden_states, temb)

        if ((io % 2) == 0) and self.use_fft:
            hidden_states = to_spatial(hidden_states)
        
        i = 1
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            
            if (((i+io) % 2) == 0) and self.use_fft:
                hidden_states = to_freq(hidden_states)

            if attn is not None:
                hidden_states = attn(hidden_states.unsqueeze(-1), temb=temb).squeeze(-1)
            hidden_states = resnet(hidden_states, temb)

            if (((i+io) % 2) == 0) and self.use_fft:
                hidden_states = to_spatial(hidden_states)

            i += 1

        return hidden_states, i
    
class DualDownBlock1D(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 temb_channels: int,
                 dropout: float = 0.0,
                 num_layers: int = 1,
                 resnet_eps: float = 1e-6,
                 resnet_time_scale_shift: str = "default",
                 resnet_act_fn: str = "swish",
                 resnet_groups: int = 32,
                 resnet_pre_norm: bool = True,
                 attention_head_dim=1,
                 output_scale_factor=1.0,
                 downsample_type=None,
                 add_attention: bool = False,
                 conv_size: int = 3,
                 use_fft: bool = False,
                 ):
        super().__init__()
        
        resnets = []
        attentions = []

        self.downsample_type = downsample_type
        self.add_attention = add_attention
        self.conv_size = conv_size
        self.num_layers = num_layers
        self.use_fft = use_fft

        if attention_head_dim is None:
            logger.warn(
                f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {out_channels}."
            )
            attention_head_dim = out_channels

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock1D(
                    in_channels=in_channels,
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
            if self.add_attention:
                attentions.append(
                    Attention(
                        out_channels,
                        heads=out_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=resnet_groups,
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                        dropout=dropout,
                    )
                )
            else:
                attentions.append(None)
                
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)


        if downsample_type is None:
            self.downsamplers = None
        elif downsample_type == "resnet":
            self.downsamplers = nn.ModuleList(
                [
                    ResnetBlock1D(
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
        elif downsample_type == "kernel":
            self.downsamplers = nn.ModuleList([
                Downsample1d("lanczos3", pad_mode="constant")
            ])
        else:
            raise ValueError(f"Unknown downsample_type : {downsample_type} ")

    def forward(self, hidden_states, temb=None, upsample_size=None, io=0):
        output_states = ()

        i = 0
        for resnet, attn in zip(self.resnets, self.attentions):

            if (((i+io) % 2) == 0) and self.use_fft:
                hidden_states = to_freq(hidden_states)
                
            hidden_states = resnet(hidden_states, temb)
            if attn is not None:
                hidden_states = attn(hidden_states.unsqueeze(-1)).squeeze(-1)

            if (((i+io) % 2) == 0) and self.use_fft:
                hidden_states = to_spatial(hidden_states)

            output_states = output_states + (hidden_states,)
            i += 1

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states, temb=temb)

            output_states += (hidden_states,)

        return hidden_states, output_states, i

class DualUpBlock1D(nn.Module):
    def __init__(self,
                 in_channels: int,
                 prev_output_channel: int,
                 out_channels: int,
                 temb_channels: int,
                 dropout: float = 0.0,
                 num_layers: int = 1,
                 resnet_eps: float = 1e-6,
                 resnet_time_scale_shift: str = "default",
                 resnet_act_fn: str = "swish",
                 resnet_groups: int = 32,
                 resnet_pre_norm: bool = True,
                 attention_head_dim=1,
                 output_scale_factor=1.0,
                 upsample_type=None,
                 add_attention: bool = False,
                 conv_size: int = 3,
                 use_fft: bool = False,
                 ):
        super().__init__()
        resnets = []
        attentions = []

        self.upsample_type = upsample_type
        self.add_attention = add_attention
        self.conv_size = conv_size
        self.num_layers = num_layers
        self.use_fft = use_fft

        if attention_head_dim is None:
            logger.warn(
                f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {out_channels}."
            )
            attention_head_dim = out_channels

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock1D(
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
            if self.add_attention:
                attentions.append(
                    Attention(
                        out_channels,
                        heads=out_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=resnet_groups,
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                        dropout=dropout,
                    )
                )
            else:
                attentions.append(None)

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if upsample_type is None:
            self.upsamplers = None
        elif upsample_type == "resnet":
            self.upsamplers = nn.ModuleList(
                [
                    ResnetBlock1D(
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
        elif upsample_type == "kernel":
            self.upsamplers = nn.ModuleList([
                Upsample1d("lanczos3", pad_mode="constant")
            ])
        else:
            raise ValueError(f"Unknown upsample_type : {upsample_type} ")

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None, io=0):
        i = 0
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if (((i+io) % 2) == 0) and self.use_fft:
                hidden_states = to_freq(hidden_states)

            hidden_states = resnet(hidden_states, temb)
            if attn is not None:
                hidden_states = attn(hidden_states.unsqueeze(-1)).squeeze(-1)

            if (((i+io) % 2) == 0) and self.use_fft:
                hidden_states = to_spatial(hidden_states)

            i += 1

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, temb=temb)

        return hidden_states, i