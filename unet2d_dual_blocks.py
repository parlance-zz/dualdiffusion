import torch
from torch import nn

from diffusers.utils import logging
from diffusers.models.attention_processor import Attention
from diffusers.models.resnet import ResnetBlock2D, Upsample2D, Downsample2D, FirDownsample2D, FirUpsample2D


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def AddPositionalCoding(x, num_heads):        
    
    import math

    max_seq_length = x.shape[1]
    d_model = x.shape[2]# // num_heads
    #d_model = num_heads

    pe = torch.zeros(max_seq_length, d_model, device=x.device)
    position = torch.arange(0, max_seq_length, dtype=torch.float32, device=x.device)
    position = ((position + 0.5) / max_seq_length).log()
    position -= position[0].item()
    position = (position / position[-1].item() * max_seq_length).unsqueeze(1)

    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=x.device) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)#.repeat(1, 1, num_heads)  # Add batch dimension
    #pe = pe.unsqueeze(0).repeat(1, 1, x.shape[2]//d_model)  # Add batch dimension
    #print("d_model: ", d_model, "max_seq_length: ", max_seq_length, "pe shape: ", pe.shape, "x shape: ", x.shape)

    return x# + pe

def shape_for_attention(hidden_states, attn_dim):

    if attn_dim == 3:
        hidden_states = hidden_states.permute(0, 3, 1, 2).contiguous()
    elif attn_dim == 2:
        hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous()        
    else:
        raise ValueError(f"attn_dim must be 2 or 3, got {attn_dim}")
    
    return hidden_states.view(-1, hidden_states.shape[2], 1, hidden_states.shape[3])

def unshape_for_attention(hidden_states, attn_dim, original_shape):

    if attn_dim == 3:
        hidden_states = hidden_states.view(original_shape[0], original_shape[3], original_shape[1], original_shape[2])
        hidden_states = hidden_states.permute(0, 2, 3, 1).contiguous()
    elif attn_dim == 2:
        hidden_states = hidden_states.view(original_shape[0], original_shape[2], original_shape[1], original_shape[3])
        hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous()
    else:
        raise ValueError(f"attn_dim must be 2 or 3, got {attn_dim}")
    
    return hidden_states

class SeparableAttnDownBlock2D(nn.Module):
    def __init__(
        self,
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
        downsample_padding=1,
        downsample_type="conv",
    ):
        super().__init__()
        resnets = []
        attentions = []
        self.downsample_type = downsample_type

        if attention_head_dim is None:
            logger.warn(
                f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {out_channels}."
            )
            attention_head_dim = out_channels

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
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
                )
            )
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
                )
            )

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
                    ResnetBlock2D(
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

        i = 0
        for resnet, attn in zip(self.resnets, self.attentions):

            hidden_states = resnet(hidden_states, temb)
            if i % 2 == 0: attn_dim = 3
            else: attn_dim = 2
            original_shape = hidden_states.shape
            hidden_states = shape_for_attention(hidden_states, attn_dim)
            hidden_states = attn(hidden_states)
            hidden_states = unshape_for_attention(hidden_states, attn_dim, original_shape)

            output_states = output_states + (hidden_states,)
            i += 1

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                if self.downsample_type == "resnet":
                    hidden_states = downsampler(hidden_states, temb=temb)
                else:
                    hidden_states = downsampler(hidden_states)

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
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attention_head_dim=1,
        output_scale_factor=1.0,
        upsample_type="conv",
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.upsample_type = upsample_type

        if attention_head_dim is None:
            logger.warn(
                f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {out_channels}."
            )
            attention_head_dim = out_channels

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
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
                )
            )
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
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if upsample_type == "conv":
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        elif upsample_type == "resnet":
            self.upsamplers = nn.ModuleList(
                [
                    ResnetBlock2D(
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

        i = 0
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb)
            if i % 2 == 0: attn_dim = 2
            else: attn_dim = 3
            original_shape = hidden_states.shape
            hidden_states = shape_for_attention(hidden_states, attn_dim)
            hidden_states = attn(hidden_states)
            hidden_states = unshape_for_attention(hidden_states, attn_dim, original_shape)

            i += 1

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                if self.upsample_type == "resnet":
                    hidden_states = upsampler(hidden_states, temb=temb)
                else:
                    hidden_states = upsampler(hidden_states)

        return hidden_states