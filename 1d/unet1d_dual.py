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
from diffusers.models.unet_1d import UNet1DOutput

from unet1d_dual_blocks import DualDownBlock1D, DualUpBlock1D, DualMidBlock1D

class UNet1DDualModel(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 2,
        out_channels: int = 2,
        time_embedding_type: str = "positional",
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        down_block_types: Tuple[str] = ("DualDownBlock1D", "DualDownBlock1D", "DualDownBlock1D", "DualDownBlock1D"),
        up_block_types: Tuple[str] = ("DualUpBlock1D", "DualUpBlock1D", "DualUpBlock1D", "DualUpBlock1D"),
        block_out_channels: Tuple[int] = (224, 448, 672, 896),
        layers_per_block: int = 2,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        attention_head_dim: Union[int, Tuple[int]] = 8,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        resnet_time_scale_shift: str = "default",
        add_attention: bool = True,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        conv_size: int = 3,
        downsample_type = "kernel",
        upsample_type = "kernel",
        use_fft: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.sample_size = sample_size
        self.conv_size = conv_size
        self.downsample_type = downsample_type
        self.upsample_type = upsample_type
        self.use_fft = use_fft

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

        if not isinstance(attention_head_dim, int) and len(attention_head_dim) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `attention_head_dim` as `down_block_types`. `attention_head_dim`: {attention_head_dim}. `down_block_types`: {down_block_types}."
            )
        
        # input
        self.conv_in = nn.Conv1d(in_channels, block_out_channels[0], kernel_size=conv_size, padding=conv_size//2)

        """
        self.conv_in = []
        for i in range(len(down_block_types)):
            self.conv_in.append(nn.Conv1d(in_channels, block_out_channels[0], kernel_size=conv_size, padding=conv_size//2))
        self.conv_in = nn.ModuleList(self.conv_in)
        """

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

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            if is_final_block is True:
                _downsample_type = None
            else:
                _downsample_type = downsample_type
                
            if (down_block_type == "DualDownBlock1D") or (down_block_type == "DualAttnDownBlock1D"):
                _add_attention = (down_block_type == "DualAttnDownBlock1D")
                down_block = DualDownBlock1D(
                    num_layers=layers_per_block,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    attention_head_dim=attention_head_dim[i],
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    downsample_type=_downsample_type,
                    add_attention=_add_attention,
                    conv_size=conv_size,
                    use_fft=use_fft,
                    dropout=dropout,
                )
            else:
                raise ValueError(f"Unrecognized down block type: {down_block_type}")

            self.down_blocks.append(down_block)

        # mid
        self.mid_block = DualMidBlock1D(
            #num_layers=layers_per_block,
            #num_layers=3,
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attention_head_dim=attention_head_dim[-1],
            resnet_groups=norm_num_groups,
            add_attention=add_attention,
            conv_size=conv_size,
            use_fft=use_fft,
            dropout=dropout,
        )

        reversed_attention_head_dim = list(reversed(attention_head_dim))

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            is_final_block = i == len(block_out_channels) - 1

            if (up_block_type == "DualUpBlock1D") or (up_block_type == "DualAttnUpBlock1D"):
                _add_attention = (up_block_type == "DualAttnUpBlock1D")
                if is_final_block is True:
                    _upsample_type = None
                else:
                    _upsample_type = upsample_type
                    
                up_block = DualUpBlock1D(
                    num_layers=layers_per_block + 1,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    temb_channels=time_embed_dim,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    attention_head_dim=reversed_attention_head_dim[i],
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    upsample_type=_upsample_type,
                    add_attention=_add_attention,
                    conv_size=conv_size,
                    use_fft=use_fft,
                    dropout=dropout,
                )
            else:
                raise ValueError(f"Unrecognized up block type: {up_block_type}")
            
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        num_groups_out = norm_num_groups if norm_num_groups is not None else min(block_out_channels[0] // 4, 32)
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=num_groups_out, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv1d(block_out_channels[0], out_channels, kernel_size=conv_size, padding=conv_size//2)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet1DOutput, Tuple]:

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

        #"""
        # 2. pre-process
        sample = self.conv_in(sample)
        io = 0

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            sample, res_samples, io = downsample_block(hidden_states=sample, temb=emb, io=io)
            down_block_res_samples += res_samples

        # 4. mid
        sample, io = self.mid_block(sample, emb, io=io)

        # 5. up
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            sample, io = upsample_block(sample, res_samples, emb, io=io)
        #"""

        """
        original_sample_fft = torch.fft.fft(sample, norm="ortho")
        max_freq = sample.shape[2]//2
        
        # 3. down
        down_block_res_samples = (self.conv_in[0](sample),)
        i = 0
        for downsample_block in self.down_blocks:
            
            if i == 0:
                sample = down_block_res_samples[0]
            else:
                sample = self.conv_in[i](sample)
            
            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

            if downsample_block != self.down_blocks[-1]:
                sample = original_sample_fft[:, :, max_freq//2:max_freq]
                sample = torch.cat((sample, torch.zeros_like(sample)), dim=2)
                sample = torch.fft.ifft(sample, norm="ortho").real
                max_freq //= 2

            i += 1
        # 4. mid

        sample = self.mid_block(sample, emb)

        # 5. up
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            sample = upsample_block(sample, res_samples, emb)
        """

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if self.config.time_embedding_type == "fourier":
            timesteps = timesteps.reshape((sample.shape[0], *([1] * len(sample.shape[1:]))))
            sample = sample / timesteps

        if not return_dict:
            return (sample,)

        return UNet1DOutput(sample=sample)

