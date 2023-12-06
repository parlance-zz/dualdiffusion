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
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.utils import deprecate, logging
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.lora import LoRACompatibleLinear, LoRALinearLayer

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@maybe_allow_in_graph
class SeparableAttention(nn.Module):

    def __init__(
        self,
        query_dim: int,
        freq_embedding_dim: int,
        time_embedding_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias=False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        norm_num_groups: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        separate_attn_dim: int = 0,
    ):
        super().__init__()

        if heads <= 0:
            raise ValueError(f"heads must be a positive integer, got {heads}")
        
        self.separate_attn_dim = separate_attn_dim
        self.freq_embedding_dim = freq_embedding_dim
        self.time_embedding_dim = time_embedding_dim
        self.query_dim = query_dim + freq_embedding_dim + time_embedding_dim
        self.inner_query_dim = query_dim # + (freq_embedding_dim + time_embedding_dim) * (heads - 1)

        self.inner_dim = dim_head * heads
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.dropout = dropout
        self._from_deprecated_attn_block = False

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0
        self.heads = heads

        if norm_num_groups is not None:
            if norm_num_groups < 0: norm_num_groups = self.inner_dim // abs(norm_num_groups)
            self.group_norm_v = nn.GroupNorm(num_channels=self.inner_dim, num_groups=norm_num_groups, eps=eps, affine=True)
            self.group_norm_qk = nn.GroupNorm(num_channels=self.inner_dim, num_groups=norm_num_groups, eps=eps, affine=True)
            self.group_norm_embedding = nn.GroupNorm(num_channels=freq_embedding_dim+time_embedding_dim, num_groups=1, eps=eps, affine=True)
        else:
            self.group_norm_v = None
            self.group_norm_qk = None
            self.group_norm_embedding = None

        self.to_q = LoRACompatibleLinear(self.query_dim, self.inner_query_dim, bias=bias)
        self.to_k = LoRACompatibleLinear(self.query_dim, self.inner_query_dim, bias=bias)
        self.to_v = LoRACompatibleLinear(self.inner_dim, self.inner_dim, bias=bias)

        self.to_out = nn.ModuleList([])
        self.to_out.append(LoRACompatibleLinear(self.inner_dim, self.inner_dim, bias=out_bias))
        self.to_out.append(nn.Dropout(dropout))

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("SeparableAttention requires PyTorch2+ scaled_dot_product_attention")
        
        self.processor = SeparableAttnProcessor2_0()

    @staticmethod
    def shape_for_attention(hidden_states, attn_dim):

        if attn_dim == 0:
            return hidden_states
        elif attn_dim == 3:
            hidden_states = hidden_states.permute(0, 3, 1, 2)
            return hidden_states.reshape(hidden_states.shape[0]*hidden_states.shape[1], hidden_states.shape[2], hidden_states.shape[3], 1)
        elif attn_dim == 2:
            hidden_states = hidden_states.permute(0, 2, 1, 3)
            return hidden_states.reshape(hidden_states.shape[0]*hidden_states.shape[1], hidden_states.shape[2], 1, hidden_states.shape[3])
        else:
            raise ValueError(f"attn_dim must be 2, 3, or 0. got {attn_dim}")

    @staticmethod
    def unshape_for_attention(hidden_states, attn_dim, original_shape):

        if attn_dim == 0:
            return hidden_states
        elif attn_dim == 3:
            hidden_states = hidden_states.view(original_shape[0], original_shape[3], original_shape[1], original_shape[2])
            hidden_states = hidden_states.permute(0, 2, 3, 1)
        elif attn_dim == 2:
            hidden_states = hidden_states.view(original_shape[0], original_shape[2], original_shape[1], original_shape[3])
            hidden_states = hidden_states.permute(0, 2, 1, 3)
        else:
            raise ValueError(f"attn_dim must be 2, 3, or 0. got {attn_dim}")
        
        return hidden_states.contiguous()

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):

        hidden_states = self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

        return hidden_states

    def get_attention_scores(self, query, key, attention_mask=None):
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=self.scale,
        )
        del baddbmm_input

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        return attention_probs

    def prepare_attention_mask(self, attention_mask, target_length, batch_size=None, out_dim=3):
        if batch_size is None:
            deprecate(
                "batch_size=None",
                "0.22.0",
                (
                    "Not passing the `batch_size` parameter to `prepare_attention_mask` can lead to incorrect"
                    " attention mask preparation and is deprecated behavior. Please make sure to pass `batch_size` to"
                    " `prepare_attention_mask` when preparing the attention_mask."
                ),
            )
            batch_size = 1

        head_size = self.heads
        if attention_mask is None:
            return attention_mask

        current_length: int = attention_mask.shape[-1]
        if current_length != target_length:
            if attention_mask.device.type == "mps":
                # HACK: MPS: Does not support padding by greater than dimension of input tensor.
                # Instead, we can manually construct the padding tensor.
                padding_shape = (attention_mask.shape[0], attention_mask.shape[1], target_length)
                padding = torch.zeros(padding_shape, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, padding], dim=2)
            else:
                # TODO: for pipelines such as stable-diffusion, padding cross-attn mask:
                #       we want to instead pad by (0, remaining_length), where remaining_length is:
                #       remaining_length: int = target_length - current_length
                # TODO: re-enable tests/models/test_models_unet_2d_condition.py#test_model_xattn_padding
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)

        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * head_size:
                attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(head_size, dim=1)

        return attention_mask

class SeparableAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.cached_embeddings = None

    # v8
    @staticmethod
    @torch.no_grad()
    def get_embeddings(hidden_states_shape, freq_embedding_dim, time_embedding_dim, dtype=torch.float32, device="cuda"):

        if freq_embedding_dim % 2 != 0 or time_embedding_dim % 2 != 0:
            raise ValueError(f"freq_embedding_dim and time_embedding_dim must be divisible by 2. got freq_embedding_dim: {freq_embedding_dim} time_embedding_dim: {time_embedding_dim}")

        embeddings = None

        if freq_embedding_dim > 0:    
            num_freq_orders = freq_embedding_dim // 2
            ln_x = torch.arange(0.5, hidden_states_shape[2]*num_freq_orders+0.5, device=device).log2()
            ln_x *= torch.pi / ln_x[-1]
            ln_x = ln_x.view(hidden_states_shape[2], num_freq_orders).permute(1, 0).contiguous()
            ln_x *= torch.arange(0, num_freq_orders, device=device).view(-1, 1) + 0.5
            freq_embeddings = torch.view_as_real(torch.exp(1j * ln_x)).permute(0, 2, 1).reshape(1, freq_embedding_dim, hidden_states_shape[2], 1)
            freq_embeddings = freq_embeddings.repeat(hidden_states_shape[0], 1, 1, hidden_states_shape[3])

            embeddings = freq_embeddings

        if time_embedding_dim > 0:
            num_time_orders = time_embedding_dim // 2
            k = torch.arange(0, num_time_orders, device=device) + 0.5
            x = torch.arange(0, num_time_orders*hidden_states_shape[3], device=device)
            x = x.view(hidden_states_shape[3], num_time_orders).permute(1, 0).contiguous()
            time_embeddings = k.view(-1, 1) * x / (hidden_states_shape[3]*num_time_orders) * torch.pi
            time_embeddings = torch.view_as_real(torch.exp(1j * time_embeddings)).permute(0, 2, 1).reshape(1, time_embedding_dim, 1, hidden_states_shape[3])
            time_embeddings = time_embeddings.repeat(hidden_states_shape[0], 1, hidden_states_shape[2], 1)

            if embeddings is None:
                embeddings = time_embeddings
            else:
                embeddings = torch.cat((embeddings, time_embeddings), dim=1)

        return (embeddings * (2 ** 0.5)).type(dtype) # std = 1

    def __call__(
        self,
        attn: SeparableAttention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale: float = 1.0,
    ):
        residual = hidden_states
        hidden_states_original_shape = hidden_states.shape

        if attn.group_norm_v is not None:
            v_hidden_states = attn.group_norm_v(hidden_states)
            qk_hidden_states = attn.group_norm_qk(hidden_states)

        v_hidden_states = attn.shape_for_attention(v_hidden_states, attn.separate_attn_dim)
        qk_hidden_states = attn.shape_for_attention(qk_hidden_states, attn.separate_attn_dim)
    
        batch_size, v_channel, height, width = v_hidden_states.shape
        qk_channel = v_channel + attn.freq_embedding_dim + attn.time_embedding_dim

        if attn.freq_embedding_dim > 0 or attn.time_embedding_dim > 0:
            
            if self.cached_embeddings is None:
                self.cached_embeddings = self.get_embeddings(qk_hidden_states.shape, attn.freq_embedding_dim, attn.time_embedding_dim, qk_hidden_states.dtype, qk_hidden_states.device)
            else:
                if self.cached_embeddings.shape != hidden_states.shape:
                    self.cached_embeddings = self.get_embeddings(qk_hidden_states.shape, attn.freq_embedding_dim, attn.time_embedding_dim, qk_hidden_states.dtype, qk_hidden_states.device)
                else:
                    if self.cached_embeddings.dtype != qk_hidden_states.dtype or self.cached_embeddings.device != qk_hidden_states.device:
                        self.cached_embeddings = self.cached_embeddings.to(qk_hidden_states.dtype).to(qk_hidden_states.device)

            if attn.group_norm_embedding is not None:
                qk_embeddings = attn.group_norm_embedding(self.cached_embeddings)
            else:
                qk_embeddings = self.cached_embeddings
            qk_hidden_states = torch.cat((qk_hidden_states, qk_embeddings), dim=1)

        v_hidden_states = v_hidden_states.view(batch_size, v_channel, height * width).transpose(1, 2)
        qk_hidden_states = qk_hidden_states.view(batch_size, qk_channel, height * width).transpose(1, 2)

        query = attn.to_q(qk_hidden_states, scale=scale)
        key = attn.to_k(qk_hidden_states, scale=scale)
        value = attn.to_v(v_hidden_states, scale=scale)

        inner_dim_qk = key.shape[-1]
        inner_dim_v = value.shape[-1]

        head_dim_qk = inner_dim_qk // attn.heads
        head_dim_v = inner_dim_v // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim_qk).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim_qk).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim_v).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        v_hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        v_hidden_states = v_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim_v)
        v_hidden_states = v_hidden_states.to(query.dtype)

        v_hidden_states = attn.to_out[0](v_hidden_states, scale=scale) # linear proj
        v_hidden_states = attn.to_out[1](v_hidden_states) # dropout
        v_hidden_states = v_hidden_states.transpose(-1, -2).reshape(batch_size, v_channel, height, width)

        v_hidden_states = attn.unshape_for_attention(v_hidden_states, attn.separate_attn_dim, hidden_states_original_shape)

        if attn.residual_connection:
            v_hidden_states = v_hidden_states + residual
        return v_hidden_states / attn.rescale_output_factor