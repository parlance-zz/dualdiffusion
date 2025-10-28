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
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


class SlidingWindowAttention(nn.Module):
    
    def __init__(
        self,
        window_size: int,
        causal: bool = False,
        head_dim: Optional[int] = None
    ):
        """
        Args:
            window_size: The size of the attention window (e.g., 64, 128, 256)
            causal: Whether to apply causal masking (autoregressive)
            head_dim: Optional head dimension for verification
        """
        super().__init__()
        self.window_size = window_size
        self.causal = causal
        self.head_dim = head_dim

        self.mask_fn = self._create_mask_fn()
        
    def _create_mask_fn(self) -> callable:
        """Create the mask function for sliding window + causal attention."""
        
        def sliding_window_causal_mask(b, h, q_idx, kv_idx):
            """
            Mask function for sliding window + causal attention.
            
            Args:
                b: batch index
                h: head index
                q_idx: query position index
                kv_idx: key/value position index
                
            Returns:
                Boolean tensor indicating which positions to attend to
            """
            # Causal constraint: can only attend to past tokens
            causal = q_idx >= kv_idx
            
            # Sliding window constraint: attention limited to window_size
            in_window = q_idx - kv_idx <= self.window_size
            
            return causal & in_window
        
        def sliding_window_mask(b, h, q_idx, kv_idx):
            """Mask function for sliding window attention (non-causal)."""
            # Attend to tokens within window on both sides
            distance = torch.abs(q_idx - kv_idx)
            return distance <= self.window_size
        
        mask_fn = sliding_window_causal_mask if self.causal else sliding_window_mask
        return mask_fn
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply sliding window attention.
        
        Args:
            query: [batch_size, num_heads, seq_len, head_dim]
            key: [batch_size, num_heads, seq_len, head_dim]
            value: [batch_size, num_heads, seq_len, head_dim]
            scale: Attention scale (typically 1/sqrt(head_dim))
            
        Returns:
            Attention output [batch_size, num_heads, seq_len, head_dim]
        """
        
        batch_size, num_heads, seq_len, head_dim = query.shape
        
        if self.head_dim is not None:
            assert head_dim == self.head_dim, \
                f"Expected head_dim={self.head_dim}, got {head_dim}"
        
        block_mask = create_block_mask(
            self.mask_fn,
            B=batch_size,
            H=num_heads,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=query.device,
        )
        
        output = flex_attention(
            query,
            key,
            value,
            block_mask=block_mask
        )
        
        return output
