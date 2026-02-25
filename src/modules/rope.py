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

import torch


def _rope_pair_rotate_partial(x: torch.Tensor, rope_tables: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """
    Rotate only the first rope_ch channels of x using RoPE pair-wise rotation.
    - x: [..., D]
    - cos/sin: broadcastable to [..., rope_ch/2]
    - rope_ch: even integer, 0 <= rope_ch <= D
    Returns x with first rope_ch channels rotated, the rest unchanged.
    """

    cos, sin = rope_tables
    rope_ch = cos.shape[-1] * 2
    
    x_rot  = x[..., :rope_ch]
    x_tail = x[..., rope_ch:]

    x_even = x_rot[..., 0::2]
    x_odd  = x_rot[..., 1::2]
    xr_even = x_even * cos - x_odd * sin
    xr_odd  = x_odd * cos + x_even * sin

    return torch.cat([xr_even, xr_odd, x_tail], dim=-1)

def _build_rope_width(W: int, rope_ch: int, base: float = 10000., scale: float = 1, device: torch.device = "cpu", dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build RoPE cos/sin tables for width axis only.
    Returns cos, sin with shape [W, rope_ch/2] in float32.
    """
    assert rope_ch % 2 == 0, "rope_ch must be even"
    if rope_ch == 0:
        # Dummy tensors (won't be used)
        return torch.tensor([], device=device, dtype=torch.float32), torch.tensor([], device=device, dtype=torch.float32)
    inv_freq = 1. / (base ** (torch.arange(0, rope_ch, 2, device=device, dtype=torch.float32) / rope_ch))
    cols = torch.arange(W, device=device, dtype=torch.float32) * scale
    ang = torch.einsum("w,d->wd", cols, inv_freq)  # [W, rope_ch/2]
    cos = torch.cos(ang).to(dtype=dtype)
    sin = torch.sin(ang).to(dtype=dtype)
    return cos, sin

def _rope_tables_for_seq(N: int, rope_ch: int, rope_base: float = 10000., scale: float = 1, device: torch.device = "cpu", dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build RoPE cos/sin for a 1D sequence of length N using mla.py utilities.
    Returns cos, sin shaped for broadcasting over [B, H, N, rope_ch/2].
    """
    assert rope_ch % 2 == 0 and rope_ch >= 0
    if rope_ch == 0:
        return (
            torch.tensor([], device=device, dtype=torch.float32).view(1, 1, N, 0),
            torch.tensor([], device=device, dtype=torch.float32).view(1, 1, N, 0),
        )
    cos_w, sin_w = _build_rope_width(N, rope_ch, rope_base, scale=scale, device=device, dtype=dtype)  # [N, rope_ch/2]
    cos = cos_w.view(1, 1, N, rope_ch // 2)
    sin = sin_w.view(1, 1, N, rope_ch // 2)
    return cos, sin


@torch.no_grad()
def run_test(N: int = 31, t0: int | None = None, rope_ch: int = 2, rope_base: float = 10000.0):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, H = 1, 1
    D = N
    if t0 is None:
        t0 = N // 4  # 25%

    assert rope_ch % 2 == 0 and 0 <= rope_ch <= D, "rope_ch must be even and <= D"

    cos, sin = _rope_tables_for_seq(N, rope_ch, rope_base, scale=1, device=device)

    q = torch.zeros(B, H, N, D, device=device, dtype=torch.float32)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    q[..., :rope_ch] = 1.
    k[..., :rope_ch] = 1.
    v[:, :, t0, :] = 1.

    from modules.mp_tools import normalize
    q = normalize(q, dim=3)
    k = normalize(k, dim=3)
    v = normalize(v, dim=3)

    q_rot  = _rope_pair_rotate_partial(q, (cos, sin))
    k_same = _rope_pair_rotate_partial(k, (cos, sin))

    y_same = torch.nn.functional.scaled_dot_product_attention(q_rot, k_same, v, dropout_p=0.0)  # [B,H,N,D]
    attn_same = y_same.mean(dim=(0,1,3)).cpu()  # [N]

    print(f"\nN={N}, t0={t0} (25% default), rope_ch={rope_ch}, rope_base={rope_base}")
    print(f"Same-sign (expected peak at {t0}):")
    for i, w in enumerate(attn_same.tolist()):
        print(f"  idx={i:3d}  dist_from_t0={abs(i - t0):3d}  w={w:.6f}")

    arg_same = int(attn_same.argmax().item())
    print(f"Same-sign peak ({attn_same.amax().item():.6f}) at index {arg_same}, expected {t0}")
    assert arg_same == t0, f"Same-sign peak expected at t0={t0}, got {arg_same}"


if __name__ == "__main__":
    from utils.dual_diffusion_utils import init_cuda
    init_cuda()
    run_test(N=688, t0=688 // 4, rope_ch=112, rope_base=10000.)