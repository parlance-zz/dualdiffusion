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


def _roll_each(t: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    idx = (torch.arange(t.shape[-1], device=t.device) + offsets[:, None]) % t.shape[-1]
    return torch.gather(t, dim=-1, index=idx[:, None, None, :].expand_as(t))

def lipschitz_loss(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:

    assert x.shape[-1] % y.shape[-1] == 0
    assert x.shape[-2] % y.shape[-2] == 0
    assert x.shape[-1] // y.shape[-1] == x.shape[-2] // y.shape[-2]

    num_iterations = y.shape[-1] - 1
    x = x.repeat_interleave(num_iterations, dim=0)
    y = y.repeat_interleave(num_iterations, dim=0)

    downsample_ratio = x.shape[-1] // y.shape[-1]
    bsz = x.shape[0]

    rnd_offsets = torch.arange(bsz, device=x.device)
    x2 = _roll_each(x, rnd_offsets * downsample_ratio)
    y2 = _roll_each(y, rnd_offsets)

    dx = torch.nn.functional.avg_pool2d((x - x2) ** 2, kernel_size=downsample_ratio).mean(dim=1, keepdim=True).detach()
    dy = ((y - y2) ** 2).mean(dim=1, keepdim=True)

    loss = ((dx + eps) / (dy + eps)).log().pow(2)
    return loss.mean().expand(bsz // num_iterations)


if __name__ == "__main__":
    torch.manual_seed(0)

    B, C, H, W = 8, 64, 16, 16
    x = torch.randn(B, C, H*8, W*8)
    y = torch.randn(B, C, H, W)

    loss = lipschitz_loss(x, y)
    print(f"Lipschitz loss: {loss.mean().item():.6f}")

    assert lipschitz_loss(x, x).mean().item() == 0.0, "Same input must yield zero loss"