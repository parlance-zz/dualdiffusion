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
import torch.nn.functional as F


def lipschitz_loss(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Compute the Lipschitz Loss between two batches of feature maps.

    The loss measures the discrepancy between pairwise cosine-similarity
    structures of two representation spaces.  It encourages the mapping
    between the spaces to be Lipschitz-continuous in the cosine-similarity
    metric.

    Steps
    -----
    1. Flatten each (B, C, H, W) tensor to (B, D) where D = C*H*W.
    2. L2-normalise every sample along the feature dimension.
    3. Compute pairwise cosine-similarity matrices (B, B) for both
       tensors using ``torch.einsum`` for performance.
    4. Return the mean absolute difference between the two similarity
       matrices.

    Parameters
    ----------
    x : torch.Tensor
        First batch of feature maps, shape ``(B, C, H, W)``.
    y : torch.Tensor
        Second batch of feature maps, shape ``(B, C, H, W)``.

    Returns
    -------
    torch.Tensor
        Scalar loss value.

    Raises
    ------
    ValueError
        If the inputs are not 4-D or their batch sizes differ.
    """
    if x.ndim != 4 or y.ndim != 4:
        raise ValueError(
            f"Expected 4-D tensors (B, C, H, W), got x: {x.ndim}-D, y: {y.ndim}-D"
        )
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"Batch sizes must match, got x: {x.shape[0]}, y: {y.shape[0]}"
        )

    # ---------- 1. Flatten to (B, D) ----------
    b = x.shape[0]
    x_flat = x.reshape(b, -1)          # (B, D_x)
    y_flat = y.reshape(b, -1)          # (B, D_y)

    # ---------- 2. L2-normalise along feature dim ----------
    x_norm = F.normalize(x_flat, dim=1)   # (B, D_x)
    y_norm = F.normalize(y_flat, dim=1)   # (B, D_y)

    # ---------- 3. Pairwise cosine similarity via einsum ----------
    # After L2 normalisation, cosine similarity == dot product.
    cos_x = torch.einsum("id, jd -> ij", x_norm, x_norm)   # (B, B)
    cos_y = torch.einsum("id, jd -> ij", y_norm, y_norm)   # (B, B)

    # ---------- 4. Loss = mean | cos_x - cos_y | ----------
    loss = (cos_x - cos_y).abs().mean()

    return loss


if __name__ == "__main__":
    torch.manual_seed(0)

    B, C, H, W = 8, 64, 16, 16
    x = torch.randn(B, C, H, W)
    y = torch.randn(B, C, H, W)

    loss = lipschitz_loss(x, y)
    print(f"Lipschitz loss: {loss.item():.6f}")   # e.g. ≈ 0.04–0.08 for random inputs

    # Identical inputs → zero loss
    assert lipschitz_loss(x, x).item() == 0.0, "Same input must yield zero loss"
    print("✓ All checks passed.")