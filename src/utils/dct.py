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

from typing import Literal

import torch


def dct(x: torch.Tensor, dim: int = -1, norm: Literal["ortho", "backward", "forward"] = "ortho") -> torch.Tensor:
    """
    Computes the Discrete Cosine Transform type II (DCT-II) along a specific dimension.
    Supports arbitrary tensor shapes.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): The dimension along which to compute the DCT.
        norm (str): Normalization mode. 'ortho' makes the transform orthonormal.

    Returns:
        torch.Tensor: DCT coefficients with the same shape as x.
    """
    # 1. Handle dimension and shapes
    ndim = x.ndim
    dim = dim % ndim
    N = x.shape[dim]
    
    # 2. Symmetric extension: [x, flip(x)] along target dimension
    # This creates a signal of length 2N with even symmetry
    x_sym = torch.cat([x, x.flip(dims=[dim])], dim=dim)
    
    # 3. FFT along target dimension (size 2N)
    Y: torch.Tensor = torch.fft.fft(x_sym, dim=dim)
    
    # 4. Truncate to first N coefficients
    # Use narrow to slice along an arbitrary dimension
    Y = Y.narrow(dim, 0, N)
    
    # 5. Apply complex chirp (twiddle factors)
    # Create index vector k of shape [1, ..., N, ..., 1] where N is at `dim`
    k_shape = [1] * ndim
    k_shape[dim] = N
    k = torch.arange(N, device=x.device, dtype=x.dtype).reshape(k_shape)
    
    # Formula: DCT_k = Re( e^{-i*pi*k/2N} * FFT_{2N}[k] )
    chirp = torch.exp(-1j * torch.pi * k / (2 * N))
    dct_coeffs = torch.real(Y * chirp)
    
    # 6. Scaling (remove factor of 2 from symmetric FFT sum)
    dct_coeffs = 0.5 * dct_coeffs
    
    if norm == 'ortho':
        # Apply Ortho Normalization
        scale = (2.0 / N)**0.5
        dct_coeffs = dct_coeffs * scale
        
        # Correction for DC component (k=0)
        # We need to slice dynamically to access index 0 at `dim`
        # Create a slice object like (:, :, 0:1, :)
        slices = [slice(None)] * ndim
        slices[dim] = slice(0, 1)
        dct_coeffs[tuple(slices)] /= 2.0**0.5
        
    return dct_coeffs

def idct(dct: torch.Tensor, dim: int = -1, norm: Literal["ortho", "backward", "forward"] = "ortho") -> torch.Tensor:
    """
    Computes the Inverse Discrete Cosine Transform (DCT-III) along a specific dimension.
    
    Args:
        coeffs (torch.Tensor): DCT coefficients.
        dim (int): The dimension along which to compute the IDCT.
        norm (str): Normalization mode. Must match the forward pass.
        
    Returns:
        torch.Tensor: Reconstructed tensor.
    """
    ndim = dct.ndim
    dim = dim % ndim
    N = dct.shape[dim]
    
    # 1. Undo Ortho Scaling if needed
    if norm == 'ortho':
        dct = dct.clone()
        # Reverse DC correction
        slices = [slice(None)] * ndim
        slices[dim] = slice(0, 1)
        dct[tuple(slices)] *= 2.0**0.5
        
        # Reverse global scaling
        scale = (2.0 / N)**0.5
        dct = dct / scale
        
    # 2. Reconstruct the half-spectrum Y_half
    k_shape = [1] * ndim
    k_shape[dim] = N
    k = torch.arange(N, device=dct.device, dtype=dct.dtype).reshape(k_shape)
    
    chirp_inv = torch.exp(1j * torch.pi * k / (2 * N))
    Y_half = 2 * dct * chirp_inv
    
    # 3. Construct full 2N spectrum for IFFT
    # We need to reconstruct the conjugate symmetric part.
    # The FFT of a real symmetric sequence: indices N+1 to 2N-1 are conjugates of N-1 to 1.
    
    # Slice out indices 1 to N-1 (exclude DC)
    # Slicing logic: narrow(dim, start, length)
    Y_ac = Y_half.narrow(dim, 1, N - 1)
    
    # Flip and Conjugate
    Y_rest = torch.conj(Y_ac.flip(dims=[dim]))
    
    # Nyquist frequency bin (at index N) is 0 for this specific symmetry
    # Create a zero slice of width 1 along target dim
    zeros_shape = list(dct.shape)
    zeros_shape[dim] = 1
    zeros = torch.zeros(zeros_shape, device=dct.device, dtype=Y_half.dtype)
    
    # Concatenate: [Y_half (0..N-1), Zero (N), Y_rest (N+1..2N-1)]
    Y_full = torch.cat([Y_half, zeros, Y_rest], dim=dim)
    
    # 4. Inverse FFT
    x_sym = torch.fft.ifft(Y_full, dim=dim)
    
    # 5. Take real part and truncate to N
    x_rec = torch.real(x_sym)
    x_rec = x_rec.narrow(dim, 0, N)
    
    return x_rec


if __name__ == "__main__":

    print("Running generalized reconstruction test...")
    torch.manual_seed(123)
    
    # Test case 1: Standard Image-like tensor, operate on Width (last dim)
    shape1 = (8, 4, 256, 1024)
    dim1 = -1
    x1 = torch.randn(shape1)
    
    coeffs1 = dct(x1, dim=dim1, norm='ortho')
    recon1 = idct(coeffs1, dim=dim1, norm='ortho')
    err1 = (x1 - recon1).abs().max().item()
    
    print(f"Test 1 (Shape {shape1}, dim={dim1}): Max Error = {err1:.2e}")
    assert err1 < 1e-5, "Test 1 Failed"

    # Test case 2: 3D tensor, operate on middle dimension
    shape2 = (10, 50, 20)
    dim2 = 1
    x2 = torch.randn(shape2)
    
    coeffs2 = dct(x2, dim=dim2, norm='ortho')
    recon2 = idct(coeffs2, dim=dim2, norm='ortho')
    err2 = (x2 - recon2).abs().max().item()
    
    print(f"Test 2 (Shape {shape2}, dim={dim2}): Max Error = {err2:.2e}")
    assert err2 < 1e-5, "Test 2 Failed"
    
    # Test case 3: Verify against standard transformation logic (energy compaction)
    # A flat constant signal should result in only DC component being non-zero
    shape3 = (1, 8)
    x3 = torch.ones(shape3)
    coeffs3 = dct(x3, dim=-1, norm='ortho')
    
    # DC component for ones vector length N should be sqrt(N) in ortho mode (or N scaled)
    # With ortho: sum(x^2) = N. sum(coeffs^2) = N. DC^2 = N -> DC = sqrt(N).
    expected_dc = 8**0.5
    actual_dc = coeffs3[0, 0].item()
    
    print(f"Test 3 (Energy Check): Expected DC ~{expected_dc:.4f}, Got {actual_dc:.4f}")
    assert abs(actual_dc - expected_dc) < 1e-4, "Test 3 Failed"

    x1_mse = (x1 - recon1).pow(2).mean()
    print(f"Test 1 MSE: {x1_mse.item():.2e}")
    x2_mse = (x2 - recon2).pow(2).mean()
    print(f"Test 2 MSE: {x2_mse.item():.2e}")

    rfft_coeffs = torch.fft.rfft(x1, dim=-2, norm="ortho")
    x_rfft_recon = torch.fft.irfft(rfft_coeffs, dim=-2, norm="ortho")

    rfft_error = (x1 - x_rfft_recon).abs().max()
    rfft_mse = (x1 - x_rfft_recon).pow(2).mean()
    print(f"RFFT Max absolute error: {rfft_error.item():.2e}")
    print(f"RFFT MSE: {rfft_mse.item():.2e}")

    print("All tests passed.")