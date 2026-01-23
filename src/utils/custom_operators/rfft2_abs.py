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
import torch.library

# ==========================================
# 1. Helper: In-Place Scaling
# ==========================================
def _apply_half_spectral_weight_inplace(grad, last_dim_size):
    """
    Modifies grad IN-PLACE. Faster and uses less memory.
    """
    last_dim_freq = grad.shape[-1]
    original_width = last_dim_size[-1]
    
    # Multiply entire tensor by 0.5
    grad.mul_(0.5)
    
    # Restore DC (Index 0) -> 0.5 * 2.0 = 1.0
    # Use slicing to keep stride properties intact
    grad[..., 0].mul_(2.0)
    
    # Restore Nyquist (Last Index) if width is even
    if original_width % 2 == 0:
        grad[..., -1].mul_(2.0)
        
    return grad

# ==========================================
# 2. Backward Operator
# ==========================================

@torch.library.custom_op("custom::rfft2_abs_backward_impl", mutates_args=())
def rfft2_abs_backward_impl(grad_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # 1. Layout Safety
    # Only force contiguous if strictly necessary for the FFT kernel.
    # Using 'contiguous_format' is generally cheap if already contiguous.
    if not x.is_contiguous(memory_format=torch.contiguous_format) and \
       not x.is_contiguous(memory_format=torch.channels_last):
        x = x.contiguous()

    # 2. Recompute Phase
    # sgn(z) = z / |z|
    complex_freq = torch.fft.rfft2(x, norm="ortho")
    phase = torch.sgn(complex_freq)
    
    # 3. Project Gradient
    # Allocate the complex gradient. 
    # This is the unavoidable major allocation of the backward pass.
    grad_complex = grad_output.to(complex_freq.dtype) * phase
    
    # 4. In-Place Correction (OPTIMIZATION)
    _apply_half_spectral_weight_inplace(grad_complex, x.shape[-2:])
    
    # 5. Inverse FFT
    if not grad_complex.is_contiguous():
        grad_complex = grad_complex.contiguous()
        
    grad_input = torch.fft.irfft2(grad_complex, s=x.shape[-2:], norm="ortho")
    
    # 6. Layout Restore
    if x.is_contiguous(memory_format=torch.channels_last):
        grad_input = grad_input.to(memory_format=torch.channels_last)
        
    return grad_input

@rfft2_abs_backward_impl.register_fake
def _(grad_output, x):
    return torch.empty_like(x)

# ==========================================
# 3. Forward Operator
# ==========================================

@torch.library.custom_op("custom::rfft2_abs", mutates_args=())
def rfft2_abs(x: torch.Tensor) -> torch.Tensor:
    if not x.is_contiguous(memory_format=torch.contiguous_format) and \
       not x.is_contiguous(memory_format=torch.channels_last):
        x = x.contiguous()

    complex_freq = torch.fft.rfft2(x, norm="ortho")
    out = complex_freq.abs()
    
    if x.is_contiguous(memory_format=torch.channels_last):
        out = out.to(memory_format=torch.channels_last)
        
    return out

@rfft2_abs.register_fake
def _(x):
    out_shape = list(x.shape)
    out_shape[-1] = out_shape[-1] // 2 + 1
    return x.new_empty(out_shape)

# ==========================================
# 4. Autograd Registration
# ==========================================

def rfft2_abs_backward(ctx, grad_output):
    x = ctx.saved_tensors[0]
    return torch.ops.custom.rfft2_abs_backward_impl(grad_output, x)

def rfft2_abs_setup_context(ctx, inputs, output):
    # Clone is safer for compile, but if you are truly squeezed for memory
    # and know your graph is static, you *might* get away with removing .clone().
    # But for general stability, keep it.
    ctx.save_for_backward(inputs[0].clone())

torch.library.register_autograd(
    "custom::rfft2_abs",
    rfft2_abs_backward,
    setup_context=rfft2_abs_setup_context
)

# ==========================================
# 5. Verification
# ==========================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("--- Verifying Manual Math with 0.5 Correction ---")
    
    # 1. Even Width Test (Requires Nyquist handling)
    x_even = torch.randn(4, 3, 64, 64, device=device, requires_grad=True)
    
    x_ref = x_even.detach().clone().requires_grad_(True)
    grad_ref = torch.autograd.grad(
        torch.fft.rfft2(x_ref, norm="ortho").abs().sum(),
        x_ref
    )[0]
    
    @torch.compile(fullgraph=True)
    def train_step(inp):
        return torch.ops.custom.rfft2_abs(inp).sum()

    loss = train_step(x_even)
    loss.backward()
    
    diff = (x_even.grad - grad_ref).abs().max()
    print(f"Gradient Diff (Even): {diff:.2e}")

    # 2. Odd Width Test (No Nyquist)
    x_odd = torch.randn(4, 3, 65, 65, device=device, requires_grad=True)
    
    x_ref_odd = x_odd.detach().clone().requires_grad_(True)
    grad_ref_odd = torch.autograd.grad(
        torch.fft.rfft2(x_ref_odd, norm="ortho").abs().sum(),
        x_ref_odd
    )[0]
    
    loss_odd = train_step(x_odd)
    loss_odd.backward()
    
    diff_odd = (x_odd.grad - grad_ref_odd).abs().max()
    print(f"Gradient Diff (Odd):  {diff_odd:.2e}")
    print(x_odd.grad.std(), grad_ref_odd.std())
    
    if diff < 1e-4 and diff_odd < 1e-4:
        print("✅ SUCCESS: Correctly scaled gradients!")
    else:
        print("❌ FAILURE")