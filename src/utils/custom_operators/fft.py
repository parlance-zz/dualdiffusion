import torch
import torch.library

# ==========================================
# 1. Define the Custom Operators
# ==========================================

# We define a library namespace 'custom'
# The input is a real tensor where the last dimension contains 
# alternating real/imaginary parts (size = 2 * N).
@torch.library.custom_op("custom::fft_1d_real", mutates_args=())
def fft_1d_real(x: torch.Tensor) -> torch.Tensor:
    """
    Performs 1D FFT on a real tensor simulating complex numbers.
    Input shape: (..., 2 * N)
    Output shape: (..., 2 * N)
    """

    needs_channels_last = (x.ndim == 4 and x.stride(1) == 1)    
    x = x.contiguous()

    x_complex = torch.view_as_complex(x.view(*x.shape[:-1], -1, 2))
    
    # 3. Perform standard PyTorch FFT
    out_complex = torch.fft.fft(x_complex)
    
    # 4. Convert back to Real (..., N, 2) and flatten to (..., 2 * N)
    out_real = torch.view_as_real(out_complex).flatten(-2)
    
    # 4. Layout Restoration
    # If the input was Channels Last, Inductor expects the output to be too.
    # We explicitly convert the result back to satisfy that contract.
    if needs_channels_last:
        out_real = out_real.to(memory_format=torch.channels_last)

    return out_real

@torch.library.custom_op("custom::ifft_1d_real", mutates_args=())
def ifft_1d_real(x: torch.Tensor) -> torch.Tensor:
    """
    Performs 1D Inverse FFT on a real tensor simulating complex numbers.
    """

    needs_channels_last = (x.ndim == 4 and x.stride(1) == 1)
    x = x.contiguous()

    x_complex = torch.view_as_complex(x.view(*x.shape[:-1], -1, 2))
    out_complex = torch.fft.ifft(x_complex)
    
    # 4. Convert back to Real (..., N, 2) and flatten to (..., 2 * N)
    out_real = torch.view_as_real(out_complex).flatten(-2)
    
    # 4. Layout Restoration
    # If the input was Channels Last, Inductor expects the output to be too.
    # We explicitly convert the result back to satisfy that contract.
    if needs_channels_last:
        out_real = out_real.to(memory_format=torch.channels_last)

    return out_real

# ==========================================
# 2. The Missing Piece: Abstract/Fake Kernels
# ==========================================
# torch.compile needs a "Fake Tensor" implementation to calculate 
# output shapes and types without actually running the data.
# Since FFT preserves shape (N complex -> N complex), 
# the real-wrapped version preserves shape (2N float -> 2N float).

@fft_1d_real.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)

@ifft_1d_real.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)

# ==========================================
# 3. Define and Register Backward Pass (Autograd)
# ==========================================

def fft_backward(ctx, grad_output):
    # Math: The gradient of FFT(x) is IFFT(grad) * N
    # N = number of complex elements (half the last dim)
    N = grad_output.shape[-1] // 2
    
    # CRITICAL: We call our OWN custom op here, not the torch primitive.
    # This ensures the backward pass is also compile-friendly!
    grad_input = torch.ops.custom.ifft_1d_real(grad_output)
    return grad_input * N

def ifft_backward(ctx, grad_output):
    # Math: The gradient of IFFT(x) is FFT(grad) / N
    N = grad_output.shape[-1] // 2
    
    grad_input = torch.ops.custom.fft_1d_real(grad_output)
    return grad_input / N

# Register the backward formulas
# setup_context is empty because we don't need to save any tensors (FFT is linear)
torch.library.register_autograd(
    "custom::fft_1d_real",
    fft_backward,
    setup_context=lambda ctx, inputs, output: None
)

torch.library.register_autograd(
    "custom::ifft_1d_real",
    ifft_backward,
    setup_context=lambda ctx, inputs, output: None
)

# ==========================================
# 4. Verification and Benchmark
# ==========================================

def dual_modulate(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    
    x_in = x.float()
    y_in = y.float()

    x_freq = torch.ops.custom.fft_1d_real(x_in) * y_in
    y_freq = torch.ops.custom.ifft_1d_real(y_in) * x_in

    x_out = torch.ops.custom.ifft_1d_real(x_freq)
    y_out = torch.ops.custom.fft_1d_real(y_freq)

    return x_out.to(dtype=x.dtype), y_out.to(dtype=y.dtype)

def dual_modulate_2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    
    x_out, y_out = dual_modulate(x.transpose(1, -1), y.transpose(1, -1))
    return x_out.transpose(1, -1), y_out.transpose(1, -1)

if __name__ == "__main__":

    N = 1024 * 1024
    # Create random "complex" data stored as alternating floats
    x = torch.randn(1, N * 2, device="cuda", dtype=torch.float32)
    
    # Define a function to compile
    def my_spectral_op(x: torch.Tensor) -> torch.Tensor:
        # 1. FFT
        freq = torch.ops.custom.fft_1d_real(x)
        # 2. Some operation in frequency domain (e.g., multiply by 2)
        freq = freq * 2
        # 3. IFFT
        out = torch.ops.custom.ifft_1d_real(freq)
        return out

    print("--- Compiling ---")
    # This usually fails with standard torch.fft.fft due to complex support
    compiled_op = torch.compile(my_spectral_op, fullgraph=True)

    print("--- Warmup Run ---")
    with torch.no_grad():
        res_eager = my_spectral_op(x)
        res_compiled = compiled_op(x)

    print("--- Checking Accuracy ---")
    # Floating point differences are expected, but should be small
    diff = (res_eager - res_compiled).abs().max()
    print(f"Max difference between Eager and Compile: {diff:.2e}")
    print(f"Input shape: {x.shape}, Output shape: {res_compiled.shape}")
    print(f"Input dtype: {x.dtype}, Output dtype: {res_compiled.dtype}")
    print(f"Input std: {x.std():.4f}, Output std: {res_compiled.std():.4f}")
    print(f"Input mean: {x.mean():.4f}, Output mean: {res_compiled.mean():.4f}")
    
    if diff < 1e-5:
        print("✅ SUCCESS: Compilation produced correct results.")
    else:
        print("❌ FAILURE: Results diverge.")