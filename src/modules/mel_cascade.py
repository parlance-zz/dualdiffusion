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
import torch.nn as nn
import numpy as np


def hz_to_mel(f: float) -> float:
    return 2595.0 * np.log10(1.0 + f / 700.0)

def mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0**(m / 2595.0) - 1.0)

def get_frequency_grid(n_bins: int, alpha: float, sample_rate: float = 32000) -> np.ndarray:
    """
    Generates the center frequencies (in Hz) for a specific stage.
    alpha = 0.0 -> Perfectly Linear spacing
    alpha = 1.0 -> Perfectly Mel spacing
    """
    f_min = 0.0
    f_max = sample_rate / 2.0
    
    # 1. Linear Grid
    lin_grid = np.linspace(f_min, f_max, n_bins)
    
    # 2. Mel Grid
    m_min = hz_to_mel(f_min)
    m_max = hz_to_mel(f_max)
    m_grid = np.linspace(m_min, m_max, n_bins)
    mel_grid_hz = mel_to_hz(m_grid)
    
    # 3. Interpolate
    # This represents the "physical" location of the bins at this stage
    grid = (1.0 - alpha) * lin_grid + alpha * mel_grid_hz
    return grid

def build_transition_matrix(source_freqs: torch.Tensor, target_freqs: torch.Tensor) -> torch.Tensor:
    """
    Builds a transition matrix (n_in, n_out) mapping source_freqs to target_freqs.
    Each column j is a triangular filter centered at target_freqs[j],
    evaluated at the positions given by source_freqs.
    """
    n_in = len(source_freqs)
    n_out = len(target_freqs)
    
    weights = torch.zeros(n_in, n_out)
    
    # We need "boundaries" for the triangles in the target domain.
    # We estimate boundaries as the midpoints between target centers.
    # We assume the range covers the full bandwidth, so we pad edges.
    centers = target_freqs
    
    # Pad to handle the first and last triangle slopes
    # Extrapolate slightly for the edges
    delta_start = centers[1] - centers[0]
    delta_end = centers[-1] - centers[-2]
    padded_centers = np.concatenate([
        [centers[0] - delta_start], 
        centers, 
        [centers[-1] + delta_end]
    ])
    
    src_tensor = torch.tensor(source_freqs, dtype=torch.float32)
    
    for j in range(n_out):
        # Triangle definition for output bin j
        # Uses the padded indices to find left/right neighbors
        c_left   = padded_centers[j]     # conceptual previous bin
        c_center = padded_centers[j+1]   # current bin
        c_right  = padded_centers[j+2]   # conceptual next bin
        
        # 1. Upslope: (f_in - Left) / (Center - Left)
        mask_up = (src_tensor >= c_left) & (src_tensor <= c_center)
        weights[mask_up, j] = (src_tensor[mask_up] - c_left) / (c_center - c_left + 1e-8)
        
        # 2. Downslope: (Right - f_in) / (Right - Center)
        mask_down = (src_tensor > c_center) & (src_tensor <= c_right)
        weights[mask_down, j] = (c_right - src_tensor[mask_down]) / (c_right - c_center + 1e-8)

    # Row Normalization (Energy Conservation Approximation)
    # If we don't normalize, energy explodes or vanishes because of the bin count change.
    # We normalize such that the sum of weights contributing to an output bin is 1.
    # (Or you can normalize rows, depending on desired physical property).
    # Here, we normalize columns so each output bin represents a weighted average.
    col_sums = weights.sum(dim=0, keepdim=True)
    weights = weights / (col_sums + 1e-8)
    
    return weights

class ResampleStage(nn.Module):
    def __init__(self, n_in: int, n_out: int, alpha_in: float, alpha_out: float, sample_rate: float, reg_lambda: float, weight_decay: float = 0.03) -> None:
        super().__init__()
        
        # 1. Get the Hz coordinates for Input and Output
        freqs_in = get_frequency_grid(n_in, alpha_in, sample_rate)
        freqs_out = get_frequency_grid(n_out, alpha_out, sample_rate)
        
        # 2. Build Forward Matrix A: Maps freqs_in -> freqs_out
        # Shape: (n_in, n_out)
        A = build_transition_matrix(freqs_in, freqs_out)
        self.resample_forward: torch.Tensor = torch.nn.Parameter(A)
        setattr(self.resample_forward, "weight_decay", weight_decay)
        #self.register_buffer('resample_forward', A, persistent=False)
        
        # 3. Build Inverse Matrix W using Least Squares
        # Solve A * W = I (approximately)
        # Result W shape: (n_out, n_in)
        # Note: We want to recover Input from Output.
        # Inverse mapping: Output @ W -> Input
        """
        target = torch.eye(n_in)
        
        # Use lstsq driver 'gelss' or 'gels' (usually safe defaults)
        # A is (N, M). Target is (N, N). Solution is (M, N).
        try:
            W = torch.linalg.lstsq(A, target).solution
        except:
            # Fallback for older torch versions or specific CUDA backends
            W = torch.pinverse(A) @ target
        """
        
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        
        # Regularized inversion of singular values: s / (s^2 + lambda)
        # This dampens the explosion for very small singular values.
        S_inv = S / (S**2 + reg_lambda)
        
        # Reconstruct Pseudo-Inverse: V @ S_inv @ U.T
        W = Vh.mH @ torch.diag(S_inv) @ U.mH

        self.resample_inverse: torch.Tensor = torch.nn.Parameter(W)
        setattr(self.resample_inverse, "weight_decay", weight_decay)
        #self.register_buffer('resample_inverse', W, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., n_in)
        return torch.matmul(x, self.resample_forward)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., n_out)
        # Reconstructs: x @ W -> (..., n_in)
        return torch.matmul(x, self.resample_inverse)

class MelCascade(nn.Module):
    def __init__(self, sample_rate: float = 32000, num_bins: int = 256, num_stages: int = 3, reg_lambda: tuple[float] = (1e-4, 1e-3, 1e-2)) -> None:
        super().__init__()

        self.sample_rate = sample_rate
        self.num_bins = num_bins
        assert len(reg_lambda) == num_stages, "len(reg_lambda) != num_stages"
        
        stages = []

        for i in range(num_stages):
            
            alpha_in  = i / num_stages
            alpha_out = (i + 1) / num_stages

            n_bins_in  = num_bins // (2 ** i)
            n_bins_out = n_bins_in // 2

            stages.append(ResampleStage(n_bins_in, n_bins_out, alpha_in, alpha_out, sample_rate, reg_lambda[i]))
        
        self.stages = torch.nn.ModuleList(stages)

    def forward(self, x: torch.Tensor, stage: int = -1) -> torch.Tensor:
        # Input: (B, C, n_bins, W) -> Permute to (B, C, W, n_bins)
        x = x.permute(0, 1, 3, 2)

        if stage == -1: # Full cascade
            for stage in self.stages:
                x = stage(x)
        else:
            x = self.stages[stage](x)

        # Return final: (B, C, n_bins // 2, W)
        return x.permute(0, 1, 3, 2)

    def inverse_transform(self, x: torch.Tensor, stage: int = -1) -> torch.Tensor:
        # Input: (B, C, n_bins // 2, W)
        x = x.permute(0, 1, 3, 2)

        if stage == -1: # Full inverse cascade
            for stage in reversed(self.stages):
                x = stage.inverse_transform(x)
        else:
            x = self.stages[stage].inverse_transform(x)

        # Return final: (B, C, n_bins, W)
        return x.permute(0, 1, 3, 2)

# --- Validating the MSE ---
if __name__ == "__main__":

    from utils import config

    import os

    torch.manual_seed(42)
    
    # Use a smooth signal instead of white noise. 
    # White noise (torch.randn) has high entropy and is impossible to reconstruct 
    # well after downsampling. A smooth signal mimics real audio spectrograms better.
    b, c, h, w = 1, 16, 256, 100
    
    # Create a synthetic signal: smoothly varying sine wave in frequency dimension
    grid = torch.linspace(0, 10, h).view(1, 1, h, 1)
    time = torch.linspace(0, 5, w).view(1, 1, 1, w)
    x_in = torch.sin(grid * time).abs() # (1, 1, 256, 100)
    
    model = MelCascade()
    
    # Forward
    out = model(x_in)
    
    # Inverse
    recon = model.inverse_transform(out)
    
    print(f"Input Shape: {x_in.shape}")
    print(f"Output Shape: {out.shape}")
    print(f"Recon Shape: {recon.shape}")
    
    # MSE
    mse = nn.functional.mse_loss(recon, x_in)
    print(f"Reconstruction MSE on smooth signal: {mse.item():.5f}")
    
    # Check Random Noise MSE just for comparison
    noise = torch.randn(1, 16, 256, 100).abs()
    noise_recon = model.inverse_transform(model(noise))
    mse_noise = nn.functional.mse_loss(noise_recon, noise)
    print(f"Reconstruction MSE on random noise:  {mse_noise.item():.5f}")

    output_path = os.path.join(config.DEBUG_PATH, "mel_cascade")
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)

        model.stages[0].resample_forward.cpu().transpose(-1,-2).detach().numpy().tofile(os.path.join(output_path, "stage1_filters.raw"))
        model.stages[1].resample_forward.cpu().transpose(-1,-2).detach().numpy().tofile(os.path.join(output_path, "stage2_filters.raw"))
        model.stages[2].resample_forward.cpu().transpose(-1,-2).detach().numpy().tofile(os.path.join(output_path, "stage3_filters.raw"))

        model.stages[0].resample_inverse.cpu().transpose(-1,-2).detach().numpy().tofile(os.path.join(output_path, "stage1_inv_filters.raw"))
        model.stages[1].resample_inverse.cpu().transpose(-1,-2).detach().numpy().tofile(os.path.join(output_path, "stage2_inv_filters.raw"))
        model.stages[2].resample_inverse.cpu().transpose(-1,-2).detach().numpy().tofile(os.path.join(output_path, "stage3_inv_filters.raw"))