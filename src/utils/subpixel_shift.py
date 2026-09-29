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


def subpixel_shift_cubic(tensor: torch.Tensor, shift_fraction: torch.Tensor) -> torch.Tensor:
    """
    Shifts a tensor of shape (b, c, h, w) in the width dimension by a 
    fractional amount using cubic interpolation.

    Args:
        tensor: Input tensor of shape (b, c, h, w)
        shift_fraction: Float between 0 and 1 indicating the shift amount 
                        in pixels (positive shifts right).

    Returns:
        Shifted tensor of shape (b, c, h, w)
    """
    #if not (0 <= shift_fraction <= 1):
    #    raise ValueError("shift_fraction must be between 0 and 1")

    b, c, h, w = tensor.shape
    device = tensor.device

    # Create a base grid of coordinates (h, w, 2)
    # Range is [-1, 1] for grid_sample
    # y coordinates run from -1 to 1
    # x coordinates run from -1 to 1
    
    # Generate standard meshgrid in normalized coordinates [-1, 1]
    y_grid = torch.linspace(-1, 1, h, device=device).view(h, 1).repeat(1, w)
    x_grid = torch.linspace(-1, 1, w, device=device).view(1, w).repeat(h, 1)
    
    # Calculate the shift in normalized coordinates
    # The width of the image in normalized coords is 2 (from -1 to 1)
    # So 1 pixel represents 2/w in normalized space.
    # We subtract because grid_sample uses sampling coordinates: 
    # if we want to shift the image RIGHT, we must sample from the LEFT.
    normalized_shift = (shift_fraction * 2) / w
    x_grid = x_grid - normalized_shift
    
    # Stack to create (b, h, w, 2) grid
    grid = torch.stack((x_grid, y_grid), dim=2)
    grid = grid.unsqueeze(0).expand(b, -1, -1, -1)

    # Use grid_sample with bicubic mode
    # align_corners=True preserves the exact corner pixel values mapping to -1 and 1
    shifted_tensor = F.grid_sample(
        tensor, 
        grid, 
        mode='bicubic', 
        padding_mode='reflection', # Or 'border'/'reflection' depending on needs
        align_corners=False
    )
    
    return shifted_tensor