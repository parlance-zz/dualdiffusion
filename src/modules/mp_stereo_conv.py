from typing import Union, Optional

import torch

from modules.mp_tools import normalize

# Stereo tensors are 4d (B, 2*C, H, W) with the stereo axis interleaved in the channel dim (channel = c*2 + z),
# so they can be viewed as 5d (B, C, 2, H, W) without a copy. MPStereoConv is equivalent to a conv3d with
# kernel size 2 in z and one-sided reflect padding (z index 2 -> 0), implemented with two 2d convs on the
# mid/side decomposition. The weight parameter has the standard conv3d shape (out_channels, in_channels // groups, 2, kh, kw).

class MPStereoConv(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel: tuple[int, int],
                 groups: int = 1, stride: Union[int, tuple[int, int]] = 1, disable_weight_norm: bool = False,
                 padding: Optional[tuple[int, int]] = None) -> None:
        
        super().__init__()

        assert len(kernel) == 2
        assert in_channels % groups == 0 and out_channels % groups == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.stride = stride
        self.disable_weight_norm = disable_weight_norm
        self.padding = padding or (kernel[0] // 2, kernel[1] // 2)

        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels // groups, 2, *kernel))
        #self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels // groups * 2, *kernel))
        self.weight.conv_groups = groups

    def forward(self, x: torch.Tensor, gain: Union[float, torch.Tensor] = 1.) -> torch.Tensor:
        
        w = self.weight.float()
        if self.training == True and self.disable_weight_norm == False:
            w = normalize(w) # traditional weight normalization
            
        w = w * (gain / w[0].numel()**0.5) # magnitude-preserving scaling

        B, _, H, W = x.shape
        x = x.view(B, -1, 2, H, W)
        x0, x1 = x[:, :, 0], x[:, :, 1]

        #w = w.view(self.out_channels, self.in_channels // self.groups, 2, w.shape[-2], w.shape[-1])
        w_mid  = ((w[:, :, 0] + w[:, :, 1]) * 0.5).to(x.dtype)
        w_side = ((w[:, :, 0] - w[:, :, 1]) * 0.5).to(x.dtype)

        y_mid  = torch.nn.functional.conv2d(x0 + x1, w_mid,  padding=self.padding, groups=self.groups, stride=self.stride)
        y_side = torch.nn.functional.conv2d(x0 - x1, w_side, padding=self.padding, groups=self.groups, stride=self.stride)

        return torch.stack((y_mid + y_side, y_mid - y_side), dim=2).reshape(B, -1, y_mid.shape[2], y_mid.shape[3])

    @torch.no_grad()
    def normalize_weights(self) -> None:
        if self.disable_weight_norm == False:
            self.weight.copy_(normalize(self.weight))
