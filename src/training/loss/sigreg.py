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


# initial implementation taken from https://github.com/kreasof-ai/sigreg (no license specified)
def sigreg_strong_loss(x: torch.Tensor, sketch_dim: int = 64,
        t_min: float = -5, t_max: float = 5, num_t: int = 17, eps: float = 1e-6) -> torch.Tensor:
    """
    Strong-SIGReg (LeJEPA): Forces ECF(x) ~ ECF(Gaussian).
    Matches all moments using random 1D projections.
    """

    # use frequency axis slices as channels for 2d image-like tensors
    B = x.shape[0]

    x = x.permute(0, 3, 1, 2)
    x = x.reshape(x.shape[0] * x.shape[1], x.shape[2] * x.shape[3])

    N, C = x.size()

    if sketch_dim <= 0:
        sketch_dim = C

    if C > sketch_dim:
        A = torch.randn(C, sketch_dim, device=x.device)
    else:
        A = torch.randn(C, C, device=x.device)
        
    A = A / (A.norm(p=2, dim=0, keepdim=True) + eps)

    t = torch.linspace(t_min, t_max, num_t, device=x.device)
    exp_f = torch.exp(-0.5 * t**2)

    proj: torch.Tensor = x @ A
    args = proj.unsqueeze(2) * t.view(1, 1,-1)
    
    ecf = torch.exp(1j * args)
    ecf = ecf.mean(dim=0)
    
    diff_sq = (ecf - exp_f.unsqueeze(0)).abs().square()
    err = diff_sq * exp_f.unsqueeze(0)
    
    loss = torch.trapz(err, t, dim=1)# * N
    return loss.mean().expand(B) # return per-sample loss for compatibility

def sigreg2(x: torch.Tensor, y: torch.Tensor, sketch_dim: int = 64,
        t_min: float = -5, t_max: float = 5, num_t: int = 17, eps: float = 1e-6) -> torch.Tensor:

    assert x.shape == y.shape
    # use frequency axis slices as channels for 2d image-like tensors
    B = x.shape[0]

    x = x.permute(0, 3, 1, 2)
    x = x.reshape(x.shape[0] * x.shape[1], x.shape[2] * x.shape[3])
    y = y.permute(0, 3, 1, 2)
    y = y.reshape(y.shape[0] * y.shape[1], y.shape[2] * y.shape[3])

    N, C = x.size()

    if sketch_dim <= 0:
        sketch_dim = C

    if C > sketch_dim:
        A = torch.randn(C, sketch_dim, device=x.device)
    else:
        A = torch.randn(C, C, device=x.device)
        
    A = A / (A.norm(p=2, dim=0, keepdim=True) + eps)

    t = torch.linspace(t_min, t_max, num_t, device=x.device)
    exp_f = torch.exp(-0.5 * t**2)

    proj_x: torch.Tensor = x @ A
    args_x = proj_x.unsqueeze(2) * t.view(1, 1,-1)
    ecf_x = torch.exp(1j * args_x)
    ecf_x = ecf_x.mean(dim=0)
    
    proj_y: torch.Tensor = y @ A
    args_y = proj_y.unsqueeze(2) * t.view(1, 1,-1)
    ecf_y = torch.exp(1j * args_y)
    ecf_y = ecf_y.mean(dim=0)
    
    diff_sq = (ecf_x - ecf_y).abs().square()
    err = diff_sq# * exp_f.unsqueeze(0)
    
    loss = torch.trapz(err, t, dim=1)# * N
    return loss.mean().expand(B) # return per-sample loss for compatibility


if __name__ == "__main__":
    
    x = torch.randn(4, 8, 256, 384)
    print(sigreg_strong_loss(x))

    y = torch.randn(4, 8, 256, 384)
    print(sigreg2(x, y))