# MIT License
# 
# Copyright (c) 2024 Keller Jordan
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
# # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# MIT License
#
# Copyright (c) 2025 zichongli5
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

# Modifications under MIT License
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

from typing import Callable, Optional
from itertools import repeat

import torch


coeffs_list = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375), # subsequent coeffs equal this numerically
]

# safety factor for numerical stability (but exclude last polynomial)
coeffs_list = [(a / 1.01, b / 1.01**3, c / 1.01**5) for (a, b, c) in coeffs_list [:-1]] + [coeffs_list[-1]]

# "The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm" by Noah Amsel, David Persson, Christopher Musco, Robert M. Gower
def _polar_express(G: torch.Tensor, steps: int) -> torch.Tensor:
    assert G.ndim >= 2

    if torch.is_complex(G):
        X = G.to(dtype=torch.complex64)
    else:
        X = G.float()#bfloat16()
    if G.size(-2) > G.size(-1): X = X.mH # this reduces FLOPs

    X = X / (X.norm(dim=(-2,-1), keepdim=True) * 1.01 + 1e-7)
    hs = coeffs_list[:steps] + list(repeat(coeffs_list[-1], steps - len(coeffs_list)))

    for a, b, c in hs:
        A = X @ X.mH
        B = b * A + c * A @ A
        X = a * X + B @ X # X <- aX + bX ˆ3 + cX ˆ5
    
    if G.size(-2) > G.size(-1): X = X.mH
    return X

# "NorMuon: Making Muon more efficient and scalable" (https://arxiv.org/abs/2510.05491) by Zichong Li, Liming Liu, Chen Liang, Weizhu Chen, Tuo Zhao
def normuon_update(grad: torch.Tensor, momentum: torch.Tensor, second_momentum: Optional[torch.Tensor],
        beta: float = 0.95, beta2: float =0.95, ns_steps: int = 5, nesterov: bool = True, groups: int = 1) -> torch.Tensor:
    
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum

    is_muon_c_update = False
    if update.ndim >= 4:

        # "Muon-C: Operator-Aligned Muon for Convolutional Kernels" (https://arxiv.org/abs/2609.09676) by Jiaxin Qing, Lexin Li
        # additionally, groups for any size kernel (even 1x1) are packed into the batch dimension before orthogonalization, with the same reasoning

        if update.shape[-1] > 1 or update.shape[-2] > 1: 
            
            update = update.unflatten(dim=0, sizes=(groups, update.shape[0] // groups))
            H, W = update.shape[-2], update.shape[-1]
            update: torch.Tensor = torch.fft.rfft2(update, norm="ortho") # g, o, i, hf, wf
            Hf, Wf = update.shape[-2], update.shape[-1]
            update = update.permute(0, 3, 4, 1, 2).flatten(start_dim=0, end_dim=2) # g*hf*wf, o, i
            
            is_muon_c_update = True
        else:
            update = update.reshape(len(update), -1)
            # convert grouped conv params into a batch of smaller matrices for newton schulz iterations
            update = update.view(groups,-1, update.size(-1))
    
    update = _polar_express(update, steps=ns_steps)#.to(dtype=grad.dtype)

    if is_muon_c_update == True:
        update = update.unflatten(dim=0, sizes=(groups, Hf, Wf))  # g, hf, wf, o, i
        update = update.permute(0, 3, 4, 1, 2)  # g, o, i, hf, wf
        update = torch.fft.irfft2(update, s=(H,W), norm="ortho")  # g, o, i, h, w
        update = update.flatten(start_dim=2, end_dim=4)  # g, o, i*h*w

    if second_momentum is not None: #NorMuon added, from https://github.com/zichongli5/NorMuon
        vnorm = update.norm(dim=(-2,-1), keepdim=True)
        v_mean = torch.mean(update * update, dim=-1, keepdim=True)
        second_momentum.lerp_(v_mean, 1 - beta2)
        step_size = 1 / second_momentum.sqrt().add_(1e-20)
        update.mul_(step_size)
        vnorm_new = update.norm(dim=(-2,-1), keepdim=True)
        update.mul_(vnorm / (vnorm_new.add_(1e-20))) # This scaling keep the update norm the same as pre-normalization

    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5

    return update

def adam_update(grad: torch.Tensor, buf1: torch.Tensor, buf2: torch.Tensor,
        step: int, betas: tuple[float, float], eps: float) -> torch.Tensor:
    
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)

class SingleDeviceNorMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Non-distributed variant of MuonWithAuxAdam.
    """
    def __init__(self, param_groups: list[dict]) -> None:

        for group in param_groups:
            assert "use_muon" in group

            # set group defaults
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["beta2"] = group.get("beta2", 0.95)
                group["normuon"] = group.get("normuon", True)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon", "beta2", "normuon"])
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])

        super().__init__(param_groups, dict())

    @torch.no_grad()
    def zero_momentum(self) -> None:

        p: torch.nn.Parameter

        for group in self.param_groups:
            
            if group["use_muon"]:
                for p in group["params"]:

                    if p.grad is not None:
                        p.grad.zero_()

                    state = self.state[p]
                    if len(state) > 0:
                        state["momentum_buffer"].zero_()
                        if group["normuon"]:
                            state["second_momentum_buffer"].zero_()

            else:
                for p in group["params"]:

                    if p.grad is not None:
                        p.grad.zero_()

                    state = self.state[p]
                    if len(state) > 0:
                        state["exp_avg"].zero_()
                        state["exp_avg_sq"].zero_()

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        p: torch.nn.Parameter
        update: torch.Tensor

        for group in self.param_groups:
            
            if group["use_muon"]:
                for p in group["params"]:

                    if p.grad is None:
                        p.grad = torch.zeros_like(p)  # workaround for ddp nuisance unused param errors
                    
                    state = self.state[p]
                    groups = getattr(p, "conv_groups", 1)

                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)

                        if group["normuon"]:
                            # shape change needed for grouped conv params
                            second_momentum_shape = (groups, p.shape[0] // groups, 1)
                            state["second_momentum_buffer"] = torch.zeros(
                                size=second_momentum_shape, device=p.device, dtype=p.dtype)
                        else:
                            state["second_momentum_buffer"] = None
                        
                    update = normuon_update(p.grad, state["momentum_buffer"], state["second_momentum_buffer"],
                        beta=group["momentum"], beta2=group["beta2"], groups=groups)

                    weight_decay = getattr(p, "weight_decay", group["weight_decay"])
                    if weight_decay > 0:
                        p.mul_(max(0, 1 - group["lr"] * weight_decay))
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:

                    if p.grad is None:
                        p.grad = torch.zeros_like(p)  # workaround for ddp nuisance unused param errors

                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1

                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])

                    weight_decay = getattr(p, "weight_decay", group["weight_decay"])
                    if weight_decay > 0:
                        p.mul_(max(0, 1 - group["lr"] * weight_decay))
                    p.add_(update, alpha=-group["lr"])

        return loss
