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

# The AMUSE y/x/z averaging scheme below is adapted from the AMUSE optimizer
# implementation at https://github.com/kjeiun/amuse (src/optim/AMUSE.py).
# Check that repository for its license terms before redistributing.

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

def _polar_express(G: torch.Tensor, steps: int) -> torch.Tensor:
    assert G.ndim >= 2

    X = G.float()#bfloat16()
    if G.size(-2) > G.size(-1): X = X.mT # this reduces FLOPs

    X = X / (X.norm(dim=(-2,-1), keepdim=True) * 1.01 + 1e-7)
    hs = coeffs_list[:steps] + list(repeat(coeffs_list[-1], steps - len(coeffs_list)))

    for a, b, c in hs:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X # X <- aX + bX ˆ3 + cX ˆ5
    
    if G.size(-2) > G.size(-1): X = X.mT
    return X

def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Batched Newton-Schulz iteration to compute an approximate 'zeroth power' or orthogonalization
    of G. Each batch element of G (shape: [out_channels, in_channels]) is treated independently.

    Args:
        G: Tensor of shape (bsz, out_channels, in_channels)
        steps: Number of Newton–Schulz iterations.

    Returns:
        Tensor of shape (bsz, out_channels, in_channels)
    """
    assert G.ndim == 3, "Expected G of shape (bsz, out_channels, in_channels)"
    
    a, b, c = (3.4445, -4.7750, 2.0315)

    X = G.float()#to(torch.bfloat16)
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.transpose(-2, -1)

    # Normalize each matrix so spectral norm ≤ 1 (approximate)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    # Perform batched Newton–Schulz iterations
    for _ in range(steps):
        A = X @ X.transpose(-2, -1)                  # (bsz, n, n)
        B = b * A + c * (A @ A)                      # quintic term
        X = a * X + B @ X                            # update step

    if transposed:
        X = X.transpose(-2, -1)

    return X

def normuon_update(grad: torch.Tensor, momentum: torch.Tensor, second_momentum: Optional[torch.Tensor],
        beta: float = 0.95, beta2: float =0.95, ns_steps: int = 5, nesterov: bool = True, groups: int = 1) -> torch.Tensor:
    
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim >= 4: # reshape instead of view is needed for conv params when channels last is enabled
        update = update.reshape(len(update), -1)

    # convert grouped conv params into a batch of smaller matrices for newton schulz iterations
    update = update.view(groups,-1, update.size(-1))
    
    #update = _zeropower_via_newtonschulz5(update, steps=ns_steps).to(dtype=grad.dtype)
    update = _polar_express(update, steps=ns_steps).to(dtype=grad.dtype)

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

class SingleDeviceAMUSENorMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Non-distributed NorMuon + aux Adam, wrapped in the AMUSE y/x/z averaging scheme.

    The per-parameter update rules are unchanged from SingleDeviceNorMuonWithAuxAdam
    (NorMuon w/ polar express for use_muon groups, adam_update for the rest).
    AMUSE only changes *where* the update is applied and how the weights are averaged:

    State convention (from AMUSE):
    - p stores y while training.
    - state["z"] stores the anchor z (the "raw" iterate that receives the updates).
    - eval() converts y -> x (the averaged weights to use for inference / validation).
    - train() converts x -> y. Call train() before resuming training after eval().
    - The optimizer starts in train mode (x == y == z at initialization).

    AMUSE hyperparameters (constructor):
    - warmup_steps: linear lr warmup length. Must be > 0. (This is a schedule hyperparameter.)
    - beta1: initial y/x interpolation, constant during warmup.
    - rho: how quickly beta1 approaches 1 after warmup.
    - r: polynomial power for the z/x averaging weights.
    - weight_lr_power: power of lr in the averaging weights.
    - weight_decay_at_y: optional extra decay applied while p is still y.

    Group hyperparameters are the same as SingleDeviceNorMuonWithAuxAdam. "lr" is the
    base lr, the effective lr is scaled by the warmup factor and exposed as group["lr"].
    Weight decay (including per-param "weight_decay" attributes) is applied to z.
    """
    def __init__(self, param_groups: list[dict], *, warmup_steps: int, beta1: float = 0.9,
                 rho: float = 1.0, r: float = 0.0, weight_lr_power: float = 2.0,
                 weight_decay_at_y: float = 0.0) -> None:

        if warmup_steps <= 0:
            raise ValueError("AMUSE requires warmup_steps > 0.")

        self.warmup_steps = int(warmup_steps)
        self.beta1_init = float(beta1)
        self.rho = float(rho)
        self.r = r
        self.weight_lr_power = weight_lr_power
        self.weight_decay_at_y = weight_decay_at_y
        self.train_mode = True # x == y == z at init, so this is equivalent to train() having been called

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

            # AMUSE per-group state
            group["base_lr"] = group["lr"]
            group["k"] = 0
            group["weight_sum"] = 0.0
            group["beta1"] = self.beta1_init

        super().__init__(param_groups, dict())

    def _compute_beta1(self, group: dict, t: int, ckp1: float) -> float:
        if t <= self.warmup_steps:
            if t == self.warmup_steps:
                group["c_warmup"] = ckp1
            return self.beta1_init

        c_warmup = group.get("c_warmup", 1.0 / self.warmup_steps)
        S_t = (ckp1 * (1.0 - c_warmup)) / (c_warmup * (1.0 - ckp1))
        return 1.0 - (S_t ** self.rho) * (1.0 - self.beta1_init)

    def _get_z(self, p: torch.Tensor) -> torch.Tensor:
        state = self.state[p]
        z = state.get("z")
        if z is None:
            z = state["z"] = torch.clone(p, memory_format=torch.preserve_format)
        return z

    @torch.no_grad()
    def eval(self) -> None:
        """y -> x, call before validation / inference / saving averaged weights"""
        if self.train_mode:
            for group in self.param_groups:
                beta1 = group["beta1"]
                for p in group["params"]:
                    z = self.state[p].get("z")
                    if z is not None:
                        p.lerp_(end=z, weight=1.0 - 1.0 / beta1)
        self.train_mode = False

    @torch.no_grad()
    def train(self) -> None:
        """x -> y, call before resuming training after eval()"""
        if not self.train_mode:
            for group in self.param_groups:
                beta1 = group["beta1"]
                for p in group["params"]:
                    z = self.state[p].get("z")
                    if z is not None:
                        p.lerp_(end=z, weight=1.0 - beta1)
        self.train_mode = True

    @torch.no_grad()
    def zero_momentum(self) -> None:

        p: torch.nn.Parameter

        for group in self.param_groups:
            
            if group["use_muon"]:
                for p in group["params"]:

                    if p.grad is not None:
                        p.grad.zero_()

                    state = self.state[p]
                    if "momentum_buffer" in state:
                        state["momentum_buffer"].zero_()
                        if group["normuon"]:
                            state["second_momentum_buffer"].zero_()

            else:
                for p in group["params"]:

                    if p.grad is not None:
                        p.grad.zero_()

                    state = self.state[p]
                    if "exp_avg" in state:
                        state["exp_avg"].zero_()
                        state["exp_avg_sq"].zero_()

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None):

        if not self.train_mode:
            raise RuntimeError("Optimizer was not in train mode when step was called. "
                               "Please insert .train() and .eval() calls on the optimizer.")

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        p: torch.nn.Parameter
        update: torch.Tensor

        for group in self.param_groups:

            # AMUSE schedule: warmup lr and averaging weights
            t = group["k"] + 1
            lr = group["base_lr"] * min(1.0, t / self.warmup_steps)
            group["lr"] = lr

            weight = (t ** self.r) * (lr ** self.weight_lr_power)
            group["weight_sum"] += weight
            ckp1 = weight / group["weight_sum"] if group["weight_sum"] > 0 else 1.0
            beta1 = self._compute_beta1(group, t, ckp1)
            group["ckp1"] = ckp1
            group["beta1"] = beta1

            for p in group["params"]:

                if p.grad is None:
                    p.grad = torch.zeros_like(p)  # workaround for ddp nuisance unused param errors

                state = self.state[p]

                # compute the base update (unchanged from NorMuon / aux Adam implementation)
                if group["use_muon"]:
                    groups = getattr(p, "conv_groups", 1)

                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)

                        if group["normuon"]:
                            # shape change needed for grouped conv params
                            second_momentum_shape = (groups, p.shape[0] // groups, 1)
                            state["second_momentum_buffer"] = torch.zeros(
                                size=second_momentum_shape, device=p.device, dtype=p.dtype)
                        else:
                            state["second_momentum_buffer"] = None

                    z = self._get_z(p)
                    self._apply_weight_decay_at_y(p, z, lr, beta1)
                    p.lerp_(end=z, weight=1.0 - 1.0 / beta1) # y_t -> x_t

                    update = normuon_update(p.grad, state["momentum_buffer"], state["second_momentum_buffer"],
                        beta=group["momentum"], beta2=group["beta2"], groups=groups).reshape(p.shape)
                else:
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1

                    z = self._get_z(p)
                    self._apply_weight_decay_at_y(p, z, lr, beta1)
                    p.lerp_(end=z, weight=1.0 - 1.0 / beta1) # y_t -> x_t

                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])

                # apply update to z (decoupled weight decay on z), then rebuild y_{t+1}
                weight_decay = getattr(p, "weight_decay", group["weight_decay"])
                if weight_decay > 0:
                    z.mul_(max(0, 1 - lr * weight_decay))
                z.add_(update, alpha=-lr)
                p.lerp_(end=z, weight=ckp1)
                p.lerp_(end=z, weight=1.0 - beta1)

            group["k"] += 1

        return loss

    def _apply_weight_decay_at_y(self, p: torch.Tensor, z: torch.Tensor, lr: float, beta1: float) -> None:
        if self.weight_decay_at_y == 0.0:
            return
        z.sub_(p, alpha=lr * self.weight_decay_at_y)
        p.sub_(p, alpha=lr * self.weight_decay_at_y * (1.0 - beta1))
