from dataclasses import dataclass, field, fields
from typing import Optional

import torch

@dataclass
class UNetConfig:

    #channel_mult: list[int]     # Per-resolution multipliers for the number of channels.
    #attn_levels: list[int]   # List of resolution levels to use self-attention.
    channel_mult: tuple[int] = (1,2)     # Per-resolution multipliers for the number of channels.
    attn_levels: tuple[int] = (1,2)  # List of resolution levels to use self-attention.

    model_channels: int  = 256               # Base multiplier for the number of channels.
    logvar_channels: int = 128               # Number of channels for training uncertainty estimation.
    
    channel_mult_noise: Optional[int] = None # Multiplier for noise embedding dimensionality.
    channel_mult_emb: Optional[int]   = None # Multiplier for final embedding dimensionality.
    channels_per_head: int    = 64           # Number of channels per attention head.
    num_layers_per_block: int = 2            # Number of resnet blocks per resolution.
    label_balance: float      = 0.5          # Balance between noise embedding (0) and class embedding (1).
    concat_balance: float     = 0.5          # Balance between skip connections (0) and main path (1).
    res_balance: float        = 0.3          # Balance between main branch (0) and residual branch (1).
    attn_balance: float       = 0.3          # Balance between main branch (0) and self-attention (1).
    
    mlp_multiplier: int = 2                  # Multiplier for the number of channels in the MLP.
    mlp_groups: int = 8                      # Number of groups for the MLPs.
    qk_attn_gain_exponent: float = 0.5       # Controls amount of modulation for attention qk gain / "sharpness"


    #__constants__: list[str] = field(init=False, default_factory=list)

    def __post_init__(self):
        type(self).__constants__ = [f.name for f in fields(self) if f.init]

class MyModule(torch.nn.Module):

    def __init__(self, config: UNetConfig):
        super().__init__()
        self.config = config

    def forward(self, x:torch.Tensor):
        a = x * self.config.model_channels
        return a * self.config.model_channels
    

module = MyModule(UNetConfig()).to("cuda")
print("__constants__:", UNetConfig.__constants__)
x = torch.tensor([1.0], device="cuda")

#backends = ['cudagraphs', 'inductor', 'onnxrt', 'openxla', 'openxla_eval', 'tvm']

#"""
module.forward = torch.compile(module.forward,
                               fullgraph=True,
                               dynamic=False,
                               mode="max-autotune"
                               #options = {"trace.graph_diagram": True, "triton.cudagraphs": True},
                               #backend="cudagraphs"
                               )
print(torch._dynamo.explain(module.forward)(x))


#module = torch.export.export(module, args=(torch.tensor([1.0], device="cuda"),))
#print(module)
#module = module.module()

print(module.forward)
print(module(x))
module.config.model_channels = 2
#print(module(x))
print(torch._dynamo.explain(module.forward)(x))
#"""

"""
from torch._dynamo.utils import CompileProfiler

with CompileProfiler() as prof:
    profiler_model = torch.compile(module.forward, backend=prof)
    print(profiler_model(x))
    print(prof.report())
    module.config.model_channels = 2
    print(profiler_model(x))
    print(prof.report())
"""