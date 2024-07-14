from dataclasses import dataclass
from typing import Literal
from .sigma_sampler import SigmaSamplerConfig, SigmaSampler
from .trainer import ModuleTrainerConfig, DualDiffusionTrainer

@dataclass
class UNetTrainerConfig(ModuleTrainerConfig):

    sigma_distribution: Literal["ln_normal", "ln_sech", "ln_sech^2", "ln_linear", "ln_pdf"] = "ln_sech"
    sigma_dist_scale: float = 1.0
    sigma_dist_offset: float = 0.1

class UNetTrainer:
    
    def __init__(self, config: UNetTrainerConfig, trainer: DualDiffusionTrainer):

        self.config = config
        self.trainer = trainer

        sigma_sampler_config = SigmaSamplerConfig(
            sigma_max=self.trainer.module.config["sigma_max"],
            sigma_min=self.trainer.module.config["sigma_min"],
            sigma_data=self.trainer.module.config["sigma_data"],
            distribution=self.config.sigma_distribution,
            sigma_dist_scale=self.config.sigma_dist_scale,
            sigma_dist_offset=self.config.sigma_dist_offset
        )
        self.sigma_sampler = SigmaSampler(sigma_sampler_config)
                                          
    def get_config_class():
        return UNetTrainerConfig
    

