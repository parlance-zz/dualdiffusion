from dataclasses import dataclass

from .sigma_sampler import SigmaSampler
from .trainer import ModuleTrainerConfig, DualDiffusionTrainer

@dataclass
class UNetTrainerConfig(ModuleTrainerConfig):
    pass

class UNetTrainer:
    
    def __init__(self, config: UNetTrainerConfig, trainer: DualDiffusionTrainer):

        self.config = config
        self.trainer = trainer

        self.sigma_sampler = SigmaSampler(self.trainer.module.config["sigma_max"],
                                          self.trainer.module.config["sigma_min"],
                                          self.trainer.module.config["sigma_data"],
                                          self.config.sigma_distribution,
                                          self.config.sigma_dist_scale,
                                          self.config.sigma_dist_offset)
                                          
    def get_config_class():
        return UNetTrainerConfig
    

