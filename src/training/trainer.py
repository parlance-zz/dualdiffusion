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

import utils.config as config

import logging
import math
import os
import shutil
import subprocess
import atexit
import importlib
import platform
import inspect
import contextlib
from datetime import datetime
from typing import Optional, Literal, Type, Union, Any
from dataclasses import dataclass
from traceback import format_exception
from fnmatch import fnmatch
from re import search

import torch
from torch.optim.lr_scheduler import LambdaLR
from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, GradientAccumulationPlugin, set_seed
from tqdm.auto import tqdm

from pipelines.dual_diffusion_pipeline import DualDiffusionPipeline
from utils.dual_diffusion_utils import dict_str, get_cuda_gpu_stats
from utils.compare_dirs import compare_dirs
from training.module_trainers.module_trainer import ModuleTrainer, ModuleTrainerConfig
from training.ema import EMA_Manager
from training.dataset import DatasetConfig, DualDiffusionDataset, custom_collate
from modules.module import DualDiffusionModule


class TrainLogger():

    def __init__(self, accelerator: Optional[Accelerator] = None) -> None:

        self.accelerator = accelerator
        self.channels: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def clear(self) -> None:
        self.channels.clear()
        self.counts.clear()

    @torch.no_grad()
    def add_log(self, key: str, value: Union[torch.Tensor, float]) -> None:

        if value is None:
            return
        
        if torch.is_tensor(value):
            if self.accelerator is not None:
                value = self.accelerator.gather(value.detach().cuda()).mean().item()
            else:
                value = value.detach().mean().item()

        if key in self.channels:
            self.channels[key] += value
            self.counts[key] += 1
        else:
            self.channels[key] = value
            self.counts[key] = 1

    def add_logs(self, logs: dict[str, Any]) -> None:
        for key, value in logs.items():
            self.add_log(key, value)

    def get_logs(self, sort: bool = False) -> dict:
        if sort == False:
            return {key: value / self.counts[key] for key, value in self.channels.items()}
        else:
            return dict(sorted(self.get_logs().items(), key=lambda item: item[1]))

@dataclass
class LRScheduleConfig:
    lr_schedule: Literal["edm2", "constant", "edm2_smooth"] = "edm2_smooth"
    learning_rate: float     = 2e-2
    lr_warmup_steps: int     = 150
    lr_reference_steps: int  = 150
    lr_decay_exponent: float = 1
    min_learning_rate: float = 1e-6

@dataclass
class OptimizerConfig:
    adam_beta1: float         = 0.9
    adam_beta2: float         = 0.99
    adam_epsilon: float       = 1e-8
    adam_weight_decay: float  = 0.

    loss_scale: float         = 250.
    max_grad_norm: float      = 100.
    grad_norm_std_ema_beta: float  = 0.9975
    grad_norm_mean_ema_beta: float = 0.9925
    dynamic_max_grad_norm_z: Optional[float] = 6

    muon_param_patterns: list[str] = ("*",)
    adam_param_patterns: list[str] = ("*gain*", "*balance*", "*logvar*", "*fourier*", "*emb_label_unconditional*", "*emb_noise*", "*bias*")
    muon_learning_rate_multiplier: float = 50
    muon_momentum_beta: float = 0.95
    muon_weight_decay: float = 0.
    muon_use_normuon: bool    = True
    muon_use_cc_scaling: bool = True

@dataclass
class DataLoaderConfig:
    load_datatypes: list[Literal["audio", "latents", "audio_embeddings", "text_embeddings"]] = ("audio", "audio_embeddings")
    load_splits: list[str] = ("train", "validation")
    filter_unnormalized_samples: bool = True
    filter_invalid_samples: bool = True
    dataset_num_proc: Optional[int] = None
    dataloader_num_workers: Optional[int] = 4
    pin_memory: bool = False
    prefetch_factor: Optional[int] = 2

@dataclass
class LoggingConfig:
    logging_dir: Optional[str] = None
    tensorboard_http_port:   Optional[int] = 6006
    tensorboard_num_scalars: Optional[int] = 2000

@dataclass
class DualDiffusionTrainerConfig:

    lr_schedule: LRScheduleConfig
    optimizer: OptimizerConfig
    dataloader: DataLoaderConfig
    logging: LoggingConfig

    module_trainer_class: Type
    module_trainer_config: ModuleTrainerConfig

    train_modules: list[str]
    module_name: str
    model_path: str
    model_name: str
    model_src_path: str
    train_config_path: Optional[str]          = None
    seed: Optional[int]                       = None
    emas: Optional[dict[str, dict[str, Any]]] = None
    device_batch_size: int                    = 8
    gradient_accumulation_steps: int          = 6
    validation_device_batch_size: int         = 6
    validation_accumulation_steps: int        = 10

    max_train_steps: int                = 1000000
    num_validation_epochs: int          = 10
    min_checkpoint_time: int            = 3600
    checkpoints_total_limit: int        = 1
    strict_checkpoint_time: bool        = False

    activation_memory_budget: Optional[float] = None
    enable_bf16_reduction_in_sdp: bool  = False
    enable_anomaly_detection: bool      = False
    enable_model_compilation: bool      = True
    enable_debug_mode: bool             = False
    enable_cuda_gpu_stats_logging: bool = True
    compile_params: Optional[dict]      = None

    @staticmethod
    def from_json(json_path, **kwargs) -> "DualDiffusionTrainerConfig":

        train_config = config.load_json(json_path)
        train_config["train_config_path"] = json_path

        for key, value in kwargs.items():
            train_config[key] = value
        
        train_config["lr_schedule"] = LRScheduleConfig(**train_config["lr_schedule"])
        train_config["optimizer"] = OptimizerConfig(**train_config["optimizer"])
        train_config["dataloader"] = DataLoaderConfig(**train_config["dataloader"])
        train_config["logging"] = LoggingConfig(**train_config["logging"])

        module_trainer_package = importlib.import_module(train_config["module_trainer"]["package"])
        module_trainer_class = getattr(module_trainer_package, train_config["module_trainer"]["class"])
        train_config.pop("module_trainer")

        module_trainer_config_class = module_trainer_class.config_class or inspect.signature(module_trainer_class.__init__).parameters["config"].annotation
        train_config["module_trainer_config"] = module_trainer_config_class(**train_config["module_trainer_config"])
        train_config["module_trainer_class"] = module_trainer_class

        if "train_modules" not in train_config or len(train_config["train_modules"]) == 0:
            assert train_config["module_name"]
            train_config["train_modules"] = [train_config["module_name"]]
        else:
            train_config["train_modules"] = sorted(train_config["train_modules"])
            train_config["module_name"] = "_".join(train_config["train_modules"])

        return DualDiffusionTrainerConfig(**train_config)

@dataclass
class TrainerPersistentState:
    total_samples_processed: int = 0
    total_train_hours: float = 0
    grad_norm_logmean: float = 0
    grad_norm_logvar: float = 0
    ext_state: Optional[dict[str, Any]] = None

class DualDiffusionTrainer:

    def __init__(self, train_config: DualDiffusionTrainerConfig) -> None:

        self.config = train_config
        self.persistent_state = TrainerPersistentState()
        self.persistent_state.grad_norm_logmean = float(math.log(train_config.optimizer.max_grad_norm))
        self.persistent_state.grad_norm_logvar = self.persistent_state.grad_norm_logmean

        self.init_accelerator()
        self.init_tensorboard()
        self.init_pytorch()
        self.init_module_pipeline()
        self.init_ema_manager()
        self.init_optimizer()
        self.init_checkpointing()
        self.init_lr_scheduler()
        self.init_dataloader()
        self.init_torch_compile()

        if self.config.num_validation_epochs > 0:
            self.optimizer, self.train_dataloader, self.validation_dataloader, self.lr_scheduler = self.accelerator.prepare(
                self.optimizer, self.train_dataloader, self.validation_dataloader, self.lr_scheduler
            )
        else:
            self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
                self.optimizer, self.train_dataloader, self.lr_scheduler
            )

    def init_accelerator(self) -> None:

        if self.config.logging.logging_dir is None:
            self.config.logging.logging_dir = os.path.join(self.config.model_path, f"logs_{self.config.module_name}")
        os.makedirs(self.config.logging.logging_dir, exist_ok=True)

        accelerator_project_config = ProjectConfiguration(project_dir=self.config.model_path,
                                                          logging_dir=self.config.logging.logging_dir)
        gradient_accumulation_plugin = GradientAccumulationPlugin(
            num_steps=self.config.gradient_accumulation_steps, sync_with_dataloader=False)
        #ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True, gradient_as_bucket_view=True)#static_graph=True)
        #ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)#, static_graph=True)

        self.accelerator = Accelerator(
            log_with="tensorboard",
            project_config=accelerator_project_config,
            gradient_accumulation_plugin=gradient_accumulation_plugin,
            #kwargs_handlers=[ddp_kwargs]
        )

        self.logger = get_logger("trainer", log_level="INFO")
        if self.accelerator.distributed_type == DistributedType.MULTI_GPU:
            log_filename_suffix = f"_ddp{self.accelerator.process_index}"
        else:
            log_filename_suffix = ""
        log_path = os.path.join(self.config.logging.logging_dir, f"train_{self.config.module_name}{log_filename_suffix}.log")
        logging.basicConfig(
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ],
            format=r"%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt=r"%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        self.logger.info(f"Logging to {log_path}")

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(self.config.model_name)

        if self.accelerator.mixed_precision == "bf16":
            self.mixed_precision_enabled = True
            self.mixed_precision_dtype = torch.bfloat16
        elif self.accelerator.mixed_precision == "fp16":
            self.mixed_precision_enabled = True
            self.mixed_precision_dtype = torch.float16
        else:
            self.mixed_precision_enabled = False
            self.mixed_precision_dtype = torch.float32

        self.logger.info(self.accelerator.state, main_process_only=False, in_order=True)
        self.logger.info(f"Distributed type: {self.accelerator.distributed_type}", main_process_only=False, in_order=True)
        self.accelerator.wait_for_everyone()

    def init_tensorboard(self) -> None:

        if self.accelerator.is_main_process and self.config.logging.tensorboard_http_port is not None:

            if shutil.which("tensorboard") is None:
                self.logger.warning("Make sure to install tensorboard if you want to use it for logging during training.")

            tensorboard_args = [
                "tensorboard",
                "--logdir",
                self.config.logging.logging_dir,
                "--bind_all",
                "--port",
                str(self.config.logging.tensorboard_http_port),
                "--samples_per_plugin",
                f"scalars={self.config.logging.tensorboard_num_scalars}",
            ]
            tensorboard_monitor_process = subprocess.Popen(tensorboard_args)

            def cleanup_process():
                try:
                    tensorboard_monitor_process.terminate()
                except Exception:
                    self.logger.warning("Failed to terminate tensorboard process")

            atexit.register(cleanup_process)

    def init_pytorch(self) -> None:

        if self.config.seed is not None:
            set_seed(self.config.seed, device_specific=True)
            self.logger.info(f"Using random seed {self.config.seed}")
        else:
            self.logger.info("Using random seed from system - Training may not be reproducible")

        if self.config.enable_anomaly_detection == True:
            torch.autograd.set_detect_anomaly(True)
            self.logger.info("Pytorch anomaly detection enabled")
        else:
            self.logger.info("Pytorch anomaly detection disabled")

        if self.config.enable_bf16_reduction_in_sdp == True:
            torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
            self.logger.info("BF16 reduction in Pytorch SDP enabled")

        if self.config.activation_memory_budget is not None:
            self.logger.info(f"Using activation memory budget: {self.config.activation_memory_budget}")
            import torch._functorch.config
            torch._functorch.config.activation_memory_budget = self.config.activation_memory_budget

    def init_module_pipeline(self) -> None:

        self.pipeline = DualDiffusionPipeline.from_pretrained(self.config.model_path)

        self.modules: list[DualDiffusionModule] = []; self.module_classes: list[type] = []
        for module_name in self.config.train_modules:
            if not hasattr(self.pipeline, module_name):
                raise ValueError(f"Module type '{module_name}' not registered in loaded pipeline")
            else:
                module = getattr(self.pipeline, module_name).requires_grad_(True).train()
                self.modules.append(module)
                self.module_classes.append(type(module))

        if self.config.enable_debug_mode == True:
            self.config.device_batch_size = 2
            self.config.validation_device_batch_size = 2

        self.sample_shape: tuple = self.pipeline.get_mel_spec_shape(bsz=self.config.device_batch_size)
        self.validation_sample_shape: tuple = self.pipeline.get_mel_spec_shape(bsz=self.config.validation_device_batch_size)
        if hasattr(self.pipeline, "dae"):
            self.latent_shape: tuple = self.pipeline.get_latent_shape(self.sample_shape)
            self.validation_latent_shape: tuple = self.pipeline.get_latent_shape(self.validation_sample_shape)
        else:
            self.latent_shape = None
            self.validation_latent_shape = None

        self.logger.info(f"Module classes: {[c.__name__ for c in self.module_classes]}")
        self.logger.info(f"Module trainer class: {self.config.module_trainer_class.__name__}")
        self.logger.info(f"Model metadata: {dict_str(self.pipeline.model_metadata)}")

        self.modules = self.accelerator.prepare(*self.modules)
        if not isinstance(self.modules, (tuple, list)):
            self.modules = [self.modules]
            
        if self.accelerator.distributed_type == DistributedType.MULTI_GPU:
            self.ddp_modules = self.modules
            self.modules = list([self.accelerator.unwrap_model(m) for m in self.modules])

        else:
            self.ddp_modules = self.modules

        if not isinstance(self.modules, (tuple, list)): self.modules = [self.modules]

        for module in self.modules:
            if hasattr(module, "normalize_weights"):
                module.normalize_weights()
    
    def get_train_module(self, module_name: str) -> DualDiffusionModule:
        if module_name not in self.config.train_modules:
            return None
        
        return self.modules[self.config.train_modules.index(module_name)]

    def get_ddp_module(self, module: DualDiffusionModule) -> DualDiffusionModule:
        return self.ddp_modules[self.modules.index(module)]
    
    def init_ema_manager(self) -> None:
        
        self.config.emas = self.config.emas or {}
        self.ema_managers: list[EMA_Manager] = []

        # only the main process saves checkpoints and processes EMA
        if self.accelerator.distributed_type == DistributedType.MULTI_GPU:
            if not self.accelerator.is_main_process:
                self.logger.info(f"EMA_Manager disabled on non-main process")
                return
        
        if len(self.config.emas) > 0:
            for module_name, module in zip(self.config.train_modules, self.modules):
                self.ema_managers.append(EMA_Manager(module_name, module, self.config.emas, self))

        if len(self.config.emas) == 0:
            self.logger.info("Not using EMA")
        else:
            self.logger.info("EMA_Emanager config:")
            self.logger.info(dict_str(self.config.emas))

        if any(len(m.get_validation_emas()) == 0 for m in self.ema_managers) and self.config.num_validation_epochs > 0:
            self.logger.error(f"Validation is enabled (num_validation_epochs: {self.config.num_validation_epochs}) but found no EMAs with include_in_validation=True")
            exit(1)

    def get_max_grad_norm(self) -> float:

        max_grad_norm = self.config.optimizer.max_grad_norm
        
        # if dynamic_max_grad_norm_z is not None then use dynamic grad norm
        if self.config.optimizer.dynamic_max_grad_norm_z is not None:
            
            grad_mean = math.exp(self.persistent_state.grad_norm_logmean)
            grad_std = math.exp(self.persistent_state.grad_norm_logvar / 2)
            max_grad_norm = grad_mean + grad_std * self.config.optimizer.dynamic_max_grad_norm_z

        return max_grad_norm

    def update_grad_norm_stats(self, grad_norm: float, eps: float = 1e-8) -> None:
        
        grad_norm = max(grad_norm, eps)
        mean_beta = self.config.optimizer.grad_norm_mean_ema_beta
        std_beta = self.config.optimizer.grad_norm_std_ema_beta

        log_mean = self.persistent_state.grad_norm_logmean
        log_var = self.persistent_state.grad_norm_logvar
        grad_var = max((grad_norm - math.exp(log_mean)) ** 2, eps)

        self.persistent_state.grad_norm_logmean = log_mean * mean_beta + (1 - mean_beta) * math.log(grad_norm)
        self.persistent_state.grad_norm_logvar = log_var * std_beta + (1 - std_beta) * math.log(grad_var)

    def get_momentum(self, optimizer: Optional[torch.optim.Optimizer] = None) -> torch.Tensor:

        optimizer = optimizer or self.optimizer

        # collect all momentum tensors
        momentum_tensors: list[torch.Tensor] = []
        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state.get(p, None)
                if state is not None:
                    exp_avg = state.get("exp_avg", None)
                    if exp_avg is None:
                        exp_avg = state.get("momentum_buffer", None)
                    if exp_avg is not None:
                        momentum_tensors.append(exp_avg)
        
        if len(momentum_tensors) == 0:
            return 0.0

        # compute norms with _foreach
        norms = torch._foreach_norm(momentum_tensors)
        return torch.linalg.vector_norm(torch.stack(norms, dim=0))
                
    def init_optimizer(self) -> None:
        
        muon_params = []; adam_params = []
        if len(self.config.optimizer.muon_param_patterns) == 0:

            opt_cls = torch.optim.AdamW

            for module in self.modules:
                adam_params.extend(module.parameters())

            self.optimizer = torch.optim.AdamW(
                adam_params,
                lr=self.config.lr_schedule.learning_rate,
                betas=(self.config.optimizer.adam_beta1, self.config.optimizer.adam_beta2),
                weight_decay=self.config.optimizer.adam_weight_decay,
                eps=self.config.optimizer.adam_epsilon,
                #foreach=True,
                fused=True,
            )

            self.use_muon = False
        else:
            try:
                from training.nor_muon import SingleDeviceNorMuonWithAuxAdam  # type: ignore
                opt_cls = SingleDeviceNorMuonWithAuxAdam
            except ImportError:
                self.logger.error("Import error: Unable to import muon and len(muon_param_patterns) > 0")
                exit(1)

            muon_param_names = []; adam_param_names = []
            for module in self.modules:
                for name, param in module.named_parameters():
                    muon_param = (any(fnmatch(name, pattern) for pattern in self.config.optimizer.muon_param_patterns) and
                                (not any(fnmatch(name, pattern) for pattern in self.config.optimizer.adam_param_patterns)))
                    
                    if (param.ndim <= 1 or param.shape[0] == 1 or param.shape[1] == 1) and muon_param == True:
                        self.logger.warning(f"Parameter '{name}' has shape {param.shape} which is unsuitable for Muon optimizer. Forcing AdamW for this parameter instead.")
                        muon_param = False

                    if muon_param == True:
                        muon_params.append(param)
                        muon_param_names.append(name)
                    else:
                        adam_params.append(param)
                        adam_param_names.append(name)

            if self.config.enable_debug_mode == True:
                self.logger.info(f"Muon  parameters: {muon_param_names}")
                self.logger.info(f"AdamW parameters: {adam_param_names}")

            param_groups = [
                {
                    "params": muon_params, "use_muon": True, "normuon": self.config.optimizer.muon_use_normuon,"cc_scaling": self.config.optimizer.muon_use_cc_scaling,
                    "lr": self.config.lr_schedule.learning_rate * self.config.optimizer.muon_learning_rate_multiplier,
                    "weight_decay": self.config.optimizer.muon_weight_decay, "momentum": self.config.optimizer.muon_momentum_beta
                }
            ]

            if len(adam_params) > 0:
                param_groups.append({
                    "params": adam_params, "use_muon": False,
                    "lr": self.config.lr_schedule.learning_rate, "weight_decay": self.config.optimizer.adam_weight_decay,
                    "betas": (self.config.optimizer.adam_beta1, self.config.optimizer.adam_beta2), "eps": self.config.optimizer.adam_epsilon
                })

            self.optimizer = SingleDeviceNorMuonWithAuxAdam(param_groups)
            self.use_muon = True

        self.logger.info(f"Using {opt_cls.__name__} optimizer with learning rate {self.config.lr_schedule.learning_rate}")
        self.logger.info(f"  AdamW param count: {len(adam_params)} Muon param count: {len(muon_params)}")
        if self.use_muon == True:
            self.logger.info(f"  Muon learning rate multiplier: {self.config.optimizer.muon_learning_rate_multiplier}")
            self.logger.info(f"  Muon momentum: {self.config.optimizer.muon_momentum_beta} weight decay: {self.config.optimizer.muon_weight_decay}")
            self.logger.info(f"  NorMuon: {self.config.optimizer.muon_use_normuon}")
            
        self.logger.info(f"  AdamW beta1: {self.config.optimizer.adam_beta1} beta2: {self.config.optimizer.adam_beta2}")
        self.logger.info(f"  AdamW eps: {self.config.optimizer.adam_epsilon} weight decay: {self.config.optimizer.adam_weight_decay}")
        
        if self.config.optimizer.dynamic_max_grad_norm_z is not None:
            self.logger.info(f"  Dynamic max grad norm enabled"
                             f"  std_ema_beta: {self.config.optimizer.grad_norm_std_ema_beta}"
                             f"  mean_ema_beta: {self.config.optimizer.grad_norm_mean_ema_beta}"
                             f"  max_z: {self.config.optimizer.dynamic_max_grad_norm_z})")
        else:
            self.logger.info(f"  Gradient clipping max norm: {self.config.optimizer.max_grad_norm}")

    def init_checkpointing(self) -> None:

        self.last_checkpoint_time = datetime.now()
        self.logger.info(f"Saving checkpoints every {self.config.min_checkpoint_time}s")
        
        def save_model_hook(models: list[DualDiffusionModule],
                            weights: list[dict[str, torch.Tensor]], output_dir: str) -> None:
            if self.accelerator.is_main_process:
                
                assert len(models) == len(weights) == len(self.config.train_modules) == len(self.modules)
                if len(self.ema_managers) > 0: assert len(self.ema_managers) == len(self.config.train_modules)

                for i, (module_name, module) in enumerate(zip(self.config.train_modules, self.modules)):
                    module.save_pretrained(output_dir, subfolder=module_name)

                    if len(self.ema_managers) > 0:
                        self.ema_managers[i].save(output_dir, subfolder=module_name)

                    weights.pop() # accelerate documentation says we need to do this, not sure why

                config.save_json(self.persistent_state.__dict__, os.path.join(output_dir, "trainer_state.json"))

        def load_model_hook(models: list[DualDiffusionModule], input_dir: str) -> None:

            assert len(models) == len(self.config.train_modules) == len(self.modules)
            if len(self.ema_managers) > 0: assert len(self.ema_managers) == len(self.config.train_modules)
            
            for i, (module_name, module_class, model) in enumerate(zip(self.config.train_modules, self.module_classes, models)):

                # copy loaded state and config into model
                load_model = module_class.from_pretrained(input_dir, subfolder=module_name)
                model.config = load_model.config
                model.load_state_dict(load_model.state_dict())
                del load_model
                
                if hasattr(model, "normalize_weights"):
                    model.normalize_weights()
                    
                if len(self.ema_managers) > 0: # load / init EMA weights
                    ema_load_errors = self.ema_managers[i].load(input_dir, subfolder=module_name, target_module=model)
                    
                    if len(ema_load_errors) > 0:
                        self.logger.error("\n  ".join([f"Errors loading EMA weights for {module_name}:"] + ema_load_errors))
                        if self.accelerator.is_main_process:
                            if input("Continue? (y/n): ").lower() not in ["y", "yes"]:
                                raise ValueError("Aborting training due to EMA load errors")
                        self.accelerator.wait_for_everyone()
                    else:
                        self.logger.info(f"Successfully loaded EMA weights for {module_name}")

            while len(models) > 0:
                models.pop()

            # load persistent trainer state
            trainer_state_path = os.path.join(input_dir, "trainer_state.json")
            try:
                self.persistent_state = TrainerPersistentState(**config.load_json(trainer_state_path))
                self.logger.info(f"Loaded persistent trainer state from {trainer_state_path}")
            except Exception as e:
                self.logger.error("".join(format_exception(type(e), e, e.__traceback__)))
                self.logger.error(f"Error loading persistent trainer state from {trainer_state_path}: {e}")
                if self.accelerator.is_main_process:
                    if input("Continue? (y/n): ").lower() not in ["y", "yes"]:
                        raise ValueError("Aborting training due to persistent trainer state load error")
            
            self.accelerator.wait_for_everyone()

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)
        self.logger.info("Registered accelerator hooks for model load/save")

        # make temporary copy of source code, model_index and train config to be copied to saved checkpoints
        if self.accelerator.is_main_process:
            
            tmp_path = os.path.join(self.config.model_path, "tmp")
            if os.path.isdir(tmp_path): shutil.rmtree(tmp_path)
            os.makedirs(tmp_path)

            shutil.copytree(self.config.model_src_path, os.path.join(tmp_path, "src"),
                            ignore=shutil.ignore_patterns("*.pyc", "__pycache__"),
                            dirs_exist_ok=True)
            
            if self.config.train_config_path is not None:
                shutil.copy(self.config.train_config_path, tmp_path)

            self.pipeline.save_pretrained(tmp_path, save_config_only=True)

    def init_lr_scheduler(self) -> None:

        self.logger.info(f"Using learning rate schedule {self.config.lr_schedule.lr_schedule}")
        self.logger.info(f" with warmup steps = {self.config.lr_schedule.lr_warmup_steps}")
        self.logger.info(f" reference steps = {self.config.lr_schedule.lr_reference_steps}")
        self.logger.info(f" decay exponent = {self.config.lr_schedule.lr_decay_exponent}")
        
        scaled_lr_warmup_steps = self.config.lr_schedule.lr_warmup_steps * self.accelerator.num_processes
        scaled_lr_reference_steps = self.config.lr_schedule.lr_reference_steps * self.accelerator.num_processes

        if self.config.lr_schedule.lr_schedule == "edm2_smooth":

            def lr_schedule(current_step: int) -> float:

                lr = 1.

                if current_step < scaled_lr_warmup_steps:
                    theta = current_step / scaled_lr_warmup_steps * math.pi + math.pi
                    lr *= (math.cos(theta) + 1) / 2
                
                lr /= 1 + (current_step / scaled_lr_reference_steps) ** self.config.lr_schedule.lr_decay_exponent

                return lr
        
        elif self.config.lr_schedule.lr_schedule == "edm2":

            def lr_schedule(current_step: int) -> float:
                lr = 1.
                if current_step < scaled_lr_warmup_steps:
                    lr *= current_step / scaled_lr_warmup_steps
                if current_step > scaled_lr_reference_steps:
                    lr /= (current_step / scaled_lr_reference_steps) ** self.config.lr_schedule.lr_decay_exponent
                    lr = max(lr * self.config.lr_schedule.learning_rate,
                        self.config.lr_schedule.min_learning_rate) / self.config.lr_schedule.learning_rate
                return lr
            
        elif self.config.lr_schedule.lr_schedule == "constant":

            def lr_schedule(current_step: int) -> float:
                if current_step < scaled_lr_warmup_steps:
                    return current_step / scaled_lr_warmup_steps
                return 1.

        else:
            raise ValueError(f"Unsupported learning rate schedule: {self.config.lr_schedule.lr_schedule}")

        if len(self.optimizer.param_groups) == 2:
            schedules = [lr_schedule, lr_schedule]
        else:
            schedules = lr_schedule
        self.lr_scheduler = LambdaLR(self.optimizer, schedules)
   
    def init_dataloader(self) -> None:
        
        if self.config.enable_debug_mode == True:
            self.config.dataloader.load_splits = ["debug"]

        self.local_batch_size = self.config.device_batch_size * self.config.gradient_accumulation_steps
        self.total_batch_size = self.local_batch_size * self.accelerator.num_processes
        self.validation_local_batch_size = self.config.validation_device_batch_size * self.config.validation_accumulation_steps
        self.validation_total_batch_size = self.validation_local_batch_size * self.accelerator.num_processes

        dataset_config = DatasetConfig(
            data_dir=config.DATASET_PATH,
            raw_crop_width=self.pipeline.format.get_raw_crop_width(),
            latents_crop_width=self.latent_shape[-1] if self.latent_shape is not None else 0,
            num_proc=self.config.dataloader.dataset_num_proc,
            load_datatypes=self.config.dataloader.load_datatypes,
            load_splits=self.config.dataloader.load_splits,
            filter_unnormalized_samples=self.config.dataloader.filter_unnormalized_samples,
            filter_invalid_samples=self.config.dataloader.filter_invalid_samples,
        )
        self.dataset = DualDiffusionDataset(dataset_config, self.pipeline.format.config, self.pipeline.embedding.config)

        self.train_split_name = self.config.dataloader.load_splits[0]
        self.train_dataloader = torch.utils.data.DataLoader(
            self.dataset[self.train_split_name], shuffle=True,
            batch_size=self.local_batch_size,
            num_workers=self.config.dataloader.dataloader_num_workers or 0,
            pin_memory=self.config.dataloader.pin_memory,
            persistent_workers=True if self.config.dataloader.dataloader_num_workers else False,
            prefetch_factor=self.config.dataloader.prefetch_factor if self.config.dataloader.dataloader_num_workers else None,
            drop_last=True, collate_fn=custom_collate
        )

        if self.config.num_validation_epochs > 0:
            self.validation_dataloader = torch.utils.data.DataLoader(self.dataset["validation"], batch_size=1)
        else:
            self.validation_dataloader = None

        self.logger.info(f"Using dataset path {config.DATASET_PATH} with {dataset_config.num_proc or 1} dataset processes)")
        self.logger.info(f"  {len(self.dataset[self.train_split_name])} samples ({self.dataset.num_filtered_samples[self.train_split_name]} filtered) in split '{self.train_split_name}'")
        if self.config.num_validation_epochs > 0:
            self.logger.info(f"  {len(self.dataset['validation'])} validation samples ({self.dataset.num_filtered_samples['validation']} filtered)")
        self.logger.info(f"Using train dataloader with {self.config.dataloader.dataloader_num_workers or 0} workers")
        self.logger.info(f"  prefetch_factor = {self.config.dataloader.prefetch_factor}")
        self.logger.info(f"  pin_memory = {self.config.dataloader.pin_memory}")

        self.num_update_steps_per_epoch = len(self.train_dataloader) // self.accelerator.num_processes

    def init_torch_compile(self) -> None:

        if self.config.enable_debug_mode == True:
            self.config.enable_model_compilation = False
            self.logger.info("Debug mode enabled - skipping model compilation")
            return
        
        if self.config.enable_model_compilation:
            if platform.system() == "Linux":
                self.config.compile_params = self.config.compile_params or {"fullgraph": True, "dynamic": False}
                self.logger.info(f"Compiling model(s) with options: {dict_str(self.config.compile_params)}")
            else:
                self.config.enable_model_compilation = False
                self.logger.warning("PyTorch model compilation is currently only supported on Linux - skipping compilation")
        else:
            self.logger.info("PyTorch model compilation is disabled")

    def save_checkpoint(self) -> None:
        
        save_path = os.path.join(
            self.config.model_path, f"{self.config.module_name}_checkpoint-{self.global_step}")

        # copy all source code / scripts / config to checkpoint folder for posterity
        source_src_path = os.path.join(self.config.model_path, "tmp")
        self.logger.info(f"Copying source code and config at '{source_src_path}' to checkpoint '{save_path}'")
        try:
            shutil.copytree(source_src_path, save_path, dirs_exist_ok=True)
        except Exception as e:
            self.logger.error("".join(format_exception(type(e), e, e.__traceback__)))
            self.logger.warning(f"Failed to copy source code from {source_src_path} to {save_path}: {e}")

        # save model checkpoint and training / optimizer state
        for module in self.modules:
            module.config.last_global_step = self.global_step
        self.accelerator.save_state(save_path)
        self.logger.info(f"Saved state to {save_path}")

        # copy logs
        source_logs_path = os.path.join(self.config.model_path, f"logs_{self.config.module_name}")
        target_logs_path = os.path.join(save_path, f"logs_{self.config.module_name}")
        self.logger.info(f"Copying logs at '{source_logs_path}' to checkpoint folder '{target_logs_path}'")
        try:
            shutil.copytree(source_logs_path, target_logs_path, dirs_exist_ok=True)
        except Exception as e:
            self.logger.error("".join(format_exception(type(e), e, e.__traceback__)))
            self.logger.warning(f"Failed to copy logs from {source_logs_path} to {target_logs_path}: {e}")

        # delete old checkpoints AFTER saving new checkpoint
        if self.config.checkpoints_total_limit is not None:
            try:
                checkpoints = os.listdir(self.config.model_path)
                checkpoints = [d for d in checkpoints if d.startswith(f"{self.config.module_name}_checkpoint")]
                checkpoints = sorted(checkpoints, key=lambda x: int(search(r'\d+', x.split('-')[1]).group()))

                if len(checkpoints) > self.config.checkpoints_total_limit:
                    num_to_remove = len(checkpoints) - self.config.checkpoints_total_limit
                    if num_to_remove > 0:
                        removing_checkpoints = checkpoints[0:num_to_remove]
                        self.logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                        self.logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(self.config.model_path, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)

            except Exception as e:
                self.logger.error("".join(format_exception(type(e), e, e.__traceback__)))
                self.logger.error(f"Error removing old checkpoints: {e}")

        self.logger.info("Checkpointing complete - Resuming Training...")
        self.last_checkpoint_time = datetime.now()

    def load_checkpoint(self) -> None:

        self.accum_step = 0
        self.local_step = 0
        self.global_step = 0
        self.resume_step = 0
        self.epoch = 0
        
        # get latest checkpoint path
        dirs = os.listdir(self.config.model_path)
        dirs = [d for d in dirs if d.startswith(f"{self.config.module_name}_checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(search(r'\d+', x.split('-')[1]).group()))
        path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            self.logger.warning(f"No existing checkpoints found, starting a new training run.")
            for module, module_name in zip(self.modules, self.config.train_modules):
                if module.config.last_global_step > 0:
                    self.logger.warning(f"Last global step in {module_name} module config is {module.config.last_global_step}, but no checkpoint found")
                    # todo: set global_step/resume_step/first_epoch and override step counts in optimizer / lr scheduler
        else:
            self.global_step = int(search(r'\d+', path).group())
            self.logger.info(f"Resuming from checkpoint {path} (global step: {self.global_step})")
            checkpoint_full_path = os.path.join(self.config.model_path, path)
            self.accelerator.load_state(checkpoint_full_path)

            # update any optimizer params that have changed
            target_adam_lr = self.config.lr_schedule.learning_rate
            target_muon_lr = target_adam_lr * self.config.optimizer.muon_learning_rate_multiplier

            updated_adam_learn_rate = False; updated_adam_betas = False
            updated_adam_weight_decay = False; updated_adam_eps = False
            updated_muon_learn_rate = False; updated_muon_momentum = False
            updated_muon_weight_decay = False

            for i, g in enumerate(self.optimizer.param_groups):

                if g.get("use_muon", False) == True:
                    if self.lr_scheduler.scheduler.base_lrs[i] != target_muon_lr:
                        g["initial_lr"] = target_muon_lr
                        self.lr_scheduler.scheduler.base_lrs[i] = target_muon_lr
                        updated_muon_learn_rate = True
                    if g["momentum"] != self.config.optimizer.muon_momentum_beta:
                        g["momentum"] = self.config.optimizer.muon_momentum_beta
                        updated_muon_momentum = True
                    if g["weight_decay"] != self.config.optimizer.muon_weight_decay:
                        g["weight_decay"] = self.config.optimizer.muon_weight_decay
                        updated_muon_weight_decay = True
                else:
                    if self.lr_scheduler.scheduler.base_lrs[i] != target_adam_lr:
                        g["initial_lr"] = target_adam_lr
                        self.lr_scheduler.scheduler.base_lrs[i] = target_adam_lr
                        updated_adam_learn_rate = True
                    if g["betas"] != (self.config.optimizer.adam_beta1, self.config.optimizer.adam_beta2):
                        g["betas"] = (self.config.optimizer.adam_beta1, self.config.optimizer.adam_beta2)
                        updated_adam_betas = True
                    if g["weight_decay"] != self.config.optimizer.adam_weight_decay:
                        g["weight_decay"] = self.config.optimizer.adam_weight_decay
                        updated_adam_weight_decay = True
                    if g["eps"] != self.config.optimizer.adam_epsilon:
                        g["eps"] = self.config.optimizer.adam_epsilon
                        updated_adam_eps = True

            if self.use_muon == True:
                if updated_muon_learn_rate:
                    self.logger.info(f"Using updated Muon learning rate: {target_muon_lr}")
                if updated_muon_momentum:
                    self.logger.info(f"Using updated Muon momentum: {self.config.optimizer.muon_momentum_beta}")
                if updated_muon_weight_decay:
                    self.logger.info(f"Using updated Muon weight decay: {self.config.optimizer.muon_weight_decay}")

            if updated_adam_learn_rate:
                self.logger.info(f"Using updated AdamW learning rate: {target_adam_lr}")
            if updated_adam_betas:
                self.logger.info(f"Using updated AdamW beta1: {self.config.optimizer.adam_beta1} beta2: {self.config.optimizer.adam_beta2}")
            if updated_adam_weight_decay:
                self.logger.info(f"Using updated AdamW weight decay: {self.config.optimizer.adam_weight_decay}")
            if updated_adam_eps:
                self.logger.info(f"Using updated AdamW epsilon: {self.config.optimizer.adam_epsilon}")

            # any and all source code or config changes differences relative
            # to the loaded checkpoint will be saved in this diff file
            if self.accelerator.is_main_process:
                tmp_path = os.path.join(self.config.model_path, "tmp")
                diff_output_path = os.path.join(tmp_path, "src_config_changes.diff")
                compare_dirs(diff_output_path, checkpoint_full_path, tmp_path,
                    ignore_patterns=["*_loss.json"], whitelist_patterns=["*.json", "*.py"])
                
                # find all diff files in logging dir with a global step after our resume global step
                # rename them to avoid confusion as they are not part of the resumed run
                logged_diff_files = [f for f in os.listdir(self.config.logging.logging_dir)
                                if f.startswith("src_config_changes_") and f.endswith(".diff")]
                
                for logged_diff_file in logged_diff_files:
                    logged_diff_output_path = os.path.join(self.config.logging.logging_dir, logged_diff_file)
                    if os.path.isfile(logged_diff_output_path):
                        logged_step = int(logged_diff_file.split("_")[3].split(".diff")[0])
                        if logged_step > self.global_step:
                            renamed_logged_diff_output_path = os.path.join(
                                self.config.logging.logging_dir, f"_{logged_diff_file}")
                            os.rename(logged_diff_output_path, renamed_logged_diff_output_path)

                # copy new diff output to logging dir
                logged_diff_output_path = os.path.join(
                    self.config.logging.logging_dir, f"src_config_changes_{str(self.global_step).zfill(7)}.diff")
                shutil.copy(diff_output_path, logged_diff_output_path)

        if self.global_step > 0:
            self.epoch = self.global_step // self.num_update_steps_per_epoch
            self.resume_step = self.global_step % self.num_update_steps_per_epoch
            self.local_step = self.resume_step
            self.accelerator.step = 0 # required to keep accum steps in sync if resuming after changing device batch size

            # if we have no persistent trainer state, recalculate the number of processed samples
            if self.persistent_state.total_samples_processed == 0:
                self.persistent_state.total_samples_processed = self.global_step * self.total_batch_size

    def train(self) -> None:

        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num examples = {len(self.dataset[self.train_split_name])}")
        self.logger.info(f"  Instantaneous batch size per device = {self.config.device_batch_size}")
        self.logger.info(f"  Gradient accumulation steps = {self.config.gradient_accumulation_steps}")
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {self.total_batch_size}")
        self.logger.info(f"  Total optimization steps for full run = {self.config.max_train_steps}")
        self.logger.info(f"  Path to save/load checkpoints = {self.config.model_path}")
        if self.sample_shape is not None:
            self.logger.info(f"  Sample shape: {self.sample_shape}")
        if self.latent_shape is not None:
            self.logger.info(f"  Latent shape: {self.latent_shape}")

        self.load_checkpoint()
        self.resume_dataloader = self.accelerator.skip_first_batches(self.train_dataloader, self.resume_step) if self.resume_step > 0 else None
        self.module_trainer: ModuleTrainer = self.config.module_trainer_class(self.config.module_trainer_config, self)

        # tracks / logs individual sample losses for anomalous sample detection
        self.train_sample_logger = TrainLogger(self.accelerator)
        self.validation_sample_logger = TrainLogger(self.accelerator)

        while True:        
            self.run_train_epoch()

            if self.accelerator.is_main_process:
                
                # save per-sample training loss statistics
                sample_log_path = os.path.join(self.config.model_path, "tmp", "sample_loss.json")
                try:
                    config.save_json(self.train_sample_logger.get_logs(sort=True), sample_log_path)
                except Exception as e:
                    self.logger.error("".join(format_exception(type(e), e, e.__traceback__)))
                    self.logger.warning(f"Error saving sample logs to {sample_log_path}: {e}")

                # if strict checkpointing is disabled check if we should save one at the end of the epoch
                if self.config.strict_checkpoint_time == False:
                    if (datetime.now() - self.last_checkpoint_time).total_seconds() >= self.config.min_checkpoint_time:
                        self.save_checkpoint()

            # if validation is enabled, run validation every n'th epoch
            self.accelerator.wait_for_everyone()
            if self.config.num_validation_epochs > 0:
                if self.epoch % self.config.num_validation_epochs == 0:
                    self.run_validation()

            # do switch ema if enabled
            if len(self.ema_managers) > 0:
                for module_name, ema_manager in zip(self.config.train_modules, self.ema_managers):
                    switched_ema_name = ema_manager.switch_ema()
                    if switched_ema_name is not None:
                        self.logger.info(f"Loaded train weights for {module_name} from switch EMA: {switched_ema_name}")

            if self.global_step >= self.config.max_train_steps: break
            else: self.epoch += 1

        # hurray!
        if self.accelerator.is_main_process:
            self.save_checkpoint()

        self.logger.info(f"Reached max train steps ({self.config.max_train_steps}) - Training complete")
        self.accelerator.end_training()

    def run_train_epoch(self):

        train_logger = TrainLogger(self.accelerator) # accumulates per-batch logs / statistics
        progress_bar = tqdm(total=self.num_update_steps_per_epoch, disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {self.epoch}")

        if self.resume_dataloader is not None:
            progress_bar.update(self.resume_step)
            progress_bar.refresh()
            self.local_step = self.resume_step
        else:
            self.local_step = 0

        last_sync_time = None # used for tracking total training time

        for local_batch in (self.resume_dataloader or self.train_dataloader):
            
            train_logger.clear() # accumulates per-batch logs / statistics
            batch_init_logs = self.module_trainer.init_batch()
            if batch_init_logs is not None:
                train_logger.add_logs(batch_init_logs)
        
            for self.accum_step in range(self.config.gradient_accumulation_steps):
                    
                with self.accelerator.accumulate(*self.ddp_modules):

                    device_batch = { # get sub-batch of device batch size from local batch for each grad_accum_step
                        key: value[self.accum_step * self.config.device_batch_size: (self.accum_step+1) * self.config.device_batch_size]
                        for key, value in local_batch.items()
                    }

                    module_logs = self.module_trainer.train_batch(device_batch)
                    train_logger.add_logs(module_logs)
                    for i, sample_path in enumerate(device_batch["sample_paths"]):
                        self.train_sample_logger.add_log(sample_path, module_logs["loss"][i])

                # loss is multiplied by grad accum steps for consistent grad norm
                self.accelerator.backward(module_logs["loss"].mean() * self.config.optimizer.loss_scale)

                if self.accelerator.sync_gradients:
                    assert self.accum_step == (self.config.gradient_accumulation_steps - 1), \
                        f"accum_step out of sync with sync_gradients: {self.accum_step} != {self.config.gradient_accumulation_steps - 1}"
                    
                    # clip grad norm and check for inf/nan grad
                    max_grad_norm = self.get_max_grad_norm()

                    opt_params = [] # collect params for all train modules
                    for module_name, module in zip(self.config.train_modules, self.modules):
                        module_params = tuple(module.parameters())
                        opt_params.extend(module_params)

                        # if training multiple modules log individual module grad norms
                        if len(self.config.train_modules) > 1:
                            module_params = [p.grad for p in module_params if p.grad is not None]
                            if len(module_params) > 0:
                                module_param_norms = torch._foreach_norm(module_params)
                                module_grad_norm = torch.linalg.vector_norm(torch.tensor(module_param_norms))
                                train_logger.add_log(f"grad_norm/{module_name}", module_grad_norm)
                            else:
                                train_logger.add_log(f"grad_norm/{module_name}", 0)

                    grad_norm = self.accelerator.clip_grad_norm_(opt_params, max_grad_norm).item()
                    train_logger.add_log("grad_norm", grad_norm)
                    train_logger.add_log("grad_norm/max", max_grad_norm)
                    train_logger.add_log("grad_norm/clipped", min(max_grad_norm, grad_norm))
                    train_logger.add_log("grad_norm/ema_mean", math.exp(self.persistent_state.grad_norm_logmean))
                    train_logger.add_log("grad_norm/ema_std", math.exp(self.persistent_state.grad_norm_logvar/2))
                    
                    self.update_grad_norm_stats(grad_norm)
                    
                    if math.isinf(grad_norm) or math.isnan(grad_norm):
                        self.logger.warning(f"Warning: grad norm is {grad_norm} step={self.global_step}")
                    if math.isnan(grad_norm):
                        self.logger.error(f"Error: grad norm is {grad_norm}, aborting...")
                        if self.accelerator.is_main_process:
                            import pdb; pdb.set_trace()
                        else:
                            exit(1)
                else:
                    #assert self.accum_step != (self.config.gradient_accumulation_steps - 1), \
                    #    f"accum_step out of sync, no sync_gradients but {self.accum_step} == {self.config.gradient_accumulation_steps - 1}"
                    if self.accum_step == (self.config.gradient_accumulation_steps - 1):
                        self.logger.warning(f"Finished all grad accumulation steps but accelerator.sync_gradients is False. self.accum_step: {self.accum_step}")

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

            # weights have now been updated in the last optimizer step
            if self.accelerator.sync_gradients:
                
                """
                if self.global_step % 100 == 0:
                    def flat_checksum(model):
                        import torch
                        s = torch.zeros((), device=self.accelerator.device, dtype=torch.float64)
                        numel = 0
                        with torch.no_grad():
                            for p in model.parameters():
                                s += p.detach().double().sum()
                                numel += p.numel()
                        return s / numel

                    # after optimizer.step()
                    cs = flat_checksum(self.modules[1])
                    all_cs = self.accelerator.gather(cs).cpu().numpy()
                    param_error = all_cs[0] - all_cs[1]
                    self.logger.info(f"Avg param error: {param_error}")
                """
                
                # log total train time, total samples processed, epoch #, and it/s stats
                if last_sync_time is not None:
                    self.persistent_state.total_train_hours += (
                        datetime.now() - last_sync_time).total_seconds() / 3600
                    train_logger.add_logs({
                        "train_stats/total_train_hours": self.persistent_state.total_train_hours})
                last_sync_time = datetime.now()

                train_logger.add_logs({
                    "train_stats/total_samples_processed": self.persistent_state.total_samples_processed})
                if progress_bar.format_dict["rate"] is not None:
                    train_logger.add_logs({
                        "train_stats/it_per_second": progress_bar.format_dict["rate"]})
                
                if self.local_step == 0:
                    train_logger.add_log("train_stats/epoch", self.epoch)
                
                # log average momentum norm
                train_logger.add_log("grad_norm/momentum", self.get_momentum())

                # optionally, log cuda gpu stats (every 25th step to avoid overhead)
                if self.config.enable_cuda_gpu_stats_logging == True and self.global_step % 25 == 0:
                    try:
                        for idx, stats in enumerate(get_cuda_gpu_stats()):
                            train_logger.add_logs({
                                f"gpu_stats/temp_{idx}": stats["temperature_C"],
                                f"gpu_stats/power_{idx}": stats["power_W"],
                                f"gpu_stats/vram_{idx}": stats["vram_used_MB"] / 1024,
                            })
                    except Exception as e:
                        self.logger.warning(f"Error logging CUDA GPU stats: {e}")

                # update emas and normalize weights
                for ema_manager in self.ema_managers:
                    ema_manager.update()
                for module in self.modules:
                    module.normalize_weights()

                # ema config is shared between train modules for now so we don't need to log all ema managers
                if len(self.ema_managers) > 0: 
                    train_logger.add_logs({f"ema_betas/{name}": beta for
                        name, beta in self.ema_managers[0].get_ema_betas().items()})
                
                # update logs
                batch_finish_logs = self.module_trainer.finish_batch()
                if batch_finish_logs is not None:
                    train_logger.add_logs(batch_finish_logs)
                
                last_learn_rates = self.lr_scheduler.get_last_lr()
                if self.use_muon == True:
                    train_logger.add_logs({"learn_rate/muon": last_learn_rates[0]})
                train_logger.add_logs({"learn_rate/adamw": last_learn_rates[-1]})
                train_logger.add_log("learn_rate/muon_base_rate", self.lr_scheduler.scheduler.base_lrs[0])

                logs = train_logger.get_logs()
                self.accelerator.log(logs, step=self.global_step)

                # update progress global_steps, total_samples_processed and progress bar
                self.local_step += 1
                self.global_step += 1
                self.persistent_state.total_samples_processed += self.total_batch_size

                progress_bar.update(1)
                progress_bar.set_postfix(loss=logs["loss"], grad_norm=logs["grad_norm"], global_step=self.global_step)
                
                if self.accelerator.is_main_process:
                    # if using strict checkpoint time, save it immediately instead of at the end of the epoch
                    _save_checkpoint = False
                    if self.config.strict_checkpoint_time == True:
                        if (datetime.now() - self.last_checkpoint_time).total_seconds() >= self.config.min_checkpoint_time:
                            _save_checkpoint = True

                    # saves a checkpoint immediately if a file named "_save_checkpoint" is found in the model path
                    _save_checkpoint_path = os.path.join(self.config.model_path, "_save_checkpoint")
                    if os.path.isfile(_save_checkpoint_path): _save_checkpoint = True
                    if _save_checkpoint == True:
                        self.save_checkpoint()
                        if os.path.isfile(_save_checkpoint_path):
                            os.remove(_save_checkpoint_path)
                        last_sync_time = datetime.now() # exclude checkpoint saving time from train total time
                        progress_bar.refresh()
            else:
                #assert False, "finished local_batch but accelerator.sync_gradients isn't True"
                self.logger.warning(f"Finished local batch but accelerator.sync_gradients isn't True. self.accum_steps: {self.accum_step}")

            if self.global_step >= self.config.max_train_steps: break

        progress_bar.close()
        self.accelerator.wait_for_everyone()
        self.resume_dataloader = None # throw away resume dataloader (if any) at end of epoch
        
    @torch.no_grad()
    def run_validation(self):

        raise NotImplementedError("Validation needs refactor for multi-module training")
    
        self.logger.info("***** Running validation *****")
        self.logger.info(f"  Epoch = {self.global_step // self.num_update_steps_per_epoch} (step: {self.global_step})")
        self.logger.info(f"  Num examples = {len(self.dataset['validation'])}")
        self.logger.info(f"  Total validation batch size (w. parallel, distributed & accumulation) = {self.validation_total_batch_size}")
        if self.validation_sample_shape is not None:
            self.logger.info(f"  Sample shape: {self.validation_sample_shape}")
        if self.validation_latent_shape is not None:
            self.logger.info(f"  Latent shape: {self.validation_latent_shape}")

        # create a backup copy of train weights and set model to eval mode
        start_validation_time = datetime.now()
        self.module.eval()
        backup_module_state_dict = {k: v.cpu() for k, v in self.module.state_dict().items()}

        # get validation logs / losses for each ema
        for ema_name in self.ema_manager.get_validation_emas():

            self.module.load_state_dict(self.ema_manager.ema_modules[ema_name])
            self.module.normalize_weights()

            ema_validation_logs = self.run_validation_epoch(ema_name)
            loss_validation = ema_validation_logs[f"loss_validation_ema_{ema_name}"]
            self.logger.info(f"EMA: {ema_name} - loss_validation: {loss_validation}")
            self.accelerator.log(ema_validation_logs, step=self.global_step)

        self.module.load_state_dict(backup_module_state_dict)
        del backup_module_state_dict

        self.logger.info(f"Validation complete (runtime: {(datetime.now() - start_validation_time).total_seconds()}s)")
        self.module.train()

    @torch.inference_mode()
    def run_validation_epoch(self, ema_name: str) -> dict:

        raise NotImplementedError("Validation needs refactor for multi-module training")
    
        validation_logger = TrainLogger(self.accelerator)
        progress_bar = tqdm(total=len(self.validation_dataloader), disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description(f"Validation ema_{ema_name}")

        for batch in self.validation_dataloader:
            
            validation_batch = {} # expand each individual validation sample to validation_device_batch_size
            for key, value in batch.items():
                if torch.is_tensor(value):
                    validation_batch[key] = value.repeat((self.config.validation_device_batch_size,) + (1,) * (value.ndim - 1))
                elif isinstance(value, list):
                    validation_batch[key] = value * self.config.validation_device_batch_size
                else:
                    raise ValueError(f"Unsupported validation batch value type: {type(value)} '{key}': '{value}'")

            self.module_trainer.init_batch(validation=True)
            for self.accum_step in range(self.config.validation_accumulation_steps):
                module_logs = self.module_trainer.train_batch(validation_batch)
                validation_logger.add_logs(module_logs)

                # log per-sample validation loss
                for i, sample_path in enumerate(validation_batch["sample_paths"]):
                    self.validation_sample_logger.add_log(sample_path, module_logs["loss"][i])

            validation_logger.add_logs(self.module_trainer.finish_batch())
            progress_bar.update(1)
        
        progress_bar.close()

        # validation log labels use _validation suffix
        validation_logs = {}
        for key, value in validation_logger.get_logs().items():
            key = key.replace("/", f"_validation_ema_{ema_name}/", 1) if "/" in key else key + f"_validation_ema_{ema_name}"
            validation_logs[key] = value

        # save per-sample validation loss statistics
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            sample_log_path = os.path.join(self.config.model_path, "tmp", f"sample_loss_validation_ema_{ema_name}.json")
            try:
                config.save_json(self.validation_sample_logger.get_logs(sort=True), sample_log_path)
            except Exception as e:
                self.logger.error("".join(format_exception(type(e), e, e.__traceback__)))
                self.logger.warning(f"Error saving validation sample logs to {sample_log_path}: {e}")

        return validation_logs