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
from datetime import datetime
from typing import Optional, Literal, Type, Union, Any
from dataclasses import dataclass

import torch
import torch.utils.checkpoint
from torch.optim.lr_scheduler import LambdaLR
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, GradientAccumulationPlugin, set_seed
from tqdm.auto import tqdm

from pipelines.dual_diffusion_pipeline import DualDiffusionPipeline
from utils.dual_diffusion_utils import dict_str
from training.module_trainers.module_trainer import ModuleTrainer, ModuleTrainerConfig
from training.ema import EMA_Manager
from training.dataset import DatasetConfig, DualDiffusionDataset
from modules.module import DualDiffusionModule


class TrainLogger():

    def __init__(self, accelerator: Accelerator) -> None:

        self.accelerator = accelerator
        self.channels: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def clear(self) -> None:
        self.channels.clear()
        self.counts.clear()

    @torch.inference_mode()
    def add_log(self, key: str, value: Union[torch.Tensor, float]) -> None:
        if torch.is_tensor(value):
            value = self.accelerator.gather(value.detach()).mean().item()

        if key in self.channels:
            self.channels[key] += value
            self.counts[key] += 1
        else:
            self.channels[key] = value
            self.counts[key] = 1

    def add_logs(self, logs) -> None:
        for key, value in logs.items():
            self.add_log(key, value)

    def get_logs(self, sorted: bool = False) -> dict:
        if sorted == False:
            return {key: value / self.counts[key] for key, value in self.channels.items()}
        else:
            return dict(sorted(self.get_logs().items(), key=lambda item: item[1]))

@dataclass
class LRScheduleConfig:
    lr_schedule: Literal["edm2", "constant"] = "edm2"
    learning_rate: float     = 3e-3
    lr_warmup_steps: int     = 5000
    lr_reference_steps: int  = 70000
    lr_decay_exponent: float = 0.5
    min_learning_rate: float = 1e-4

@dataclass
class OptimizerConfig:
    adam_beta1: float        = 0.9
    adam_beta2: float        = 0.99
    adam_epsilon: float      = 1e-8
    adam_weight_decay: float = 0.
    max_grad_norm: float     = 1.

@dataclass
class DataLoaderConfig:
    load_datatypes: list[Literal["audio", "latents", "audio_embeddings", "text_embeddings"]] = ("audio", "audio_embeddings")
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

    enable_anomaly_detection: bool      = False
    enable_model_compilation: bool      = True
    enable_channels_last: bool          = True
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

        return DualDiffusionTrainerConfig(**train_config)

@dataclass
class TrainerPersistentState:
    total_samples_processed: int = 0

class DualDiffusionTrainer:

    def __init__(self, train_config: DualDiffusionTrainerConfig) -> None:

        self.config = train_config
        self.persistent_state = TrainerPersistentState()

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

        self.optimizer, self.train_dataloader, self.validation_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.optimizer, self.train_dataloader, self.validation_dataloader, self.lr_scheduler
        )

    def init_accelerator(self) -> None:

        if self.config.logging.logging_dir is None:
            self.config.logging.logging_dir = os.path.join(self.config.model_path, f"logs_{self.config.module_name}")
        os.makedirs(self.config.logging.logging_dir, exist_ok=True)

        accelerator_project_config = ProjectConfiguration(project_dir=self.config.model_path,
                                                          logging_dir=self.config.logging.logging_dir)
        gradient_accumulation_plugin = GradientAccumulationPlugin(
            num_steps=self.config.gradient_accumulation_steps, sync_with_dataloader=False)
        self.accelerator = Accelerator(
            log_with="tensorboard",
            project_config=accelerator_project_config,
            gradient_accumulation_plugin=gradient_accumulation_plugin,
        )

        self.logger = get_logger("trainer", log_level="INFO")
        log_path = os.path.join(self.config.logging.logging_dir, f"train_{self.config.module_name}.log")
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

        if self.config.enable_anomaly_detection:
            torch.autograd.set_detect_anomaly(True)
            self.logger.info("Pytorch anomaly detection enabled")
        else:
            self.logger.info("Pytorch anomaly detection disabled")

    def init_module_pipeline(self) -> None:

        self.pipeline = DualDiffusionPipeline.from_pretrained(self.config.model_path)

        if not hasattr(self.pipeline, self.config.module_name):
            raise ValueError(f"Module type '{self.config.module_name}' not registered in loaded pipeline")
        self.module: DualDiffusionModule = getattr(self.pipeline, self.config.module_name).requires_grad_(True).train()
        self.module_class: Type[DualDiffusionModule] = type(self.module)

        self.sample_shape: tuple = self.pipeline.get_sample_shape(bsz=self.config.device_batch_size)
        self.validation_sample_shape: tuple = self.pipeline.get_sample_shape(bsz=self.config.validation_device_batch_size)
        if hasattr(self.pipeline, "vae"):
            self.latent_shape: tuple = self.pipeline.get_latent_shape(self.sample_shape)
            self.validation_latent_shape: tuple = self.pipeline.get_latent_shape(self.validation_sample_shape)
        else:
            self.latent_shape = None
            self.validation_latent_shape = None

        self.logger.info(f"Module class: {self.module_class.__name__}")
        self.logger.info(f"Module trainer class: {self.config.module_trainer_class.__name__}")
        self.logger.info(f"Model metadata: {dict_str(self.pipeline.model_metadata)}")

        if self.config.enable_channels_last == True:
            self.module = self.module.to(memory_format=torch.channels_last)

        self.module = self.accelerator.prepare(self.module)
        if isinstance(self.module, torch.nn.parallel.DistributedDataParallel):
            self.module = self.module.module
        
        if hasattr(self.module, "normalize_weights"):
            self.module.normalize_weights()
        
    def init_ema_manager(self) -> None:
        
        self.config.emas = self.config.emas or {}
        self.ema_manager = EMA_Manager(self.config.emas, self)
        if len(self.config.emas) == 0:
            self.logger.info("Not using EMA")
        else:
            self.logger.info("EMA_Emanager config:")
            self.logger.info(dict_str(self.config.emas))

        if len(self.ema_manager.get_validation_emas()) == 0 and self.config.num_validation_epochs > 0:
            self.logger.error(f"Validation is enabled (num_validation_epochs: {self.config.num_validation_epochs}) but found no EMAs with include_in_validation=True")
            exit(1)

    def init_optimizer(self) -> None:
        
        self.optimizer = torch.optim.AdamW(
            self.module.parameters(),
            lr=self.config.lr_schedule.learning_rate,
            betas=(self.config.optimizer.adam_beta1, self.config.optimizer.adam_beta2),
            weight_decay=self.config.optimizer.adam_weight_decay,
            eps=self.config.optimizer.adam_epsilon,
            #foreach=True,
            fused=True,
        )

        self.logger.info(f"Using AdamW optimiser with learning rate {self.config.lr_schedule.learning_rate}")
        self.logger.info(f"  AdamW beta1: {self.config.optimizer.adam_beta1} beta2: {self.config.optimizer.adam_beta2}")
        self.logger.info(f"  AdamW eps: {self.config.optimizer.adam_epsilon} weight decay: {self.config.optimizer.adam_weight_decay}")
        self.logger.info(f"  Gradient clipping max norm: {self.config.optimizer.max_grad_norm}")
        
    def init_checkpointing(self) -> None:

        self.last_checkpoint_time = datetime.now()
        self.logger.info(f"Saving checkpoints every {self.config.min_checkpoint_time}s")
        
        def save_model_hook(models: list[DualDiffusionModule],
                            weights: list[dict[str, torch.Tensor]], output_dir: str) -> None:
            if self.accelerator.is_main_process:
                
                if len(models) != 1: # as below, not sure why len(models) would be > 1
                    self.logger.warning(f"Found {len(models)} models in save_model_hook, expected 1")

                for model in models:
                    model.save_pretrained(output_dir, subfolder=self.config.module_name)
                    weights.pop() # accelerate documentation says we need to do this, not sure why

                self.ema_manager.save(output_dir, subfolder=self.config.module_name)
                config.save_json(self.persistent_state.__dict__, os.path.join(output_dir, "trainer_state.json"))

        def load_model_hook(models: list[DualDiffusionModule], input_dir: str) -> None:
            assert len(models) == 1 # not sure why len(models) would be > 1
            model = models.pop()

            # copy loaded state and config into model
            load_model = self.module_class.from_pretrained(input_dir, subfolder=self.config.module_name)
            model.config = load_model.config
            model.load_state_dict(load_model.state_dict())
            del load_model
            
            if hasattr(model, "normalize_weights"):
                model.normalize_weights()
                
            # load / init EMA weights
            ema_load_errors = self.ema_manager.load(input_dir,
                subfolder=self.config.module_name, target_module=model)
            
            if len(ema_load_errors) > 0:
                self.logger.error("\n  ".join(["Errors loading EMA weights:"] + ema_load_errors))
                if self.accelerator.is_main_process:
                    if input("Continue? (y/n): ").lower() not in ["y", "yes"]:
                        raise ValueError("Aborting training due to EMA load errors")
                self.accelerator.wait_for_everyone()
            elif len(self.ema_manager.ema_configs) > 0:
                self.logger.info(f"Successfully loaded EMA weights")

            # load persistent trainer state
            trainer_state_path = os.path.join(input_dir, "trainer_state.json")
            try:
                self.persistent_state = TrainerPersistentState(**config.load_json(trainer_state_path))
                self.logger.info(f"Loaded persistent trainer state from {trainer_state_path}")
            except Exception as e:
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
            shutil.copy(os.path.join(self.config.model_path, "model_index.json"), tmp_path)
            if self.config.train_config_path is not None:
                shutil.copy(self.config.train_config_path, tmp_path)

    def init_lr_scheduler(self) -> None:

        self.logger.info(f"Using learning rate schedule {self.config.lr_schedule.lr_schedule}")
        self.logger.info(f" with warmup steps = {self.config.lr_schedule.lr_warmup_steps}")
        self.logger.info(f" reference steps = {self.config.lr_schedule.lr_reference_steps}")
        self.logger.info(f" decay exponent = {self.config.lr_schedule.lr_decay_exponent}")
        
        scaled_lr_warmup_steps = self.config.lr_schedule.lr_warmup_steps * self.accelerator.num_processes
        scaled_lr_reference_steps = self.config.lr_schedule.lr_reference_steps * self.accelerator.num_processes

        if self.config.lr_schedule.lr_schedule == "edm2":

            def lr_schedule(current_step: int) -> float:
                lr = 1.
                if current_step < scaled_lr_warmup_steps:
                    lr *= current_step / scaled_lr_warmup_steps
                if current_step > scaled_lr_reference_steps:
                    lr *= (scaled_lr_reference_steps / current_step) ** self.config.lr_schedule.lr_decay_exponent
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
        
        self.lr_scheduler = LambdaLR(self.optimizer, lr_schedule)
   
    def init_dataloader(self) -> None:
        
        self.local_batch_size = self.config.device_batch_size * self.config.gradient_accumulation_steps
        self.total_batch_size = self.local_batch_size * self.accelerator.num_processes
        self.validation_local_batch_size = self.config.validation_device_batch_size * self.config.validation_accumulation_steps
        self.validation_total_batch_size = self.validation_local_batch_size * self.accelerator.num_processes

        dataset_config = DatasetConfig(
            data_dir=config.DATASET_PATH, cache_dir=config.CACHE_PATH,
            latents_crop_width=self.latent_shape[-1] if self.latent_shape is not None else 0,
            num_proc=self.config.dataloader.dataset_num_proc,
            load_datatypes=self.config.dataloader.load_datatypes,
            filter_invalid_samples=self.config.dataloader.filter_invalid_samples,
        )
        self.dataset = DualDiffusionDataset(dataset_config, self.pipeline.format.config)

        self.train_dataloader = torch.utils.data.DataLoader(
            self.dataset["train"], shuffle=True,
            batch_size=self.local_batch_size,
            num_workers=self.config.dataloader.dataloader_num_workers or 0,
            pin_memory=self.config.dataloader.pin_memory,
            persistent_workers=True if self.config.dataloader.dataloader_num_workers else False,
            prefetch_factor=self.config.dataloader.prefetch_factor if self.config.dataloader.dataloader_num_workers else None,
            drop_last=True,
        )
        self.validation_dataloader = torch.utils.data.DataLoader(self.dataset["validation"], batch_size=1)

        self.logger.info(f"Using dataset path {config.DATASET_PATH} with {dataset_config.num_proc or 1} dataset processes)")
        self.logger.info(f"  {len(self.dataset['train'])} train samples ({self.dataset.num_filtered_samples['train']} filtered)")
        self.logger.info(f"  {len(self.dataset['validation'])} validation samples ({self.dataset.num_filtered_samples['validation']} filtered)")
        self.logger.info(f"Using train dataloader with {self.config.dataloader.dataloader_num_workers or 0} workers")
        self.logger.info(f"  prefetch_factor = {self.config.dataloader.prefetch_factor}")
        self.logger.info(f"  pin_memory = {self.config.dataloader.pin_memory}")

        self.num_update_steps_per_epoch = len(self.train_dataloader) // self.accelerator.num_processes

    def init_torch_compile(self) -> None:

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
        
        # save model checkpoint and training / optimizer state
        self.module.config.last_global_step = self.global_step
        save_path = os.path.join(self.config.model_path, f"{self.config.module_name}_checkpoint-{self.global_step}")
        self.accelerator.save_state(save_path)
        self.logger.info(f"Saved state to {save_path}")

        # copy all source code / scripts / config to checkpoint folder for posterity
        source_src_path = os.path.join(self.config.model_path, "tmp")
        self.logger.info(f"Copying source code and config at '{source_src_path}' to checkpoint '{save_path}'")
        try:
            shutil.copytree(source_src_path, save_path, dirs_exist_ok=True)
        except Exception as e:
            self.logger.warning(f"Failed to copy source code from {source_src_path} to {save_path}: {e}")

        # copy logs
        source_logs_path = os.path.join(self.config.model_path, f"logs_{self.config.module_name}")
        target_logs_path = os.path.join(save_path, f"logs_{self.config.module_name}")
        self.logger.info(f"Copying logs at '{source_logs_path}' to checkpoint folder '{target_logs_path}'")
        try:
            shutil.copytree(source_logs_path, target_logs_path, dirs_exist_ok=True)
        except Exception as e:
            self.logger.warning(f"Failed to copy logs from {source_logs_path} to {target_logs_path}: {e}")

        # delete old checkpoints AFTER saving new checkpoint
        if self.config.checkpoints_total_limit is not None:
            try:
                checkpoints = os.listdir(self.config.model_path)
                checkpoints = [d for d in checkpoints if d.startswith(f"{self.config.module_name}_checkpoint")]
                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

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
                self.logger.error(f"Error removing old checkpoints: {e}")

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
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            self.logger.warning(f"No existing checkpoints found, starting a new training run.")
            if self.module.config.last_global_step > 0:
                self.logger.warning(f"Last global step in module config is {self.module.config.last_global_step}, but no checkpoint found")
                # todo: set global_step/resume_step/first_epoch and override step counts in optimizer / lr scheduler
        else:
            self.global_step = int(path.split("-")[1])
            self.logger.info(f"Resuming from checkpoint {path} (global step: {self.global_step})")
            self.accelerator.load_state(os.path.join(self.config.model_path, path))

            # update any optimizer params that have changed
            updated_learn_rate = False; updated_adam_betas = False
            updated_weight_decay = False; updated_adam_eps = False
            for g in self.optimizer.param_groups:

                if g["initial_lr"] != self.config.lr_schedule.learning_rate:
                    g["initial_lr"] = self.config.lr_schedule.learning_rate
                    updated_learn_rate = True
                if g["betas"] != (self.config.optimizer.adam_beta1, self.config.optimizer.adam_beta2):
                    g["betas"] = (self.config.optimizer.adam_beta1, self.config.optimizer.adam_beta2)
                    updated_adam_betas = True
                if g["weight_decay"] != self.config.optimizer.adam_weight_decay:
                    g["weight_decay"] = self.config.optimizer.adam_weight_decay
                    updated_weight_decay = True
                if g["eps"] != self.config.optimizer.adam_epsilon:
                    g["eps"] = self.config.optimizer.adam_epsilon
                    updated_adam_eps = True
                
            if updated_learn_rate:
                self.lr_scheduler.scheduler.base_lrs = [self.config.lr_schedule.learning_rate]
                self.logger.info(f"Using updated learning rate: {self.config.lr_schedule.learning_rate}")
            if updated_adam_betas:
                self.logger.info(f"Using updated Adam beta1: {self.config.optimizer.adam_beta1} beta2: {self.config.optimizer.adam_beta2}")
            if updated_weight_decay:
                self.logger.info(f"Using updated Adam weight decay: {self.config.optimizer.adam_weight_decay}")
            if updated_adam_eps:
                self.logger.info(f"Using updated Adam epsilon: {self.config.optimizer.adam_epsilon}")

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
        self.logger.info(f"  Num examples = {len(self.dataset['train'])}")
        self.logger.info(f"  Instantaneous batch size per device = {self.config.device_batch_size}")
        self.logger.info(f"  Gradient accumulation steps = {self.config.gradient_accumulation_steps}")
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {self.total_batch_size}")
        self.logger.info(f"  Total optimization steps for full run = {self.config.max_train_steps}")
        self.logger.info(f"  Total samples processed so far = {self.persistent_state.total_samples_processed}")
        self.logger.info(f"  Path to save/load checkpoints = {self.config.model_path}")
        if not self.config.dataloader.use_pre_encoded_latents:
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
                    config.save_json(self.train_sample_logger.get_logs(sorted=True), sample_log_path)
                except Exception as e:
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
            switched_ema_name = self.ema_manager.switch_ema()
            if switched_ema_name is not None:
                self.logger.info(f"Loaded train weights from switch EMA: {switched_ema_name}")

            if self.global_step >= self.config.max_train_steps: break
            else: self.epoch += 1

        # hurray!
        self.logger.info(f"Reached max train steps ({self.config.max_train_steps}) - Training complete")
        self.accelerator.end_training()

    def run_train_epoch(self):

        train_logger = TrainLogger(self.accelerator) # accumulates per-batch logs / statistics
        progress_bar = tqdm(total=self.num_update_steps_per_epoch, disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {self.epoch}")

        if self.resume_dataloader is not None:
            progress_bar.update(self.resume_step)
            self.local_step = self.resume_step
        else:
            self.local_step = 0

        for local_batch in (self.resume_dataloader or self.train_dataloader):

            train_logger.clear() # accumulates per-batch logs / statistics
            self.module_trainer.init_batch()
            
            for self.accum_step in range(self.config.gradient_accumulation_steps):

                device_batch = { # get sub-batch of device batch size from local batch for each grad_accum_step
                    key: value[self.accum_step * self.config.device_batch_size: (self.accum_step+1) * self.config.device_batch_size]
                    for key, value in local_batch.items()
                }

                with self.accelerator.accumulate(self.module):

                    module_logs = self.module_trainer.train_batch(device_batch)
                    train_logger.add_logs(module_logs)
                    for i, sample_path in enumerate(device_batch["sample_paths"]):
                        self.train_sample_logger.add_log(sample_path, module_logs["loss"][i])

                    self.accelerator.backward(module_logs["loss"].sum())

                    if self.accelerator.sync_gradients:
                        assert self.accum_step == (self.config.gradient_accumulation_steps - 1), \
                            f"accum_step out of sync with sync_gradients: {self.accum_step} != {self.config.gradient_accumulation_steps - 1}"
                        
                        # clip grad norm and check for inf/nan grad
                        grad_norm = self.accelerator.clip_grad_norm_(self.module.parameters(), self.config.optimizer.max_grad_norm)
                        train_logger.add_log("grad_norm", grad_norm)

                        if math.isinf(grad_norm) or math.isnan(grad_norm):
                            self.logger.warning(f"Warning: grad norm is {grad_norm} step={self.global_step}")
                        if math.isnan(grad_norm):
                            self.logger.error(f"Error: grad norm is {grad_norm}, aborting...")
                            if self.accelerator.is_main_process:
                                import pdb; pdb.set_trace()
                            else:
                                exit(1)
                    else:
                        assert self.accum_step != (self.config.gradient_accumulation_steps - 1), \
                            f"accum_step out of sync, no sync_gradients but {self.accum_step} == {self.config.gradient_accumulation_steps - 1}"
                        
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

            # weights have now been updated in the last optimizer step
            if self.accelerator.sync_gradients:
                
                # update emas and normalize weights
                self.ema_manager.update()
                self.module.normalize_weights()
                
                # update logs
                train_logger.add_logs({"lr": self.lr_scheduler.get_last_lr()[0]})
                train_logger.add_logs(self.module_trainer.finish_batch())
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
            else:
                assert False, "finished local_batch but accelerator.sync_gradients isn't True"
            
            if self.global_step >= self.config.max_train_steps: break

        progress_bar.close()
        self.accelerator.wait_for_everyone()
        self.resume_dataloader = None # throw away resume dataloader (if any) at end of epoch
        
    @torch.no_grad()
    def run_validation(self):

        self.logger.info("***** Running validation *****")
        self.logger.info(f"  Epoch = {self.global_step // self.num_update_steps_per_epoch} (step: {self.global_step})")
        self.logger.info(f"  Num examples = {len(self.dataset['validation'])}")
        self.logger.info(f"  Total validation batch size (w. parallel, distributed & accumulation) = {self.validation_total_batch_size}")
        if not self.config.dataloader.use_pre_encoded_latents:
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
                config.save_json(self.validation_sample_logger.get_logs(sorted=True), sample_log_path)
            except Exception as e:
                self.logger.warning(f"Error saving validation sample logs to {sample_log_path}: {e}")

        return validation_logs