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
from datetime import datetime
from typing import Optional, Literal, Type, Union
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
from training.module_trainers.module_trainer import ModuleTrainerConfig
from .ema import PowerFunctionEMA
from .dataset import DatasetConfig, DualDiffusionDataset

class TrainLogger():

    def __init__(self, accelerator: Accelerator) -> None:

        self.accelerator = accelerator
        self.channels: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def clear(self) -> None:
        self.channels.clear()
        self.counts.clear()

    @torch.no_grad()
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

    def get_logs(self) -> dict:
        return {key: value / self.counts[key] for key, value in self.channels.items()}

@dataclass
class LRScheduleConfig:
    lr_schedule: Literal["edm2", "constant"] = "edm2"
    learning_rate: float     = 1e-2
    lr_warmup_steps: int     = 5000
    lr_reference_steps: int  = 5000
    lr_decay_exponent: float = 1.

@dataclass
class OptimizerConfig:
    adam_beta1: float        = 0.9
    adam_beta2: float        = 0.99
    adam_epsilon: float      = 1e-8
    adam_weight_decay: float = 0.
    max_grad_norm: float     = 10.
    add_grad_noise: float    = 0.

@dataclass
class EMAConfig:
    use_ema: bool           = False
    ema_stds: tuple[float]  = (0.02,)
    ema_cpu_offload: bool   = False

@dataclass
class DataLoaderConfig:
    use_pre_encoded_latents: bool = False
    filter_invalid_samples: bool = False
    dataset_num_proc: Optional[int] = None
    dataloader_num_workers: Optional[int] = None
    pin_memory: bool = False
    prefetch_factor: Optional[int] = None

@dataclass
class LoggingConfig:
    logging_dir: Optional[str] = None
    tensorboard_http_port:   Optional[int] = 6006
    tensorboard_num_scalars: Optional[int] = 2000

@dataclass
class DualDiffusionTrainerConfig:

    lr_schedule: LRScheduleConfig
    optimizer: OptimizerConfig
    ema: EMAConfig
    dataloader: DataLoaderConfig
    logging: LoggingConfig

    module_trainer_class: Type
    module_trainer_config: ModuleTrainerConfig

    module_name: str
    model_path: str
    model_name: str
    model_src_path: str
    train_config_path: Optional[str]    = None
    seed: Optional[int]                 = None
    device_batch_size: int               = 1
    gradient_accumulation_steps: int    = 1

    num_train_epochs: int               = 500000
    num_validation_epochs: int          = 10
    min_checkpoint_time: int            = 3600
    checkpoints_total_limit: int        = 1
    strict_checkpoint_time: bool        = False

    enable_anomaly_detection: bool      = False
    enable_model_compilation: bool      = True
    compile_params: Optional[dict]      = None

    @staticmethod
    def from_json(json_path, **kwargs) -> "DualDiffusionTrainerConfig":

        train_config = config.load_json(json_path)
        train_config["train_config_path"] = json_path

        for key, value in kwargs.items():
            train_config[key] = value
        
        train_config["lr_schedule"] = LRScheduleConfig(**train_config["lr_schedule"])
        train_config["optimizer"] = OptimizerConfig(**train_config["optimizer"])
        train_config["ema"] = EMAConfig(**train_config["ema"])
        train_config["dataloader"] = DataLoaderConfig(**train_config["dataloader"])
        train_config["logging"] = LoggingConfig(**train_config["logging"])

        module_trainer_module = importlib.import_module(train_config["module_trainer_class"][0])
        train_config["module_trainer_class"] = getattr(module_trainer_module, train_config["module_trainer_class"][1])
        train_config["module_trainer_config"] = train_config["module_trainer_class"].get_config_class()(**train_config["module_trainer_config"])

        return DualDiffusionTrainerConfig(**train_config)

class DualDiffusionTrainer:

    def __init__(self, train_config: DualDiffusionTrainerConfig) -> None:

        self.config = train_config

        self.init_accelerator()
        self.init_tensorboard()
        self.init_pytorch()
        self.init_module_pipeline()
        self.init_ema_module()
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
        self.module = getattr(self.pipeline, self.config.module_name).requires_grad_(True).train()
        self.module_class = type(self.module)

        self.sample_shape = self.pipeline.format.get_sample_shape(bsz=self.config.device_batch_size)
        if hasattr(self.pipeline, "vae"):
            self.latent_shape = self.pipeline.vae.get_latent_shape(self.sample_shape)
        else:
            self.latent_shape = None

        self.logger.info(f"Module class: {self.module_class.__name__}")
        self.logger.info(f"Module trainer class: {self.config.module_trainer_class.__name__}")

        self.module = self.accelerator.prepare(self.module)

        if hasattr(self.module, "normalize_weights"):
            self.module.normalize_weights()
        
    def init_ema_module(self) -> None:
        
        if self.config.ema.use_ema:
            ema_device = "cpu" if self.config.ema.ema_cpu_offload else self.accelerator.device
            self.ema_module = PowerFunctionEMA(self.module, stds=self.config.ema.ema_stds, device=self.accelerator.device)

            self.logger.info(f"Using EMA model with stds: {self.config.ema.ema_stds}")
            self.logger.info(f"  EMA CPU offloading {'enabled' if ema_device == 'cpu' else 'disabled'}")
        else:
            self.logger.info("Not using EMA model")

    def init_optimizer(self) -> None:
        
        self.optimizer = torch.optim.AdamW(
            self.module.parameters(),
            lr=self.config.lr_schedule.learning_rate,
            betas=(self.config.optimizer.adam_beta1, self.config.optimizer.adam_beta2),
            weight_decay=self.config.optimizer.adam_weight_decay,
            eps=self.config.optimizer.adam_epsilon,
            foreach=True,
            #fused=True,
        )

        self.logger.info(f"Using AdamW optimiser with learning rate {self.config.lr_schedule.learning_rate}")
        self.logger.info(f"  AdamW beta1: {self.config.optimizer.adam_beta1} beta2: {self.config.optimizer.adam_beta2}")
        self.logger.info(f"  AdamW eps: {self.config.optimizer.adam_epsilon} weight decay: {self.config.optimizer.adam_weight_decay}")

    def init_checkpointing(self) -> None:

        self.last_checkpoint_time = datetime.now()
        self.logger.info(f"Saving checkpoints every {self.config.min_checkpoint_time}s")

        def save_model_hook(models, weights, output_dir):
            if self.accelerator.is_main_process:
                assert len(models) == 1
                model = models.pop()
                model.save_pretrained(os.path.join(output_dir, self.config.module_name))
                weights.pop() # not sure if this is needed

                if self.ema_module is not None:
                    self.ema_module.save(os.path.join(output_dir, f"{self.config.module_name}_ema"))

        def load_model_hook(models, input_dir):
            assert len(models) == 1
            model = models.pop()

            # copy loaded state and config into model
            load_model = self.module_class.from_pretrained(input_dir, subfolder=self.config.module_name)
            model.config = load_model.config
            model.load_state_dict(load_model.state_dict())
            del load_model
            
            if self.ema_module is not None: # load / create EMA models
                ema_model_dir = os.path.join(input_dir, f"{self.config.module_name}_ema")
                ema_load_errors = self.ema_module.load(ema_model_dir, target_module=model)
                if len(ema_load_errors) > 0:
                    self.logger.warning(f"Errors loading EMA model(s) - Missing EMA(s) initialized from checkpoint model:")
                    self.logger.warning("\n".join(ema_load_errors))
                else:
                    self.logger.info(f"Successfully loaded EMA model(s) from {ema_model_dir}")

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)
        self.logger.info("Registered accelerator hooks for model load/save")

        # make temporary copy of source code, model_index and train config to be copied to saved checkpoints
        if self.accelerator.is_main_process:
            
            tmp_path = os.path.join(self.config.model_path, "tmp")
            if os.path.isdir(tmp_path):
                shutil.rmtree(tmp_path)
            os.makedirs(tmp_path)

            shutil.copytree(self.config.model_src_path,
                            os.path.join(tmp_path, "src"),
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

            def lr_schedule(current_step: int):
                lr = 1.
                if current_step < scaled_lr_warmup_steps:
                    lr *= current_step / scaled_lr_warmup_steps
                if current_step > scaled_lr_reference_steps:
                    lr *= (scaled_lr_reference_steps / current_step) ** self.config.lr_schedule.lr_decay_exponent
                return lr
            
        elif self.config.lr_schedule.lr_schedule == "constant":

            def lr_schedule(current_step: int):
                if current_step < scaled_lr_warmup_steps:
                    return current_step / scaled_lr_warmup_steps
                return 1.

        else:
            raise ValueError(f"Unsupported learning rate schedule: {self.config.lr_schedule.lr_schedule}")
        
        self.lr_scheduler = LambdaLR(self.optimizer, lr_schedule)
   
    def init_dataloader(self) -> None:
        
        self.local_batch_size = self.config.device_batch_size * self.config.gradient_accumulation_steps
        self.total_batch_size = self.local_batch_size * self.accelerator.num_processes
        self.validation_total_batch_size = self.config.device_batch_size * self.accelerator.num_processes

        latents_crop_width = self.latent_shape[-1] if self.latent_shape is not None else 0
        sample_raw_crop_width = self.pipeline.format.sample_raw_crop_width()
        
        dataset_config = DatasetConfig(
            data_dir=config.DATASET_PATH,
            cache_dir=config.CACHE_PATH,
            sample_rate=self.pipeline.format.config.sample_rate,
            sample_raw_crop_width=sample_raw_crop_width,
            sample_raw_channels=self.pipeline.format.config.sample_raw_channels,
            use_pre_encoded_latents=self.config.dataloader.use_pre_encoded_latents,
            latents_crop_width=latents_crop_width,
            num_proc=self.config.dataloader.dataset_num_proc,
            t_scale=getattr(self.pipeline.unet.config, "t_scale", None) if self.config.module_name == "unet" else None,
            filter_invalid_samples=self.config.dataloader.filter_invalid_samples,
        )
        self.dataset = DualDiffusionDataset(dataset_config)

        self.train_dataloader = torch.utils.data.DataLoader(
            self.dataset["train"],
            shuffle=True,
            batch_size=self.local_batch_size,
            num_workers=self.config.dataloader.dataloader_num_workers or 0,
            pin_memory=self.config.dataloader.pin_memory,
            persistent_workers=True if self.config.dataloader.dataloader_num_workers else False,
            prefetch_factor=self.config.dataloader.prefetch_factor if self.config.dataloader.dataloader_num_workers else None,
            drop_last=True,
        )
        self.validation_dataloader = torch.utils.data.DataLoader(
            self.dataset["validation"],
            batch_size=1,
            drop_last=False,
        )

        self.logger.info(f"Using dataset path {config.DATASET_PATH} with {dataset_config.num_proc or 1} dataset processes)")
        self.logger.info(f"  {len(self.dataset['train'])} train samples ({self.dataset.num_filtered_samples['train']} filtered)")
        self.logger.info(f"  {len(self.dataset['validation'])} validation samples ({self.dataset.num_filtered_samples['validation']} filtered)")
        self.logger.info(f"Using train dataloader with {self.config.dataloader.dataloader_num_workers or 0} workers")
        self.logger.info(f"  prefetch_factor = {self.config.dataloader.prefetch_factor}")
        self.logger.info(f"  pin_memory = {self.config.dataloader.pin_memory}")

        self.num_update_steps_per_epoch = len(self.train_dataloader) // self.accelerator.num_processes
        self.max_train_steps = self.num_update_steps_per_epoch * self.config.num_train_epochs

    def init_torch_compile(self) -> None:

        if self.config.enable_model_compilation:
            if platform.system() == "Linux":
                self.config.compile_params = self.config.compile_params or {"fullgraph": True}
                self.logger.info(f"Compiling model(s) with options: {dict_str(self.config.compile_params)}")
            else:
                self.config.enable_model_compilation = False
                self.logger.warning("PyTorch model compilation is currently only supported on Linux - skipping compilation")
        else:
            self.logger.info("PyTorch model compilation is disabled")

    def save_checkpoint(self, global_step) -> None:
        
        # save model checkpoint and training / optimizer state
        self.module.config.last_global_step = global_step
        save_path = os.path.join(self.config.model_path, f"{self.config.module_name}_checkpoint-{global_step}")
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

        global_step = 0
        resume_step = 0
        first_epoch = 0
        
        # get lasted checkpoint path
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
            global_step = int(path.split("-")[1])
            self.logger.info(f"Resuming from checkpoint {path} (global step: {global_step})")
            self.accelerator.load_state(os.path.join(self.config.model_path, path))

            # update learning rate in case we've changed it, todo: update other optimizer params as well
            updated_learn_rate = False
            for g in self.optimizer.param_groups:
                if g["lr"] != self.config.lr_schedule.learning_rate:
                    g["lr"] = self.config.lr_schedule.learning_rate
                    updated_learn_rate = True
            if updated_learn_rate:
                self.lr_scheduler.scheduler.base_lrs = [self.config.lr_schedule.learning_rate]
                self.logger.info(f"Using updated learning rate: {self.config.lr_schedule.learning_rate}")

        if global_step > 0:
            first_epoch = global_step // self.num_update_steps_per_epoch
            resume_step = global_step % self.num_update_steps_per_epoch

        return global_step, resume_step, first_epoch

    def train(self) -> None:

        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num examples = {len(self.dataset['train'])}")
        self.logger.info(f"  Num Epochs = {self.config.num_train_epochs}")
        self.logger.info(f"  Instantaneous batch size per device = {self.config.device_batch_size}")
        self.logger.info(f"  Gradient Accumulation steps = {self.config.gradient_accumulation_steps}")
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {self.total_batch_size}")
        self.logger.info(f"  Total optimization steps for full run = {self.max_train_steps}")
        self.logger.info(f"  Path to save/load checkpoints = {self.config.model_path}")
        if not self.config.dataloader.use_pre_encoded_latents:
            self.logger.info(f"  Sample shape: {self.sample_shape}")
        if self.latent_shape is not None:
            self.logger.info(f"  Latent shape: {self.latent_shape}")

        global_step, resume_step, first_epoch = self.load_checkpoint()
        resume_dataloader = self.accelerator.skip_first_batches(self.train_dataloader, resume_step) if resume_step > 0 else None
        self.module_trainer = self.config.module_trainer_class(self.config.module_trainer_config, self)

        train_logger = TrainLogger(self.accelerator)
        sample_logger = TrainLogger(self.accelerator)

        for epoch in range(first_epoch, self.config.num_train_epochs):
            
            progress_bar = tqdm(total=self.num_update_steps_per_epoch, disable=not self.accelerator.is_local_main_process)
            if resume_dataloader is not None: progress_bar.update(resume_step)
            progress_bar.set_description(f"Epoch {epoch}")

            for local_batch in (resume_dataloader or self.train_dataloader):

                train_logger.clear()
                self.module_trainer.init_batch()
                
                for grad_accum_steps in range(self.config.gradient_accumulation_steps):

                    device_batch = { # get sub-batch of device batch size from local batch for each grad_accum_step
                        key: value[grad_accum_steps * self.config.device_batch_size: (grad_accum_steps+1) * self.config.device_batch_size]
                        for key, value in local_batch.items()
                    }

                    with self.accelerator.accumulate(self.module):

                        module_logs = self.module_trainer.train_batch(device_batch, grad_accum_steps)
                        train_logger.add_logs(module_logs)
                        for i, sample_path in enumerate(device_batch["sample_paths"]):
                            sample_logger.add_log(sample_path, module_logs["loss"][i])

                        self.accelerator.backward(module_logs["loss"].mean())

                        if self.accelerator.sync_gradients:
                            assert grad_accum_steps == (self.config.gradient_accumulation_steps - 1), \
                                f"grad_accum_steps out of sync with sync_gradients - {grad_accum_steps} != {self.config.gradient_accumulation_steps - 1}"
                            
                            grad_norm = self.accelerator.clip_grad_norm_(self.module.parameters(), self.config.optimizer.max_grad_norm)
                            train_logger.add_log("grad_norm", grad_norm)

                            if math.isinf(grad_norm) or math.isnan(grad_norm):
                                self.logger.warning(f"Warning: grad norm is {grad_norm} step={global_step}")
                            if math.isnan(grad_norm):
                                self.logger.error(f"Error: grad norm is {grad_norm}, aborting...")
                                import pdb; pdb.set_trace()

                            if self.config.optimizer.add_grad_noise > 0:
                                for p in self.module.parameters():
                                    p_noise = torch.randn_like(p) * (torch.linalg.vector_norm(p.grad) / p.grad.numel() ** 0.5)
                                    p.grad += p_noise * self.config.optimizer.add_grad_noise * self.lr_scheduler.get_last_lr()[0]
                        else:
                            assert grad_accum_steps != (self.config.gradient_accumulation_steps - 1), \
                                f"grad_accum_steps out of sync, no sync_gradients but {grad_accum_steps} == {self.config.gradient_accumulation_steps - 1}"
                            
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                
                    if hasattr(self.module, "normalize_weights"):
                        self.module.normalize_weights()
                        
                    train_logger.add_logs({"lr": self.lr_scheduler.get_last_lr()[0], "step": global_step})
                    progress_bar.update(1)
                    global_step += 1
                    
                    if self.config.ema.use_ema:
                        std_betas = self.ema_module.update(global_step * self.total_batch_size, self.total_batch_size)
                        for std, beta in std_betas:
                            train_logger.add_log(f"ema/std_{std:.3f}_beta", beta)

                    train_logger.add_logs(self.module_trainer.finish_batch())

                    logs = train_logger.get_logs()
                    self.accelerator.log(logs, step=global_step)
                    progress_bar.set_postfix(loss=logs["loss"], grad_norm=logs["grad_norm"], global_step=global_step)
                    
                    if self.accelerator.is_main_process:
                        _save_checkpoint = False
                        if self.config.strict_checkpoint_time:
                            if (datetime.now() - self.last_checkpoint_time).total_seconds() >= self.config.min_checkpoint_time:
                                _save_checkpoint = True
                        
                        _save_checkpoint_path = os.path.join(self.config.model_path, "_save_checkpoint")
                        if os.path.isfile(_save_checkpoint_path): _save_checkpoint = True
                        if _save_checkpoint: self.save_checkpoint(global_step)
                        if os.path.isfile(_save_checkpoint_path): os.remove(_save_checkpoint_path)
                else:
                    assert False, "finished local_batch but accelerator.sync_gradients isn't True"
                
                if global_step >= self.max_train_steps:
                    self.logger.info(f"Reached max train steps ({self.max_train_steps})")
                    break
            
            progress_bar.close()
            self.accelerator.wait_for_everyone()
            resume_dataloader = None
            
            if self.accelerator.is_main_process:
                
                sample_log_path = os.path.join(self.config.model_path, "tmp", "sample_loss.json")
                try:
                    sorted_sample_logs = dict(sorted(sample_logger.get_logs().items(), key=lambda item: item[1]))
                    config.save_json(sorted_sample_logs, sample_log_path)
                except Exception as e:
                    self.logger.warning(f"Error saving sample logs to {sample_log_path}: {e}")

                if not self.config.strict_checkpoint_time:
                    if (datetime.now() - self.last_checkpoint_time).total_seconds() >= self.config.min_checkpoint_time:
                        self.save_checkpoint(global_step)

            self.accelerator.wait_for_everyone()
            if self.config.num_validation_epochs > 0:
                if epoch % self.config.num_validation_epochs == 0:
                    self.run_validation(global_step)
        
        self.logger.info("Training complete")
        self.accelerator.end_training()

    @torch.no_grad()
    def run_validation(self, global_step):

        self.logger.info("***** Running validation *****")
        self.logger.info(f"  Num examples = {len(self.dataset['validation'])}")

        start_validation_time = datetime.now()

        validation_logger = TrainLogger(self.accelerator)
        sample_logger = TrainLogger(self.accelerator)

        progress_bar = tqdm(total=len(self.validation_dataloader), disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description(f"Validation")

        for _, batch in enumerate(self.validation_dataloader):
            
            validation_batch = {} # expand each individual validation sample to device_batch_size
            for key, value in batch.items():
                if torch.is_tensor(value):
                    validation_batch[key] = value.repeat((self.config.device_batch_size,) + (1,) * (value.ndim - 1))
                elif isinstance(value, list):
                    validation_batch[key] = value * self.config.device_batch_size
                else:
                    raise ValueError(f"Unsupported validation batch value type: {type(value)} '{key}': '{value}'")

            self.module_trainer.init_batch(total_batch_size=self.validation_total_batch_size, validation=True)
            module_logs = self.module_trainer.train_batch(validation_batch, 0)
            validation_logger.add_logs(module_logs)
            for i, sample_path in enumerate(validation_batch["sample_paths"]):
                sample_logger.add_log(sample_path, module_logs["loss"][i])

            validation_logger.add_logs(self.module_trainer.finish_batch())
            progress_bar.update(1)
        
        progress_bar.close()

        validation_logs = {}
        for key, value in validation_logger.get_logs().items():
            key = key.replace("/", "_validation/", 1) if "/" in key else key + "_validation"
            validation_logs[key] = value
        self.accelerator.log(validation_logs, step=global_step)

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:                   
            sample_log_path = os.path.join(self.config.model_path, "tmp", "sample_loss_validation.json")
            try:
                sorted_sample_logs = dict(sorted(sample_logger.get_logs().items(), key=lambda item: item[1]))
                config.save_json(sorted_sample_logs, sample_log_path)
            except Exception as e:
                self.logger.warning(f"Error saving validation sample logs to {sample_log_path}: {e}")

        self.logger.info(f"Validation complete (runtime: {(datetime.now() - start_validation_time).total_seconds()}s)")