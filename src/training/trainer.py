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
from typing import Optional, Literal, Type
from dataclasses import dataclass

import datasets
import torch
import torch.utils.checkpoint
from torch.optim.lr_scheduler import LambdaLR
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm

import diffusers
from diffusers.optimization import get_scheduler
from diffusers.utils import is_tensorboard_available

from pipelines.dual_diffusion_pipeline import DualDiffusionPipeline
from utils.dual_diffusion_utils import dict_str
from training.module_trainers.module_trainer import ModuleTrainerConfig
from .ema import PowerFunctionEMA
from .dataset import DatasetConfig, DualDiffusionDataset

class TrainLogger():

    def __init__(self, accelerator: Accelerator) -> None:

        self.accelerator = accelerator
        self.channels, self.counts = {}, {}

    def clear(self) -> None:
        self.channels.clear()
        self.counts.clear()

    @torch.no_grad()
    def add_log(self, key, value) -> None:
        if torch.is_tensor(value):
            value = self.accelerator.gather(value.detach()).mean().item()

        if key in self.channels:
            self.channels[key] += value
            self.counts[key] += 1
        else:
            self.channels[key] = value
            self.counts[key] = 1

    def add_logs(self, logs) -> None:
        for key, value in logs:
            self.add_log(key, value)

    def get_logs(self) -> dict:
        return {key: value / self.counts[key] for key, value in self.items()}

@dataclass
class LRScheduleConfig:
    lr_schedule: Literal["edm2", "linear", "cosine", "cosine_with_restarts",
                         "polynomial", "constant", "constant_with_warmup"] = "edm2"
    learning_rate: float     = 1e-2
    lr_warmup_steps: int     = 5000
    lr_reference_steps: int  = 5000
    lr_decay_exponent: float = 1.

@dataclass
class OptimizerConfig:
    adam_beta1: float       = 0.9
    adam_beta2: float       = 0.99
    adam_epsilon: float     = 1e-8
    max_grad_norm: float    = 10.
    add_grad_noise: float   = 0.

@dataclass
class EMAConfig:
    use_ema: bool           = False
    ema_stds: tuple[float]  = (0.01,)
    ema_cpu_offload: bool   = False

@dataclass
class DataLoaderConfig:
    use_pre_encoded_latents: bool = False
    dataloader_num_workers: int   = 0

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
    train_batch_size: int               = 1
    gradient_accumulation_steps: int    = 1

    num_train_epochs: int               = 500000
    num_validation_epochs: int          = 100
    min_validation_time: int            = 3600
    min_checkpoint_time: int            = 3600
    checkpoints_total_limit: int        = 1
    strict_checkpoint_time: bool        = False

    enable_anomaly_detection: bool      = False
    compile_full_graph: bool            = True

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

        self.init_logging()
        self.init_accelerator()
        self.init_tensorboard()
        self.init_pytorch()
        self.init_module_pipeline()
        self.init_ema_module()
        self.init_optimizer()
        self.init_checkpointing()
        self.init_lr_scheduler()
        self.init_dataloader()

        self.module, self.optimizer, self.train_dataloader, self.validation_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.module, self.optimizer, self.train_dataloader, self.validation_dataloader, self.lr_scheduler
        )

    def init_logging(self) -> None:
        
        self.logger = get_logger("dual_diffusion_training", log_level="INFO")

        if self.config.logging.logging_dir is None:
            self.config.logging.logging_dir = os.path.join(self.config.model_path, f"logs_{self.config.module_name}")
            
        os.makedirs(self.config.logging.logging_dir, exist_ok=True)

        log_path = os.path.join(self.config.logging.logging_dir, f"train_{self.config.module_name}.log")
        logging.basicConfig(
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ],
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        
        self.logger.info(f"Logging to {log_path}")

        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()

    def init_accelerator(self) -> None:

        accelerator_project_config = ProjectConfiguration(project_dir=self.config.model_path,
                                                          logging_dir=self.config.logging.logging_dir)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            log_with="tensorboard",
            project_config=accelerator_project_config,
        )

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

        if not hasattr(self.pipeline, self.config.module_name):
            raise ValueError(f"Module type '{self.config.module_name}' not registered in loaded pipeline")
        
        self.pipeline = DualDiffusionPipeline.from_pretrained(self.config.model_path)
        self.module = getattr(self.pipeline, self.config.module_name).requires_grad_(True).train()
        self.module_class = type(self.module)

        self.vae = self.pipeline.vae

        if self.config.module_name == "unet":
            if not self.config.dataloader.use_pre_encoded_latents:

                self.vae = self.vae.to(self.accelerator.device).requires_grad_(False).eval()
                if self.accelerator.state.mixed_precision in ["fp16", "bf16"]:
                    self.vae = self.vae.to(self.accelerator.state.mixed_precision)

                self.logger.info(f"Training diffusion model with VAE")
            else:
                self.logger.info(f"Training diffusion model with pre-encoded latents")

        elif self.config.module_name == "vae":
            self.pipeline.unet = self.pipeline.unet.to("cpu")
       
        self.pipeline.format = self.pipeline.format.to(self.accelerator.device)
        self.sample_shape = self.pipeline.format.get_sample_shape(bsz=self.config.train_batch_size)
        self.latent_shape = self.vae.get_latent_shape(self.sample_shape)

        self.logger.info(f"Module class: {self.module_class.__name__}")
        self.logger.info(f"Module trainer class: {self.config.module_trainer_class.__name__}")
        
        if self.config.compile_params is not None and platform.system() == "Linux":
            self.logger.info(f"Compiling model with options: {dict_str(self.config.compile_params)}")
            self.module.forward = torch.compile(self.module.forward, **self.config.compile_params)

    def init_ema_module(self) -> None:
        
        if self.config.ema.use_ema:
            ema_device = "cpu" if self.config.ema.ema_cpu_offload else self.accelerator.device
            self.ema_module = PowerFunctionEMA(self.module, stds=self.config.ema.ema_stds, device=self.accelerator.device)

            self.logger.info(f"Using EMA model with stds: {self.config.ema.ema_stds}")
            self.logger.info(f"EMA CPU offloading {'enabled' if ema_device == 'cpu' else 'disabled'}")

        else:
            self.logger.info("Not using EMA model")

    def init_optimizer(self) -> None:
        
        optimizer = torch.optim.AdamW(
            self.module.parameters(),
            lr=self.config.lr_schedule.learning_rate,
            betas=(self.config.optimizer.adam_beta1, self.config.optimizer.adam_beta2),
            weight_decay=0,
            eps=self.config.optimizer.adam_epsilon,
            foreach=True,
            fused=True,
        )

        self.logger.info(f"Using Adam optimiser with learning rate {self.config.lr_schedule.learning_rate}")
        self.logger.info(f"AdamW beta1: {self.config.optimizer.adam_beta1} beta2: {self.config.optimizer.adam_beta2} eps: {self.config.optimizer.adam_epsilon}")

        return optimizer

    def init_checkpointing(self) -> None:

        self.last_checkpoint_time = datetime.now()
        self.logger.info(f"Saving checkpoints every {self.config.min_checkpoint_time}s")

        def save_model_hook(models, weights, output_dir):
            if self.accelerator.is_main_process:
                for model in models:
                    model.save_pretrained(os.path.join(output_dir, self.config.module_name))
                    weights.pop() # make sure to pop weight so that corresponding model is not saved again

                if self.ema_module is not None:
                    self.ema_module.save(os.path.join(output_dir, f"{self.config.module_name}_ema"))

        def load_model_hook(models, input_dir):
            for _ in range(len(models)):
                model = models.pop() # pop models so that they are not loaded again

                # copy loaded state and config into model
                load_model = self.module_class.from_pretrained(input_dir, subfolder=self.config.module_name)
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model
            
            if self.ema_module is not None:
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

        if self.accelerator.is_main_process: # make temporary copy of source code, model_index and train config to be copied to saved checkpoints
            
            tmp_path = os.path.join(self.config.model_path, "tmp")
            if os.path.exists(tmp_path) and os.path.isdir(tmp_path):
                shutil.rmtree(tmp_path)
            os.makedirs(tmp_path)

            shutil.copytree(self.config.model_src_path,
                            os.path.join(tmp_path, "src"),
                            ignore=shutil.ignore_patterns("*.pyc", "__pycache__"),
                            dirs_exist_ok=True)
            shutil.copy(os.path.join(self.config.model_path, "model_index.json"), tmp_path)
            if self.config.train_config_path is not None:
                shutil.copy(self.config.train_config_path, tmp_path)

    def init_dataloader(self) -> None:
        
        if self.config.module_name == "unet" and self.config.dataloader.use_pre_encoded_latents:
            data_dir = config.LATENTS_DATASET_PATH
            sample_crop_width = self.latent_shape[-1]
        else:
            data_dir = config.DATASET_PATH
            sample_crop_width = self.pipeline.format.get_sample_crop_width()
        
        dataset_config = DatasetConfig(
            data_dir=data_dir,
            cache_dir=config.CACHE_PATH,
            num_proc=self.config.dataloader.dataloader_num_workers if self.config.dataloader.dataloader_num_workers > 0 else None,
            sample_crop_width=sample_crop_width,
            use_pre_encoded_latents=self.config.dataloader.use_pre_encoded_latents,
            t_scale=getattr(self.pipeline.unet.config, "t_scale", None) if self.config.module_name == "unet" else None,
        )
        
        self.dataset = DualDiffusionDataset(dataset_config)

        self.train_dataloader = torch.utils.data.DataLoader(
            self.dataset["train"],
            shuffle=True,
            batch_size=self.config.train_batch_size,
            num_workers=self.config.dataloader.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=True if self.config.dataloader.dataloader_num_workers > 0 else False,
            prefetch_factor=2 if self.config.dataloader.dataloader_num_workers > 0 else None,
            drop_last=True,
        )

        self.validation_dataloader = torch.utils.data.DataLoader(
            self.dataset["validation"],
            batch_size=self.config.train_batch_size,
            drop_last=False,
        )

        self.logger.info(f"Using dataset path {data_dir}")
        self.logger.info(f"{len(self.dataset["train"])} train samples ({self.dataset.num_filtered_samples["train"]} filtered)")
        self.logger.info(f"{len(self.dataset["validation"])} validation samples ({self.dataset.num_filtered_samples["validation"]} filtered)")
        if self.config.dataloader.dataloader_num_workers > 0:
            self.logger.info(f"Using train dataloader with {self.config.dataloader.dataloader_num_workers} workers - prefetch factor = 2")

        num_process_steps_per_epoch = math.floor(len(self.train_dataloader) / self.accelerator.num_processes)
        self.num_update_steps_per_epoch = math.ceil(num_process_steps_per_epoch / self.config.gradient_accumulation_steps)
        self.max_train_steps = self.config.num_train_epochs * self.num_update_steps_per_epoch
        self.total_batch_size = self.config.train_batch_size * self.accelerator.num_processes * self.config.gradient_accumulation_steps
 
    def init_lr_scheduler(self) -> None:

        if self.config.lr_schedule.lr_schedule == "edm2":
            self.logger.info((
                f"Using learning rate schedule {self.config.lr_schedule.lr_schedule}",
                f" with warmup steps = {self.config.lr_schedule.lr_warmup_steps},",
                f" reference steps = {self.config.lr_schedule.lr_reference_steps},",
                f" decay exponent = {self.config.lr_schedule.lr_decay_exponent}"))
        else:
            self.logger.info((f"Using learning rate schedule {self.config.lr_schedule.lr_schedule}",
                              f"with warmup steps = {self.config.lr_schedule.lr_warmup_steps}"))
        
        scaled_lr_warmup_steps = self.config.lr_schedule.lr_warmup_steps * self.accelerator.num_processes
        scaled_lr_reference_steps = self.config.lr_schedule.lr_reference_steps * self.accelerator.num_processes
        scaled_max_train_steps = self.max_train_steps * self.accelerator.num_processes

        if self.config.lr_schedule.lr_schedule == "edm2":
            def edm2_lr_lambda(current_step: int):
                lr = 1.
                if current_step < scaled_lr_warmup_steps:
                    lr *= current_step / scaled_lr_warmup_steps
                if current_step > scaled_lr_reference_steps:
                    lr *= (scaled_lr_reference_steps / current_step) ** self.config.lr_schedule.lr_decay_exponent
                return lr
                
            self.lr_scheduler = LambdaLR(self.optimizer, edm2_lr_lambda)
        else:
            self.lr_scheduler = get_scheduler(
                self.config.lr_schedule.lr_schedule,
                optimizer=self.optimizer,
                num_warmup_steps=scaled_lr_warmup_steps,
                num_training_steps=scaled_max_train_steps,
            )
   
    def save_checkpoint(self, global_step) -> None:
        
        # save model checkpoint and training / optimizer state
        self.module.config["last_global_step"] = global_step
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
              
        dirs = os.listdir(self.config.model_path)
        dirs = [d for d in dirs if d.startswith(f"{self.config.module_name}_checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            self.logger.warning(f"No existing checkpoints found, starting a new training run.")
        else:
            global_step = int(path.split("-")[1])
            self.logger.info(f"Resuming from checkpoint {path} (global step: {global_step})")
            self.accelerator.load_state(os.path.join(self.config.model_path, path))

            # update learning rate in case we've changed it
            updated_learn_rate = False
            for g in self.optimizer.param_groups:
                if g["lr"] != self.config.lr_schedule.learning_rate:
                    g["lr"] = self.config.lr_schedule.learning_rate
                    updated_learn_rate = True
            if updated_learn_rate:
                self.lr_scheduler.scheduler.base_lrs = [self.config.lr_schedule.learning_rate]
                self.logger.info(f"Using updated learning rate: {self.config.lr_schedule.learning_rate}")

        if global_step > 0:
            resume_global_step = global_step * self.config.gradient_accumulation_steps
            first_epoch = global_step // self.num_update_steps_per_epoch
            resume_step = resume_global_step % (self.num_update_steps_per_epoch * self.config.gradient_accumulation_steps)

        return global_step, resume_step, first_epoch

    def train(self) -> None:

        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num examples = {len(self.dataset["train"])}")
        self.logger.info(f"  Num Epochs = {self.config.num_train_epochs}")
        self.logger.info(f"  Instantaneous batch size per device = {self.config.train_batch_size}")
        self.logger.info(f"  Gradient Accumulation steps = {self.config.gradient_accumulation_steps}")
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {self.total_batch_size}")
        self.logger.info(f"  Total optimization steps for full run = {self.max_train_steps}")
        self.logger.info(f"  Path to save/load checkpoints = {self.config.model_path}")

        self.last_validation_time = datetime.now()
        global_step, resume_step, first_epoch = self.load_checkpoint()
        resume_dataloader = self.accelerator.skip_first_batches(self.train_dataloader, resume_step) if resume_step > 0 else None

        self.module_trainer = self.config.module_trainer_class(self.config.module_trainer_config, self)
        train_logger = TrainLogger()
        sample_logger = TrainLogger()

        if not self.config.dataloader.use_pre_encoded_latents:
            self.logger.info(f"Sample shape: {self.sample_shape}")
        self.logger.info(f"Latent shape: {self.latent_shape}")

        if hasattr(self.module, "normalize_weights"):
            self.module.normalize_weights()
            
        for epoch in range(first_epoch, self.config.num_train_epochs):
            
            progress_bar = tqdm(total=self.num_update_steps_per_epoch, disable=not self.accelerator.is_local_main_process)
            if resume_step > 0: progress_bar.update(resume_step // self.config.gradient_accumulation_steps)
            progress_bar.set_description(f"Epoch {epoch}")

            grad_accum_steps = 0

            for batch in (resume_dataloader or self.train_dataloader):
                
                if grad_accum_steps == 0:                        
                    train_logger.clear()
                    self.module_trainer.init_batch()

                with self.accelerator.accumulate(self.module):

                    module_logs = self.module_trainer.train_batch(batch, grad_accum_steps)
                    train_logger.add_logs(module_logs)
                    for i, sample_path in enumerate(batch["sample_paths"]):
                        sample_logger.add_log(sample_path, module_logs["loss"][i])

                    grad_accum_steps += 1
                    self.accelerator.backward(module_logs["loss"].mean())

                    if self.accelerator.sync_gradients:   
                        grad_norm = self.accelerator.clip_grad_norm_(self.module.parameters(),
                                                                     self.config.optimizer.max_grad_norm)
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

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                
                    if hasattr(self.module, "normalize_weights"):
                        self.module.normalize_weights()
                        
                    train_logger.add_logs({"lr": self.lr_scheduler.get_last_lr()[0], "step": global_step})
                    progress_bar.update(1)
                    global_step += 1
                    grad_accum_steps = 0
                    
                    if self.config.ema.use_ema:
                        std_betas = self.ema_module.update(global_step * self.total_batch_size, self.total_batch_size)
                        for std, beta in std_betas:
                            train_logger.add_log(f"ema/std_{std:.3f}_beta", beta)

                    train_logger.add_logs(self.module_trainer.finish_batch())

                    logs = train_logger.get_logs()
                    self.accelerator.log(logs, step=global_step)
                    progress_bar.set_postfix(loss=logs["loss"], grad_norm=logs["grad_norm"], global_step=global_step)
                        
                    if self.accelerator.is_main_process:
                        if self.config.strict_checkpoint_time:
                            if (datetime.now() - self.last_checkpoint_time).total_seconds() >= self.config.min_checkpoint_time:
                                _save_checkpoint = True
                        
                        _save_checkpoint_path = os.path.join(self.config.model_path, "_save_checkpoint")
                        if os.path.exists(_save_checkpoint): _save_checkpoint = True
                        if _save_checkpoint: self.save_checkpoint(global_step)
                        if os.path.exists(_save_checkpoint_path): os.remove(_save_checkpoint_path)

                if global_step >= self.max_train_steps:
                    self.logger.info(f"Reached max train steps ({self.max_train_steps})")
                    break
            
            progress_bar.close()
            resume_dataloader = None

            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                
                if not self.config.strict_checkpoint_time:
                    if (datetime.now() - self.last_checkpoint_time).total_seconds() >= self.config.min_checkpoint_time:
                        self.save_checkpoint(global_step)
                        
                sample_log_path = os.path.join(self.config.model_path, "tmp", "sample_loss.json")
                try:
                    sorted_sample_logs = dict(sorted(sample_logger.get_logs().items(), key=lambda item: item[1]))
                    config.save_json(sorted_sample_logs, sample_log_path)
                except Exception as e:
                    self.logger.warning(f"Error saving sample logs to {sample_log_path}: {e}")

            self.accelerator.wait_for_everyone()
            if self.config.num_validation_epochs > 0:
                if (datetime.now() - self.last_validation_time).total_seconds() >= self.config.min_validation_time:
                    self.run_validation(global_step)
                    
        self.logger.info("Training complete")
        self.accelerator.end_training()

    @torch.no_grad()
    def run_validation(self, global_step):

        self.logger.info("***** Running validation *****")
        self.logger.info(f"  Num examples = {len(self.dataset["validation"])}")
        self.logger.info(f"  Num Epochs = {self.config.num_validation_epochs}")

        num_validation_process_steps_per_epoch = math.floor(len(self.validation_dataloader) / self.accelerator.num_processes)
        num_validation_update_steps_per_epoch = math.ceil(num_validation_process_steps_per_epoch / self.config.gradient_accumulation_steps)

        validation_logger = TrainLogger()
        sample_logger = TrainLogger()

        for epoch in range(self.config.num_validation_epochs):
            progress_bar = tqdm(total=num_validation_update_steps_per_epoch, disable=not self.accelerator.is_local_main_process)
            progress_bar.set_description(f"Validation Epoch {epoch}")

            for step, batch in enumerate(self.validation_dataloader):        
                if step % self.config.gradient_accumulation_steps == 0:
                    self.module_trainer.init_batch()
                
                module_logs = self.module_trainer.train_batch(batch, step % self.config.gradient_accumulation_steps)
                validation_logger.add_logs(module_logs)
                for i, sample_path in enumerate(batch["sample_paths"]):
                    sample_logger.add_log(sample_path, module_logs["loss"][i])

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    validation_logger.add_logs(self.module_trainer.finish_batch())
                    progress_bar.update(1)
            
            progress_bar.close()

            validation_logs = {}
            for key, value in validation_logger.get_logs():
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

        self.logger.info(f"Validation complete (runtime: {(datetime.now() - self.last_validation_time).total_seconds()}s")
        self.last_validation_time = datetime.now()