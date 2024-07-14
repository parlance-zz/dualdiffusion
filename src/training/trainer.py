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

import config

import logging
import math
import os
import shutil
import subprocess
import atexit
import importlib
from datetime import datetime
from typing import Optional, Literal, Type
from dataclasses import dataclass

import numpy as np
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

from dual_diffusion_pipeline import DualDiffusionPipeline
from .ema_edm2 import PowerFunctionEMA
from .dataset import DatasetConfig, SwitchableDataset
from utils.dual_diffusion_utils import dict_str, normalize

@dataclass
class LRScheduleConfig:
    lr_schedule: Literal["edm2", "linear", "cosine", "cosine_with_restarts",
                         "polynomial", "constant", "constant_with_warmup"]
    learning_rate: float     = 1e-2
    lr_warmup_steps: int     = 5000
    lr_reference_steps: int  = 8000
    lr_decay_exponent: float = 1.

@dataclass
class OptimizerConfig:
    adam_beta1: float       = 0.9
    adam_beta2: float       = 0.99
    adam_epsilon: float     = 1e-8
    max_grad_norm: float    = 10.

@dataclass
class EMAConfig:
    use_ema: bool           = False
    ema_stds: list          = [0.01]
    ema_cpu_offload: bool   = False

@dataclass
class DataLoaderConfig:
    use_pre_encoded_latents: bool = False
    dataloader_num_workers = 0

@dataclass
class LoggingConfig:
    logging_dir: Optional[str] = None
    tensorboard_http_port: Optional[int] = 6006
    tensorboard_num_scalars: Optional[int] = 2000

@dataclass
class ModuleTrainerConfig:
    pass

@dataclass
class DualDiffusionTrainerConfig:

    model_path: str
    model_name: str
    model_src_path: str
    train_config_path: Optional[str]    = None
    module_name: Literal["unet", "vae"]
    seed: Optional[int]                 = None
    train_batch_size: int               = 1
    gradient_accumulation_steps: int    = 1

    num_train_epochs: int               = 500000
    num_validation_epochs: int          = 100
    min_checkpoint_time: int            = 3600
    checkpoints_total_limit: int        = 1

    enable_anomaly_detection: bool      = False
    compile_params: Optional[dict]      = {"fullgraph": True, "dynamic": False}

    lr_schedule_config: LRScheduleConfig
    optimizer_config: OptimizerConfig
    ema_config: EMAConfig
    dataloader_config: DataLoaderConfig
    logging_config: LoggingConfig

    module_trainer_class: Type
    module_trainer_config: ModuleTrainerConfig

    @staticmethod
    def from_json(json_path, **kwargs):

        train_config = config.load_json(json_path)
        train_config["train_config_path"] = json_path

        for key, value in kwargs.items():
            train_config[key] = value
        
        train_config["lr_schedule"] = LRScheduleConfig(**train_config["lr_schedule"])
        train_config["optimizer_config"] = OptimizerConfig(**train_config["optimizer_config"])
        train_config["ema_config"] = EMAConfig(**train_config["ema_config"])
        train_config["data_loader_config"] = DataLoaderConfig(**train_config["data_loader_config"])
        train_config["logging_config"] = LoggingConfig(**train_config["logging_config"])

        module_trainer_module = importlib.import_module(train_config["module_trainer_class"][0])
        train_config["module_trainer_class"] = getattr(module_trainer_module, train_config["module_trainer_class"][1])
        train_config["module_trainer_config"] = train_config["module_trainer_class"].get_config_class()(**train_config["module_trainer_config"])

        return DualDiffusionTrainerConfig(**train_config)

class DualDiffusionTrainer:

    def __init__(self, train_config: DualDiffusionTrainerConfig):

        self.config = train_config

        if self.config.module_name not in ["unet", "vae"]:
            raise ValueError(f"Unknown module type {self.config.module_name}")

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

        self.module, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.module, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

    def init_logging(self):
        
        self.logger = get_logger("dual_diffusion_training", log_level="INFO")

        if self.config.logging_config.logging_dir is None:
            self.config.logging_config.logging_dir = os.path.join(self.config.model_path, f"logs_{self.config.module_name}")
            
        os.makedirs(self.config.logging_config.logging_dir, exist_ok=True)

        log_path = os.path.join(self.config.logging_config.logging_dir, f"train_{self.config.module_name}.log")
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

    def init_accelerator(self):

        accelerator_project_config = ProjectConfiguration(project_dir=self.config.model_path,
                                                          logging_dir=self.config.logging_config.logging_dir)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            log_with="tensorboard",
            project_config=accelerator_project_config,
        )

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(self.config.model_name)

        self.logger.info(self.accelerator.state, main_process_only=False, in_order=True)
        self.accelerator.wait_for_everyone()

    def init_tensorboard(self):

        if self.accelerator.is_main_process and self.config.logging_config.tensorboard_http_port is not None:

            if not is_tensorboard_available():
                raise ImportError("Make sure to install tensorboard if you want to use it for logging during training.")

            tensorboard_args = [
                "tensorboard",
                "--logdir",
                self.config.logging_config.logging_dir,
                "--bind_all",
                "--port",
                str(self.config.logging_config.tensorboard_http_port),
                "--samples_per_plugin",
                f"scalars={self.config.logging_config.tensorboard_num_scalars}",
            ]
            tensorboard_monitor_process = subprocess.Popen(tensorboard_args)

            def cleanup_process():
                try:
                    tensorboard_monitor_process.terminate()
                except Exception:
                    self.logger.warn("Failed to terminate tensorboard process")

            atexit.register(cleanup_process)

    def init_pytorch(self):

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

    def init_module_pipeline(self):

        self.pipeline = DualDiffusionPipeline.from_pretrained(self.config.model_path)
        self.module = getattr(self.pipeline, self.config.module_name)
        self.module_class = type(self.module)

        self.vae = self.pipeline.vae

        if self.config.module_name == "unet":
            if not self.config.dataloader_config.use_pre_encoded_latents:
                self.vae = self.vae.to(self.accelerator.device).to(torch.bfloat16)
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

    def init_ema_module(self):
        
        if self.config.ema_config.use_ema:
            ema_device = "cpu" if self.config.ema_config.ema_cpu_offload else self.accelerator.device
            self.ema_module = PowerFunctionEMA(self.module, stds=self.config.ema_config.ema_stds, device=self.accelerator.device)

            self.logger.info(f"Using EMA model with stds: {self.config.ema_config.ema_stds}")
            self.logger.info(f"EMA CPU offloading {'enabled' if ema_device == 'cpu' else 'disabled'}")

        else:
            self.logger.info("Not using EMA model")

    def init_optimizer(self):
        
        #optimizer = torch.optim.Adam(
        optimizer = torch.optim.AdamW(
            self.module.parameters(),
            lr=self.config.lr_schedule_config.learning_rate,
            betas=(self.config.optimizer_config.adam_beta1, self.config.optimizer_config.adam_beta2),
            weight_decay=0,
            eps=self.config.optimizer_config.adam_epsilon,
        )

        self.logger.info(f"Using Adam optimiser with learning rate {self.config.lr_schedule_config.learning_rate}")
        self.logger.info(f"AdamW beta1: {self.config.optimizer_config.adam_beta1} beta2: {self.config.optimizer_config.adam_beta2} eps: {self.config.optimizer_config.adam_epsilon}")

        return optimizer

    def init_checkpointing(self):

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

                # load diffusers style into model
                load_model = self.module_class.from_pretrained(input_dir, subfolder=self.config.module_name)
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model
            
            if self.ema_module is not None:
                ema_model_dir = os.path.join(input_dir, f"{self.config.module_name}_ema")
                ema_load_errors = self.ema_module.load(ema_model_dir, target_model=model)
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

    def init_dataloader(self):
        
        if self.config.module_name == "unet" and self.config.dataloader_config.use_pre_encoded_latents:
            train_data_dir = config.LATENTS_DATASET_PATH
            sample_crop_width = self.latent_shape[-1]
        else:
            train_data_dir = config.DATASET_PATH
            sample_crop_width = self.pipeline.format.get_sample_crop_width()
        
        dataset_config = DatasetConfig(
            train_data_dir=train_data_dir,
            cache_dir=config.CACHE_PATH,
            num_proc=self.config.dataloader_config.dataloader_num_workers if self.config.dataloader_config.dataloader_num_workers > 0 else None,
            sample_crop_width=sample_crop_width,
            use_pre_encoded_latents=self.config.dataloader_config.use_pre_encoded_latents,
            t_scale=getattr(self.pipeline.unet.config, "t_scale", None) if self.config.module_name == "unet" else None,
        )
        
        self.dataset = SwitchableDataset(dataset_config)

        self.train_dataloader = torch.utils.data.DataLoader(
            self.dataset,
            shuffle=True,
            batch_size=self.config.train_batch_size,
            num_workers=self.config.dataloader_config.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=True if self.config.dataloader_config.dataloader_num_workers > 0 else False,
            prefetch_factor=2 if self.config.dataloader_config.dataloader_num_workers > 0 else None,
            drop_last=True,
        )

        self.logger.info(f"Using training data from {train_data_dir} with {len(self.dataset)} samples ({self.dataset.get_num_filtered_samples()} filtered)")
        if self.config.dataloader_config.dataloader_num_workers > 0:
            self.logger.info(f"Using dataloader with {self.config.dataloader_config.dataloader_num_workers} workers - prefetch factor = 2")

        num_process_steps_per_epoch = math.floor(len(self.train_dataloader) / self.accelerator.num_processes)
        self.num_update_steps_per_epoch = math.ceil(num_process_steps_per_epoch / self.config.gradient_accumulation_steps)
        self.max_train_steps = self.config.num_train_epochs * self.num_update_steps_per_epoch
        self.total_batch_size = self.config.train_batch_size * self.accelerator.num_processes * self.config.gradient_accumulation_steps
 
    def init_lr_scheduler(self):

        if self.config.lr_schedule_config.lr_schedule == "edm2":
            self.logger.info((
                f"Using learning rate schedule {self.config.lr_schedule_config.lr_schedule}",
                f" with warmup steps = {self.config.lr_schedule_config.lr_warmup_steps},",
                f" reference steps = {self.config.lr_schedule_config.lr_reference_steps},",
                f" decay exponent = {self.config.lr_schedule_config.lr_decay_exponent}"))
        else:
            self.logger.info((f"Using learning rate schedule {self.config.lr_schedule_config.lr_schedule}",
                              f"with warmup steps = {self.config.lr_schedule_config.lr_warmup_steps}"))
        
        scaled_lr_warmup_steps = self.config.lr_schedule_config.lr_warmup_steps * self.accelerator.num_processes
        scaled_lr_reference_steps = self.config.lr_schedule_config.lr_reference_steps * self.accelerator.num_processes
        scaled_max_train_steps = self.max_train_steps * self.accelerator.num_processes

        if self.config.lr_schedule_config.lr_schedule == "edm2":
            def edm2_lr_lambda(current_step: int):
                lr = 1.
                if current_step < scaled_lr_warmup_steps:
                    lr *= current_step / scaled_lr_warmup_steps
                if current_step > scaled_lr_reference_steps:
                    lr *= (scaled_lr_reference_steps / current_step) ** self.config.lr_schedule_config.lr_decay_exponent
                return lr
                
            self.lr_scheduler = LambdaLR(self.optimizer, edm2_lr_lambda)
        else:
            self.lr_scheduler = get_scheduler(
                self.config.lr_schedule_config.lr_schedule,
                optimizer=self.optimizer,
                num_warmup_steps=scaled_lr_warmup_steps,
                num_training_steps=scaled_max_train_steps,
            )
   
    def save_checkpoint(self, global_step):
        
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

    def load_checkpoint(self):

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
                if g["lr"] != self.config.lr_schedule_config.learning_rate:
                    g["lr"] = self.config.lr_schedule_config.learning_rate
                    updated_learn_rate = True
            if updated_learn_rate:
                self.lr_scheduler.scheduler.base_lrs = [self.config.lr_schedule_config.learning_rate]
                self.logger.info(f"Using updated learning rate: {self.config.lr_schedule_config.learning_rate}")

        if global_step > 0:
            resume_global_step = global_step * self.config.gradient_accumulation_steps
            first_epoch = global_step // self.num_update_steps_per_epoch
            resume_step = resume_global_step % (self.num_update_steps_per_epoch * self.config.gradient_accumulation_steps)

        return global_step, resume_step, first_epoch

    def train(self):

        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num examples = {len(self.train_dataset)}")
        self.logger.info(f"  Num Epochs = {self.config.num_train_epochs}")
        self.logger.info(f"  Instantaneous batch size per device = {self.config.train_batch_size}")
        self.logger.info(f"  Gradient Accumulation steps = {self.config.gradient_accumulation_steps}")
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {self.total_batch_size}")
        self.logger.info(f"  Total optimization steps for full run = {self.max_train_steps}")
        self.logger.info(f"  Path to save/load checkpoints = {self.config.model_path}")

        global_step, resume_step, first_epoch = self.load_checkpoint()
        module_trainer = self.config.module_trainer_class(self.config.module_trainer_config, self)

        if args.module == "vae":

            latent_shape = module.get_latent_shape(sample_shape)
            channel_kl_loss_weight = model_params["vae_training_params"]["channel_kl_loss_weight"]
            recon_loss_weight = model_params["vae_training_params"]["recon_loss_weight"]
            point_loss_weight = model_params["vae_training_params"]["point_loss_weight"]
            imag_loss_weight = model_params["vae_training_params"]["imag_loss_weight"]

            logger.info("Training VAE model:")
            logger.info(f"VAE Training params: {dict_str(model_params['vae_training_params'])}")
            logger.info(f"Channel KL loss weight: {channel_kl_loss_weight}")
            logger.info(f"Recon loss weight: {recon_loss_weight} - Point loss weight: {point_loss_weight} - Imag loss weight: {imag_loss_weight}")

            target_snr = module.get_target_snr()
            target_noise_std = (1 / (target_snr**2 + 1))**0.5
            logger.info(f"VAE Target SNR: {target_snr:{8}f}")

            channel_kl_loss_weight = torch.tensor(channel_kl_loss_weight, device=accelerator.device, dtype=torch.float32)
            recon_loss_weight = torch.tensor(recon_loss_weight, device=accelerator.device, dtype=torch.float32)
            point_loss_weight = torch.tensor(point_loss_weight, device=accelerator.device, dtype=torch.float32)
            imag_loss_weight = torch.tensor(imag_loss_weight, device=accelerator.device, dtype=torch.float32)

            module_log_channels = [
                "channel_kl_loss_weight",
                "recon_loss_weight",
                "imag_loss_weight",
                "real_loss",
                "imag_loss",
                "channel_kl_loss",
                "latents_mean",
                "latents_std",
                "latents_snr",
                "point_similarity_loss",
                "point_loss_weight",
            ]

        elif args.module == "unet":
            logger.info("Training UNet model:")

            use_stratified_sigma_sampling = model_params["unet_training_params"]["stratified_sigma_sampling"]
            logger.info(f"Using stratified sigma sampling: {use_stratified_sigma_sampling}")
            sigma_ln_std = model_params["unet_training_params"]["sigma_ln_std"]
            sigma_ln_mean = model_params["unet_training_params"]["sigma_ln_mean"]
            logger.info(f"Sampling training sigmas with sigma_ln_std = {sigma_ln_std:.4f}, sigma_ln_mean = {sigma_ln_mean:.4f}")
            logger.info(f"sigma_max = {module.sigma_max}, sigma_min = {module.sigma_min}")

            if args.num_unet_loss_buckets > 0:
                logger.info(f"Using {args.num_unet_loss_buckets} loss buckets")
                unet_loss_buckets = torch.zeros(args.num_unet_loss_buckets,
                                                    device="cpu", dtype=torch.float32)
                unet_loss_bucket_counts = torch.zeros(args.num_unet_loss_buckets,
                                                        device="cpu", dtype=torch.float32)
            else:
                logger.info("UNet loss buckets are disabled")

            if vae is not None:
                if vae.config.last_global_step == 0 and args.train_data_format != ".safetensors":
                    logger.error("VAE model has not been trained, aborting...")
                    exit(1)
                latent_shape = vae.get_latent_shape(sample_shape)
                target_snr = vae.get_target_snr()
            else:
                target_snr = model_params.get("target_snr", 1e4)

            logger.info(f"Target SNR: {target_snr:.3f}")

            input_perturbation = model_params["unet_training_params"]["input_perturbation"]
            if input_perturbation > 0: logger.info(f"Using input perturbation of {input_perturbation}")
            else: logger.info("Input perturbation is disabled")

            logger.info(f"Dropout: {module.dropout} Conditioning dropout: {module.label_dropout}")

            sigma_ln_std = torch.tensor(sigma_ln_std, device=accelerator.device, dtype=torch.float32)
            sigma_ln_mean = torch.tensor(sigma_ln_mean, device=accelerator.device, dtype=torch.float32)
            module_log_channels = [
                "sigma_ln_std",
                "sigma_ln_mean",
            ]

            #sigma_sample_max_temperature = 4
            #sigma_sample_pdf_skew = 0.
            #sigma_temperature_ref_steps = 20000
            #sigma_sample_temperature = min(global_step / sigma_temperature_ref_steps, 1) * sigma_sample_max_temperature
            #sigma_sample_resolution = 128 - 1
            #sigma_max = module.sigma_max
            #sigma_min = module.sigma_data / target_snr
            #sigma_data = module.sigma_data
            #ln_sigma = torch.linspace(np.log(sigma_min), np.log(sigma_max), sigma_sample_resolution, device=accelerator.device)
            #ln_sigma_error = module.logvar_linear(module.logvar_fourier(ln_sigma/4)).float().flatten().detach()
            #sigma_distribution_pdf = (-sigma_sample_temperature * ln_sigma_error).exp() + torch.linspace(sigma_sample_pdf_skew**0.5, 0, sigma_sample_resolution, device=accelerator.device).square()
            #sigma_sampler = SigmaSampler(sigma_max, sigma_min, sigma_data,
            #                             distribution="ln_data", distribution_pdf=sigma_distribution_pdf)
            #sigma_sampler = SigmaSampler(module.sigma_max, module.sigma_data / target_snr, module.sigma_data,
            #                             distribution="log_sech", dist_scale=1, dist_offset=0.1)#0.21) #0.2)
            sigma_sampler = SigmaSampler(module.sigma_max, module.sigma_data / target_snr, module.sigma_data,
                                        distribution="log_sech", dist_scale=1., dist_offset=0.)#0.21) #0.2)

        total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        if args.train_data_format != ".safetensors":
            logger.info(f"Sample shape: {sample_shape}")
        if latent_shape is not None:
            logger.info(f"Latent shape: {latent_shape}")

        module.normalize_weights()

        for epoch in range(first_epoch, args.num_train_epochs):

            module.train().requires_grad_(True)

            train_loss = 0.
            grad_accum_steps = 0
            module_logs = {}
            for channel in module_log_channels:
                module_logs[channel] = 0.
            
            progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(train_dataloader):
                # skip steps until we reach the resumed step if resuming from checkpoint - todo: this is inefficient
                if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue
                
                if args.module == "unet" and grad_accum_steps == 0:
                    
                    if use_stratified_sigma_sampling:
                        # instead of randomly sampling each sigma, distribute the batch evenly across the distribution
                        # and add a random offset for continuous uniform coverage
                        global_quantiles = (torch.arange(total_batch_size, device=accelerator.device)+0.5) / total_batch_size
                        global_quantiles += (torch.rand(1, device=accelerator.device) - 0.5) / total_batch_size
                        global_quantiles = accelerator.gather(global_quantiles.unsqueeze(0))[0] # sync quantiles across all ranks / processes
                    
                    if args.num_unet_loss_buckets > 0:
                        unet_loss_buckets.zero_()
                        unet_loss_bucket_counts.zero_()

                with accelerator.accumulate(module):

                    raw_samples = batch["input"]
                    sample_game_ids = batch["game_ids"]
                    if model_params["t_scale"] is not None:
                        sample_t_ranges = batch["t_ranges"]
                    else:
                        sample_t_ranges = None
                    #sample_author_ids = batch["author_ids"]
                    raw_sample_paths = batch["sample_paths"]

                    if args.module == "unet":
                        
                        class_labels = pipeline.get_class_labels(sample_game_ids)
                        unet_class_embeddings = module.get_class_embeddings(class_labels)

                        if args.train_data_format == ".safetensors":
                            #samples = normalize(raw_samples + torch.randn_like(raw_samples) * np.exp(0.5 * vae.get_noise_logvar())).float()
                            samples = normalize(raw_samples).float()
                            assert samples.shape == latent_shape
                        else:
                            samples = pipeline.format.raw_to_sample(raw_samples)
                            if vae is not None:
                                vae_class_embeddings = vae.get_class_embeddings(class_labels)
                                samples = vae.encode(samples.to(torch.bfloat16), vae_class_embeddings, pipeline.format).mode().detach()
                                samples = normalize(samples).float()

                        if use_stratified_sigma_sampling:
                            process_batch_quantiles = global_quantiles[accelerator.local_process_index::accelerator.num_processes]
                            quantiles = process_batch_quantiles[grad_accum_steps * args.train_batch_size:(grad_accum_steps+1) * args.train_batch_size]
                        else:
                            quantiles = None

                        #if use_stratified_sigma_sampling:
                        #    batch_normal = sigma_ln_mean + (sigma_ln_std * (2 ** 0.5)) * (quantiles * 2 - 1).erfinv().clip(min=-5, max=5)
                        #else:
                        #    batch_normal = torch.randn(samples.shape[0], device=accelerator.device) * sigma_ln_std + sigma_ln_mean
                        #sigma = batch_normal.exp().clip(min=module.sigma_min, max=module.sigma_max)
                        sigma = sigma_sampler.sample(samples.shape[0], quantiles=quantiles).to(accelerator.device)
                        noise = torch.randn_like(samples) * sigma.view(-1, 1, 1, 1)
                        samples = samples * module.sigma_data

                        denoised, error_logvar = module(samples + noise,
                                                        sigma,
                                                        unet_class_embeddings,
                                                        sample_t_ranges,
                                                        pipeline.format,
                                                        return_logvar=True)
                        
                        mse_loss = torch.nn.functional.mse_loss(denoised, samples, reduction="none")
                        loss_weight = (sigma ** 2 + module.sigma_data ** 2) / (sigma * module.sigma_data) ** 2
                        loss = (loss_weight.view(-1, 1, 1, 1) / error_logvar.exp() * mse_loss + error_logvar).mean()
                        #loss = (loss_weight.view(-1, 1, 1, 1) * mse_loss).mean()
                        
                        if args.num_unet_loss_buckets > 0:
                            batch_loss = mse_loss.mean(dim=(1,2,3)) * loss_weight

                            global_step_quantiles = (accelerator.gather(sigma.detach()).cpu().log() - np.log(module.sigma_min)) / (np.log(module.sigma_max) - np.log(module.sigma_min))
                            global_step_batch_loss = accelerator.gather(batch_loss.detach()).cpu()
                            target_buckets = (global_step_quantiles * unet_loss_buckets.shape[0]).long().clip(min=0, max=unet_loss_buckets.shape[0]-1)
                            unet_loss_buckets.index_add_(0, target_buckets, global_step_batch_loss)
                            unet_loss_bucket_counts.index_add_(0, target_buckets, torch.ones_like(global_step_batch_loss))

                    elif args.module == "vae":

                        samples_dict = pipeline.format.raw_to_sample(raw_samples, return_dict=True)
                        vae_class_embeddings = module.get_class_embeddings(pipeline.get_class_labels(sample_game_ids))
                        
                        posterior = module.encode(samples_dict["samples"],
                                                vae_class_embeddings,
                                                pipeline.format)
                        latents = posterior.sample(pipeline.noise_fn)
                        latents_mean = latents.mean()
                        latents_std = latents.std()

                        measured_sample_std = (latents_std**2 - target_noise_std**2).clip(min=0)**0.5
                        latents_snr = measured_sample_std / target_noise_std
                        model_output = module.decode(latents,
                                                    vae_class_embeddings,
                                                    pipeline.format)

                        recon_samples_dict = pipeline.format.sample_to_raw(model_output, return_dict=True, decode=False)
                        point_similarity_loss = (samples_dict["samples"] - recon_samples_dict["samples"]).abs().mean()
                        
                        recon_loss_logvar = module.get_recon_loss_logvar()
                        real_loss, imag_loss = pipeline.format.get_loss(recon_samples_dict, samples_dict)
                        real_nll_loss = (real_loss / recon_loss_logvar.exp() + recon_loss_logvar) * recon_loss_weight
                        imag_nll_loss = (imag_loss / recon_loss_logvar.exp() + recon_loss_logvar) * (recon_loss_weight * imag_loss_weight)

                        latents_square_norm = (torch.linalg.vector_norm(latents, dim=(1,2,3), dtype=torch.float32) / latents[0].numel()**0.5).square()
                        latents_batch_mean = latents.mean(dim=(1,2,3))
                        channel_kl_loss = (latents_batch_mean.square() + latents_square_norm - 1 - latents_square_norm.log()).mean()
                        
                        loss = real_nll_loss + imag_nll_loss + channel_kl_loss * channel_kl_loss_weight + point_similarity_loss * point_loss_weight
                    else:
                        raise ValueError(f"Unknown module {args.module}")
                    
                    # Gather the losses across all processes for logging (if we use distributed training).
                    grad_accum_steps += 1
                    avg_loss = accelerator.gather(loss.detach()).mean()
                    train_loss += avg_loss.item()
                    for channel in module_log_channels:
                        avg = accelerator.gather(locals()[channel].detach()).mean()
                        module_logs[channel] += avg.item()

                    # Backpropagate
                    accelerator.backward(loss)

                    if accelerator.sync_gradients: # clip and check for nan/inf grad
                        grad_norm = accelerator.gather(accelerator.clip_grad_norm_(module.parameters(), args.max_grad_norm)).mean().item()
                        if math.isinf(grad_norm) or math.isnan(grad_norm):
                            logger.warning(f"Warning: grad norm is {grad_norm} - step={global_step} loss={loss.item()} debug_last_sample_paths={raw_sample_paths}")
                        if math.isnan(grad_norm):
                            logger.error(f"Error: grad norm is {grad_norm}, aborting...")
                            import pdb; pdb.set_trace(); exit(1)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    module.normalize_weights()

                    #sigma_sample_temperature = min(global_step / sigma_temperature_ref_steps, 1) * sigma_sample_max_temperature
                    #ln_sigma = torch.linspace(np.log(sigma_min), np.log(sigma_max), sigma_sample_resolution, device=accelerator.device)
                    #ln_sigma_error = module.logvar_linear(module.logvar_fourier(ln_sigma/4)).float().flatten().detach()
                    #sigma_distribution_pdf = (-sigma_sample_temperature * ln_sigma_error).exp() + torch.linspace(sigma_sample_pdf_skew**0.5, 0, sigma_sample_resolution, device=accelerator.device).square()
                    #sigma_sampler.update_pdf(sigma_distribution_pdf)
                
                    progress_bar.update(1)
                    global_step += 1

                    if grad_accum_steps >= args.gradient_accumulation_steps: # don't log incomplete batches
                        logs = {"loss": train_loss / grad_accum_steps,
                                "lr": lr_scheduler.get_last_lr()[0],
                                "step": global_step,
                                "grad_norm": grad_norm}
                        for channel in module_log_channels:
                            if "loss_weight" in channel:
                                channel_name = f"{args.module}_loss_weight/{channel}"
                            else:
                                channel_name = f"{args.module}/{channel}"
                            logs[channel_name] = module_logs[channel] / grad_accum_steps

                    if args.use_ema:
                        std_betas = ema_module.update(global_step * total_batch_size, total_batch_size)
                        for std, beta in std_betas:
                            logs[f"ema/std_{std:.3f}_beta"] = beta

                    if args.module == "unet" and args.num_unet_loss_buckets > 0:
                        for i in range(unet_loss_buckets.shape[0]):
                            if unet_loss_bucket_counts[i].item() > 0:
                                bucket_ln_sigma_start = np.log(module.sigma_min) + i * (np.log(module.sigma_max) - np.log(module.sigma_min)) / unet_loss_buckets.shape[0]
                                bucket_ln_sigma_end = np.log(module.sigma_min) + (i+1) * (np.log(module.sigma_max) - np.log(module.sigma_min)) / unet_loss_buckets.shape[0]
                                logs[f"unet_loss_buckets/b{i} s:{bucket_ln_sigma_start:.3f} ~ {bucket_ln_sigma_end:.3f}"] = (unet_loss_buckets[i] / unet_loss_bucket_counts[i]).item()

                    accelerator.log(logs, step=global_step)
                    progress_bar.set_postfix(**logs)

                    train_loss = 0.
                    grad_accum_steps = 0
                    for channel in module_log_channels:
                        module_logs[channel] = 0.
                        
                    if accelerator.is_main_process:
                        _save_checkpoint = ((global_step % args.checkpointing_steps) == 0)

                        if os.path.exists(os.path.join(args.output_dir, "_save_checkpoint")):
                            _save_checkpoint = True

                        if _save_checkpoint:
                            save_checkpoint(module, args.module, args.output_dir, global_step, accelerator, args.checkpoints_total_limit)

                        if os.path.exists(os.path.join(args.output_dir, "_save_checkpoint")):
                            os.remove(os.path.join(args.output_dir, "_save_checkpoint"))

                if global_step >= max_train_steps:
                    logger.info(f"Reached max train steps ({max_train_steps}) - Training complete")
                    break
            
            progress_bar.close()

            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                
                if args.num_validation_samples > 0 and args.num_validation_epochs > 0:
                    if epoch % args.num_validation_epochs == 0:
                        module.eval().requires_grad_(False)
                        logger.info("Running validation... ")
        
                        try:
                            pipeline.set_progress_bar_config(disable=True)

                            if args.module == "unet":
                                log_validation_unet(
                                    pipeline,
                                    args,
                                    accelerator,
                                    global_step,
                                )
                            elif args.module == "vae":
                                log_validation_vae(
                                    pipeline,
                                    args,
                                    accelerator,
                                    global_step,
                                )

                        except Exception as e:
                            logger.error(f"Error running validation: {e}")

                        module.train().requires_grad_(True)
        
        logger.info("Training complete")
        accelerator.end_training()