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

import argparse
import logging
import math
import os
import shutil
import subprocess
import atexit
from glob import glob
from typing import Any
from dotenv import load_dotenv

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchaudio
from torch.optim.lr_scheduler import LambdaLR
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset, Audio
from tqdm.auto import tqdm

import diffusers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import deprecate, is_tensorboard_available

#from unet_dual import UNetDualModel
from unet_edm2 import UNet
from autoencoder_kl_dual import AutoencoderKLDual
from dual_diffusion_pipeline import DualDiffusionPipeline
from dual_diffusion_utils import init_cuda, load_audio, save_audio, load_raw, save_raw, dict_str, normalize_lufs, slerp, slerp_loss


logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="DualDiffusion training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained / new model",
    )
    parser.add_argument(
        "--module",
        type=str,
        default="unet",
        required=False,
        help="Which module in the model to train. Choose between ['unet', 'vae']",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where checkpoints will be written. Defaults to pretrained_model_name_or_path.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--vae_encode_batch_size",
        type=int,
        default=0,
        help="If set, use this batch size for VAE encoding when training diffusion UNet module. Defaults to train_batch_size."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup", "edm2"] If using edm2 lr_warmup_steps is instead used as the beginning of decay (25k default)'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--num_timestep_loss_buckets",
        type=int,
        default=10,
        help=("When training a unet/diffusion module un-weighted loss for groups of timesteps can be logged separately. Set to 0 to disable."),
    )
    #parser.add_argument(
    #    "--pitch_augmentation_range",
    #    type=float,
    #    default=0, #2/12,
    #    help="Modulate the pitch of the sample by a random amount within this range (in octaves) - Currently unused",
    #)
    #parser.add_argument(
    #    "--tempo_augmentation_range",
    #    type=float,
    #    default=0, #0.167,
    #    help="Modulate the tempo of the sample by a random amount within this range (value of 1 is double/half speed)  - Currently unused",
    #)
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0, help="The inverse gamma value for the EMA decay.")
    parser.add_argument("--ema_power", type=float, default=2 / 3, help="The power value for the EMA decay.")
    parser.add_argument("--ema_min_decay", type=float, default=0., help="The minimum decay magnitude for EMA.")
    parser.add_argument("--ema_max_decay", type=float, default=0.9999, help="The maximum decay magnitude for EMA.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        required=False,
        help=(
            "Specify a dataset name to load training data from the huggingface hub. If not specified the training data will be"
            " loaded from the path specified in the DATASET_PATH environment variable."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1., type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default=None,
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " output_dir/logs_(modulename)/(model_name)"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=None,
        help=(
            "Save a checkpoint of the training state every X updates. By default, a checkpoint will be saved after every epoch."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=1,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="latest",
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint (default).'
            ' Use `"none"` to start a new training run even if checkpoints exist in the output directory.'
        ),
    )
    parser.add_argument(
        "--num_validation_epochs",
        type=int,
        default=5,
        help="Number of epochs between creating new validation samples.",
    )
    parser.add_argument(
        "--num_validation_samples",
        type=int,
        default=4,
        help="Number of samples to generate for validation.",
    )
    parser.add_argument(
        "--num_validation_steps",
        type=int,
        default=250,
        help="Number of steps to use when creating validation samples.",
    )
    parser.add_argument(
        "--validation_sample_dir",
        type=str,
        default=None,
        help="A folder containing samples used only for validation. By default the training data dir is used.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default=None,
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--enable_anomaly_detection",
        action="store_true",
        help="Enable pytorch anomaly detection - Kills performance but can be used to find the cause of NaN / inf gradients.",
    )

    args = parser.parse_args()

    # validate args / add automatic args

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    args.train_data_dir = os.environ.get("DATASET_PATH", None)
    args.train_data_format = os.environ.get("DATASET_FORMAT", None)
    args.train_data_num_channels = os.environ.get("DATASET_NUM_CHANNELS", None)
    args.train_data_raw_format = os.environ.get("DATASET_RAW_FORMAT", None)

    if args.train_data_dir is None:
        raise ValueError("DATASET_PATH environment variable is undefined.")
    if args.train_data_format is None:
        raise ValueError("DATASET_FORMAT environment variable is undefined.")
    if args.train_data_num_channels is None:
        raise ValueError("DATASET_NUM_CHANNELS environment variable is undefined.")
    else:
        args.train_data_num_channels = int(args.train_data_num_channels)
    if args.train_data_format == ".raw" and args.train_data_raw_format is None:
        raise ValueError("DATASET_FORMAT is '.raw' and DATASET_RAW_SAMPLE_FORMAT environment variable is undefined.")
        
    if not os.path.exists(args.train_data_dir) and args.dataset_name is None:
        raise ValueError(f"Training data directory {args.train_data_dir} does not exist.")
    
    args.hf_token = os.environ.get("HF_TOKEN", None)

    if args.validation_sample_dir is None:
        args.validation_sample_dir = args.train_data_dir
    else:
        if not os.path.exists(args.validation_sample_dir):
            raise ValueError(f"Validation sample directory {args.validation_sample_dir} does not exist.")
        
    args.module = args.module.lower().strip()
    if args.module not in ["unet", "vae"]:
        raise ValueError(f"Unknown module type {args.module}")
    
    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )

    if args.output_dir is None:
        args.output_dir = args.pretrained_model_name_or_path
    os.makedirs(args.output_dir, exist_ok=True)

    if args.logging_dir is None:
        args.logging_dir = os.path.join(args.output_dir, f"logs_{args.module}")
    os.makedirs(args.logging_dir, exist_ok=True)

    if args.tracker_project_name is None:
        args.tracker_project_name = os.path.basename(args.output_dir)

    if args.lr_scheduler == "edm2" and args.lr_warmup_steps == 500:
        args.lr_warmup_steps = 25000

    return args

def init_logging(accelerator, logging_dir, module_type, report_to):
    
    log_path = os.path.join(logging_dir, f"train_{module_type}.log")
    logging.basicConfig(
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ],
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    logger.info(f"Logging to {log_path}")

    datasets.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()

    if accelerator.is_main_process:
        if report_to == "tensorboard":
            if not is_tensorboard_available():
                raise ImportError("Make sure to install tensorboard if you want to use it for logging during training.")
            port = int(os.environ.get("TENSORBOARD_HTTP_PORT", 6006))
            tensorboard_args = [
                "tensorboard",
                "--logdir",
                logging_dir,
                "--bind_all",
                "--port",
                str(port),
                "--samples_per_plugin",
                "scalars=2000",
            ]
            tensorboard_monitor_process = subprocess.Popen(tensorboard_args)

            def cleanup_process():
                try:
                    tensorboard_monitor_process.terminate()
                except Exception:
                    logger.warn("Failed to terminate tensorboard process")
                    pass

            atexit.register(cleanup_process)

def init_accelerator(project_dir,
                     grad_accumulation_steps,
                     mixed_precision,
                     logging_dir,
                     log_with,
                     tracker_project_name):

    accelerator_project_config = ProjectConfiguration(project_dir=project_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=log_with,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(tracker_project_name)

    return accelerator

def init_accelerator_loadsave_hooks(accelerator, module_type, module_class, ema_module):

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            if ema_module is not None:
                ema_module.save_pretrained(os.path.join(output_dir, f"{module_type}_ema"))

            for model in models:
                model.save_pretrained(os.path.join(output_dir, module_type))
                weights.pop() # make sure to pop weight so that corresponding model is not saved again

    def load_model_hook(models, input_dir):
        if ema_module is not None:
            if not os.path.exists(os.path.join(input_dir, f"{module_type}_ema")):
                logger.info("EMA model in checkpoint not found, using new ema model")
            else:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, f"{module_type}_ema"), module_class)
                ema_module.load_state_dict(load_model.state_dict())
                ema_module.to(accelerator.device)
                del load_model

        for _ in range(len(models)):
            model = models.pop() # pop models so that they are not loaded again

            # load diffusers style into model
            load_model = module_class.from_pretrained(input_dir, subfolder=module_type)
            model.register_to_config(**load_model.config)
            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    logger.info("Registered accelerator hooks for model load/save")

def save_checkpoint(module, module_type, output_dir, global_step, accelerator, checkpoints_total_limit):
    
    module.config["last_global_step"] = global_step
    save_path = os.path.join(output_dir, f"{module_type}_checkpoint-{global_step}")
    accelerator.save_state(save_path)
    logger.info(f"Saved state to {save_path}")

    # copy all source code / scripts to model folder for posterity
    source_src_path = os.path.dirname(__file__)
    target_src_path = os.path.join(save_path, "src")
    logger.info(f"Copying source code at '{source_src_path}' to checkpoint folder '{target_src_path}'")

    try:
        os.makedirs(target_src_path, exist_ok=True)
        src_file_types = ["py", "cmd", "yml", "sh", "env"]
        src_files = []
        for file_type in src_file_types:
            src_files += glob(f"*.{file_type}") # todo: update this to be case insensitive when upgrading to python 3.12
        for src_file in src_files:
            shutil.copy(src_file, os.path.join(target_src_path, os.path.basename(src_file)))
    except Exception as e:
        logger.warning(f"Failed to copy source code to model folder: {e}")

    # delete old checkpoints AFTER saving new checkpoint
    if checkpoints_total_limit is not None:
        try:
            checkpoints = os.listdir(output_dir)
            checkpoints = [d for d in checkpoints if d.startswith(f"{module_type}_checkpoint")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

            if len(checkpoints) > checkpoints_total_limit:
                num_to_remove = len(checkpoints) - checkpoints_total_limit
                if num_to_remove > 0:
                    removing_checkpoints = checkpoints[0:num_to_remove]
                    logger.info(
                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                    )
                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                    for removing_checkpoint in removing_checkpoints:
                        removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                        shutil.rmtree(removing_checkpoint)

        except Exception as e:
            logger.error(f"Error removing old checkpoints: {e}")

def load_checkpoint(checkpoint,
                    output_dir,
                    module_type,
                    accelerator,
                    optimizer,
                    lr_scheduler,
                    learning_rate,
                    adam_weight_decay,
                    gradient_accumulation_steps,
                    num_update_steps_per_epoch):

    global_step = 0
    resume_step = 0
    first_epoch = 0
    
    if checkpoint == "none":
        return global_step, resume_step, first_epoch
    
    if checkpoint != "latest": # load specific checkpoint
        path = os.path.basename(checkpoint)
    else: # get most recent checkpoint
        dirs = os.listdir(output_dir)
        dirs = [d for d in dirs if d.startswith(f"{module_type}_checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None

    if path is None:
        logger.warning(f"Checkpoint '{checkpoint}' does not exist. Starting a new training run.")
    else:
        global_step = int(path.split("-")[1])
        logger.info(f"Resuming from checkpoint {path} (global step: {global_step})")
        accelerator.load_state(os.path.join(output_dir, path))

        # update learning rate in case we've changed it
        updated_learn_rate = False
        for g in optimizer.param_groups:
            if g["lr"] != learning_rate:
                g["lr"] = learning_rate
                updated_learn_rate = True
        if updated_learn_rate:
            lr_scheduler.scheduler.base_lrs = [learning_rate]
            logger.info(f"Using updated learning rate: {learning_rate}")

        # update weight decay in case we've changed it
        updated_weight_decay = False
        for g in optimizer.param_groups:
            if g["weight_decay"] != adam_weight_decay:
                g["weight_decay"] = adam_weight_decay
                updated_weight_decay = True
        if updated_weight_decay:
            logger.info(f"Using updated adam weight decay: {adam_weight_decay}")

    if global_step > 0:
        resume_global_step = global_step * gradient_accumulation_steps
        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = resume_global_step % (num_update_steps_per_epoch * gradient_accumulation_steps)

    return global_step, resume_step, first_epoch

def init_module_pipeline(pretrained_model_name_or_path, module_type, vae_encode_batch_size, device):

    pipeline = DualDiffusionPipeline.from_pretrained(pretrained_model_name_or_path)
    module = getattr(pipeline, module_type)

    if module_type == "unet":
        #module_class = UNetDualModel
        module_class = UNet

        vae = getattr(pipeline, "vae", None)
        if vae is not None:
            vae = vae.to(device).half()

            if vae_encode_batch_size > 0:
                vae.enable_slicing(vae_encode_batch_size)

            logger.info(f"Training diffusion model with VAE")
        else:
            logger.info(f"Training diffusion model without VAE")

    elif module_type == "vae":
        module_class = AutoencoderKLDual

        vae = None
        if getattr(pipeline, "unet", None) is not None:
            pipeline.unet = pipeline.unet.to("cpu")
    else:
        raise ValueError(f"Unknown module {module_type}")
    
    pipeline.format = pipeline.format.to(device)

    logger.info(f"Training module class: {module_class.__name__}")
    return pipeline, module, module_class, vae

def init_ema_module(pretrained_model_name_or_path,
                    module_type,
                    module_class,
                    revision,
                    ema_min_decay,
                    ema_max_decay,
                    ema_inv_gamma,
                    ema_power,
                    device):
    
    ema_module = module_class.from_pretrained(
        pretrained_model_name_or_path, subfolder=module_type, revision=revision
    )
    ema_module = EMAModel(ema_module.parameters(),
                        model_cls=module_class,
                        model_config=ema_module.config,
                        min_decay=ema_min_decay,
                        decay=ema_max_decay,
                        inv_gamma=ema_inv_gamma,
                        power=ema_power).to(device)

    logger.info(f"Using EMA model with max decay: {ema_max_decay} min decay: {ema_min_decay} inv gamma: {ema_inv_gamma} power: {ema_power}")

    return ema_module
    
def init_optimizer(use_8bit_adam,
                   learning_rate,
                   adam_beta1,
                   adam_beta2,
                   adam_weight_decay,
                   adam_epsilon,
                   module):
    
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb # type: ignore
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    
    optimizer = optimizer_cls(
        module.parameters(),
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    logger.info(f"Using optimiser {optimizer_cls.__name__} with learning rate {learning_rate}")
    logger.info(f"AdamW beta1: {adam_beta1} beta2: {adam_beta2} eps: {adam_epsilon} weight_decay: {adam_weight_decay}")

    return optimizer

def init_lr_scheduler(lr_schedule, optimizer, lr_warmup_steps, max_train_steps, num_processes):

    lr_warmup_steps *= num_processes
    max_train_steps *= num_processes

    if lr_schedule == "edm2":

        def edm2_lr_lambda(current_step: int):
            if current_step < lr_warmup_steps:
                return 1.
            else:
                return (lr_warmup_steps / current_step) ** 0.5
            
        return LambdaLR(optimizer, edm2_lr_lambda)

    return get_scheduler(
        lr_schedule,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=max_train_steps,
    )

class DatasetTransformer(torch.nn.Module):

    def __init__(self, dataset_transform_params):
        self.dataset_format = dataset_transform_params["format"]
        self.dataset_raw_format = dataset_transform_params["raw_format"]
        self.dataset_num_channels = dataset_transform_params["num_channels"]
        self.sample_crop_width = dataset_transform_params["crop_width"]
    
    def __call__(self, examples):

        samples = []
        paths = []

        for audio in examples["audio"]:
                    
            input_audio = audio["bytes"] or audio["path"]

            if self.dataset_format == ".raw":
                sample = load_raw(input_audio,
                                  dtype=self.dataset_raw_format,
                                  num_channels=self.dataset_num_channels,
                                  start=-1,
                                  count=self.sample_crop_width)    
            else:
                sample = load_audio(input_audio,
                                    start=-1,
                                    count=self.sample_crop_width)

            samples.append(sample)
            paths.append(audio["path"])

        return {"input": samples, "sample_paths": paths}

def init_dataloader(accelerator,
                    dataset_name,
                    hf_token,
                    train_data_dir,
                    cache_dir,
                    train_batch_size,
                    dataloader_num_workers,
                    max_train_samples,
                    dataset_format,
                    dataset_raw_format,
                    dataset_num_channels,
                    sample_rate,
                    sample_crop_width,
                    prefetch_factor=4):

    dataset_transform_params = {
        "format": dataset_format,
        "raw_format": dataset_raw_format,
        "num_channels": dataset_num_channels,
        "sample_rate": sample_rate,
        "crop_width": sample_crop_width,
    }
    dataset_transform = DatasetTransformer(dataset_transform_params)

    if dataset_name is None:
        dataset_name = "audiofolder"
        data_files = {"train": os.path.join(train_data_dir, "**")}
    else:
        data_files = f"**{dataset_format}"

    dataset = load_dataset(
        dataset_name,
        data_files=data_files,
        cache_dir=cache_dir,
        num_proc=dataloader_num_workers if dataloader_num_workers > 0 else None,
        token=hf_token,
    ).cast_column("audio", Audio(decode=False))

    with accelerator.main_process_first():
        if max_train_samples is not None:
            dataset["train"] = dataset["train"].select(range(max_train_samples))
        train_dataset = dataset["train"].with_transform(dataset_transform)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=train_batch_size,
        num_workers=dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True if dataloader_num_workers > 0 else False,
        prefetch_factor=prefetch_factor if dataloader_num_workers > 0 else None,
        drop_last=True,
    )

    logger.info(f"Using training data from {train_data_dir} with {len(train_dataset)} samples and batch size {train_batch_size}")
    if dataloader_num_workers > 0:
        logger.info(f"Using dataloader with {dataloader_num_workers} workers - prefetch factor: {prefetch_factor}")
    logger.info(f"Dataset transform params: {dict_str(dataset_transform_params)}")

    return train_dataset, train_dataloader

def do_training_loop(args,
                     accelerator,
                     module,
                     ema_module,
                     pipeline,
                     vae,
                     lr_scheduler,
                     optimizer,
                     first_epoch,
                     global_step,
                     resume_step,
                     num_update_steps_per_epoch,
                     max_train_steps,
                     train_dataloader):
                     
    model_params = pipeline.config["model_params"]
    sample_shape = pipeline.format.get_sample_shape(bsz=args.train_batch_size)
    latent_shape = None

    if args.module == "vae":

        latent_shape = module.get_latent_shape(sample_shape)
        kl_loss_weight = model_params["kl_loss_weight"]
        channel_kl_loss_weight = model_params["channel_kl_loss_weight"]
        recon_loss_weight = model_params["recon_loss_weight"]
        point_loss_weight = model_params["point_loss_weight"]

        logger.info("Training VAE model:")
        logger.info(f"Loss params: {dict_str(model_params['loss_params'])}")
        logger.info(f"Using KL loss weight: {kl_loss_weight} - Channel KL loss weight: {channel_kl_loss_weight}")
        logger.info(f"Recon loss weight: {recon_loss_weight} - Point loss weight: {point_loss_weight}")

        kl_loss_weight = torch.tensor(kl_loss_weight, device=accelerator.device, dtype=torch.float32)
        channel_kl_loss_weight = torch.tensor(channel_kl_loss_weight, device=accelerator.device, dtype=torch.float32)
        recon_loss_weight = torch.tensor(recon_loss_weight, device=accelerator.device, dtype=torch.float32)
        point_loss_weight = torch.tensor(point_loss_weight, device=accelerator.device, dtype=torch.float32)

        module_log_channels = [
            "kl_loss_weight",
            "channel_kl_loss_weight",
            "recon_loss_weight",
            "real_loss",
            "imag_loss",
            "kl_loss",
            "channel_kl_loss",
            "latents_mean",
            "latents_std",
            "latents_logvar",
            "point_similarity",
            "point_loss_weight",
        ]

    elif args.module == "unet":
        logger.info("Training UNet model:")

        if args.num_timestep_loss_buckets > 0:
            logger.info(f"Using {args.num_timestep_loss_buckets} timestep loss buckets")

            timestep_loss_buckets = torch.zeros(args.num_timestep_loss_buckets,
                                                device="cpu", dtype=torch.float32)
            timestep_loss_bucket_counts = torch.zeros(args.num_timestep_loss_buckets,
                                                    device="cpu", dtype=torch.float32)
        else:
            logger.info("Timestep loss buckets are disabled")

        module_log_channels = []

        if vae is not None:
            if vae.config.last_global_step == 0:
                logger.error("VAE model has not been trained, aborting...")
                exit(1)

            latent_shape = vae.get_latent_shape(sample_shape)
            latent_mean = model_params["latent_mean"]
            latent_std = model_params["latent_std"]

        timestep_ln_center = model_params["timestep_ln_center"]
        timestep_ln_scale = model_params["timestep_ln_scale"]
        logger.info(f"Sampling timesteps using logistic normal distribution - center: {timestep_ln_center} - scale: {timestep_ln_scale}")

        input_perturbation = model_params["input_perturbation"]   
        if input_perturbation > 0:
            logger.info(f"Using input perturbation of {input_perturbation}")
        else:
            logger.info("Input perturbation is disabled")

    logger.info(f"Sample shape: {sample_shape}")
    if latent_shape is not None:
        logger.info(f"Latent shape: {latent_shape}")
        if args.module == "unet":
            logger.info(f"Latent mean: {latent_mean} - Latent std: {latent_std}")


    for epoch in range(first_epoch, args.num_train_epochs):

        torch.cuda.empty_cache()
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
            
            # this breaks reproducible training but is worth it to get more value out of the min-snr weighting with smaller batches
            if args.module == "unet" and grad_accum_steps == 0:
                total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

                # logit normal timestep sampling
                #batch_timesteps = torch.randn(total_batch_size, device=accelerator.device) * timestep_ln_scale + timestep_ln_center
                #batch_timesteps = torch.sigmoid(batch_timesteps) * 999.

                # uniform timestep sampling
                #batch_timesteps = torch.rand(total_batch_size, device=accelerator.device) * 999.

                # acos static timestep sampling
                batch_timesteps = (torch.arange(total_batch_size, device=accelerator.device)+0.5) / total_batch_size
                batch_timesteps += (torch.rand(1, device=accelerator.device) - 0.5) / total_batch_size
                batch_timesteps = ((1 - 2*batch_timesteps).acos() * (999. / torch.pi)).clip(min=0, max=999.)

                #batch_timesteps = torch.erfinv(batch_timesteps * 2 - 1).sigmoid() * 999.
                #batch_timesteps = batch_timesteps.clip(min=0, max=1) * 999.

                # sync timesteps across all ranks / processes
                batch_timesteps = accelerator.gather(batch_timesteps.unsqueeze(0))[0]

                if args.num_timestep_loss_buckets > 0:
                    timestep_loss_buckets.zero_()
                    timestep_loss_bucket_counts.zero_()

            with accelerator.accumulate(module):

                raw_samples = batch["input"]
                raw_sample_paths = batch["sample_paths"]

                if args.module == "unet":
                    samples = pipeline.format.raw_to_sample(raw_samples)
                
                    if vae is not None:
                        samples = vae.encode(samples.half(), return_dict=False)[0].mode().detach().float()
                        samples -= samples.mean(dim=(1,2,3), keepdim=True)
                        samples /= samples.std(dim=(1,2,3), keepdim=True)
                    
                    noise = torch.randn_like(samples).requires_grad_(False)
                    noise -= noise.mean(dim=(1,2,3), keepdim=True)
                    noise /= noise.std(dim=(1,2,3), keepdim=True)

                    process_batch_timesteps = batch_timesteps[accelerator.local_process_index::accelerator.num_processes]
                    timesteps = process_batch_timesteps[grad_accum_steps * args.train_batch_size:(grad_accum_steps+1) * args.train_batch_size]
                    
                    normalized_timesteps = (timesteps / 999.).view(-1, 1, 1, 1)
                    model_input = slerp(samples, noise, normalized_timesteps).detach()
                    #model_output = module(model_input, timesteps).sample
                    model_output, error_logvar = module(model_input, timesteps/999., None, pipeline.format, return_logvar=True)

                    target = slerp(samples, noise, normalized_timesteps - 1).detach() # geodesic flow with v pred
                    loss = F.mse_loss(model_output.float(), target.float(), reduction="none")
                    #loss = slerp_loss(model_output.float(), target.float()) / 14.5
                    timestep_loss = loss.mean(dim=list(range(1, len(loss.shape))))

                    loss = (timestep_loss / error_logvar.exp() + error_logvar).mean()

                    if args.num_timestep_loss_buckets > 0:
                        all_timesteps = accelerator.gather(timesteps.detach()).cpu()
                        all_timestep_loss = accelerator.gather(timestep_loss.detach()).cpu()
                        target_buckets = (all_timesteps / 1000. * timestep_loss_buckets.shape[0]).long()
                        timestep_loss_buckets.index_add_(0, target_buckets, all_timestep_loss)
                        timestep_loss_bucket_counts.index_add_(0, target_buckets, torch.ones_like(all_timestep_loss))
                    
                elif args.module == "vae":

                    samples_dict = pipeline.format.raw_to_sample(raw_samples, return_dict=True)
                    posterior = module.encode(samples_dict["samples"], return_dict=False)[0]
                    latents = posterior.sample()
                    latents_mean = latents.mean()
                    latents_std = latents.std()
                    latents_logvar = module.encoder.latents_logvar.detach()
                    model_output = module.decode(latents, return_dict=False)[0]

                    recon_samples_dict = pipeline.format.sample_to_raw(model_output, return_dict=True, decode=False)
                    point_similarity = (samples_dict["samples"] - recon_samples_dict["samples"]).abs().mean()
                    
                    real_loss, imag_loss = pipeline.format.get_loss(recon_samples_dict, samples_dict)
                    real_nll_loss = (real_loss / module.recon_loss_logvar.exp() + module.recon_loss_logvar) * recon_loss_weight
                    imag_nll_loss = (imag_loss / module.recon_loss_logvar.exp() + module.recon_loss_logvar) * recon_loss_weight

                    kl_loss = posterior.kl()

                    latents_channel_mean = latents.mean(dim=(2,3))
                    latents_channel_var = latents.std(dim=(2,3)).clip(min=1e-5).square()
                    channel_kl_loss = (latents_channel_mean.square() + latents_channel_var - 1 - latents_channel_var.log()).mean()
                    
                    loss = real_nll_loss + imag_nll_loss + kl_loss * kl_loss_weight + channel_kl_loss * channel_kl_loss_weight + point_similarity * point_loss_weight
                else:
                    raise ValueError(f"Unknown module {args.module}")
                
                # Gather the losses across all processes for logging (if we use distributed training).
                grad_accum_steps += 1
                avg_loss = accelerator.gather(loss).mean()
                train_loss += avg_loss.item()
                for channel in module_log_channels:
                    avg = accelerator.gather(locals()[channel]).mean()
                    module_logs[channel] += avg.item()

                # Backpropagate
                accelerator.backward(loss)

                if accelerator.sync_gradients: # clip and check for nan/inf grad
                    grad_norm = accelerator.clip_grad_norm_(module.parameters(), args.max_grad_norm).item()
                    if math.isinf(grad_norm) or math.isnan(grad_norm):
                        logger.warning(f"Warning: grad norm is {grad_norm} - step={global_step} loss={loss.item()} debug_last_sample_paths={raw_sample_paths}")
                        logger.info(f"Batch timesteps: {batch_timesteps}")
                    if math.isnan(grad_norm):
                        logger.error(f"Error: grad norm is {grad_norm}, aborting...")
                        import pdb; pdb.set_trace(); exit(1)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:

                progress_bar.update(1)
                global_step += 1

                if grad_accum_steps >= args.gradient_accumulation_steps: # don't log incomplete batches
                    logs = {"loss": train_loss / grad_accum_steps,
                            "lr": lr_scheduler.get_last_lr()[0],
                            "step": global_step,
                            "grad_norm": grad_norm}
                    for channel in module_log_channels:
                        logs[f"{args.module}/{channel}"] = module_logs[channel] / grad_accum_steps

                if args.use_ema:
                    ema_module.step(module.parameters())
                    logs["ema_decay"] = ema_module.cur_decay_value    

                if args.module == "unet" and args.num_timestep_loss_buckets > 0:
                    for i in range(timestep_loss_buckets.shape[0]):
                        if timestep_loss_bucket_counts[i].item() > 0:
                            bucket_timestep_min = int(i / timestep_loss_buckets.shape[0] * 1000.)
                            bucket_timestep_max = int((i+1) / timestep_loss_buckets.shape[0] * 1000.)
                            logs[f"unet/timestep_loss_{bucket_timestep_min}-{bucket_timestep_max}"] = (timestep_loss_buckets[i] / timestep_loss_bucket_counts[i]).item()

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
            
            """
            if args.module == "unet":
                timestep_error_logvar = getattr(pipeline.unet, "timestep_error_logvar", None)
                if timestep_error_logvar is not None:
                    debug_path = os.environ.get("DEBUG_PATH", None)
                    if debug_path is not None:
                        save_raw(timestep_error_logvar, os.path.join(debug_path, f"timestep_error_logvar_{global_step}.raw"))
            """

            # full model saving disabled for now, not really necessary now that checkpoints are saved with config and safetensors
            """
            if checkpoint_saved_this_epoch == True:
                logger.info(f"Saving model to {args.output_dir}")

                module = accelerator.unwrap_model(module)
                if args.use_ema:
                    ema_module.store(module.parameters())
                    ema_module.copy_to(module.parameters())

                setattr(pipeline, args.module, module)
                pipeline.save_pretrained(args.output_dir, safe_serialization=True)
            """
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

            #if args.use_ema:
            #    ema_module.restore(module.parameters())
            
    accelerator.end_training()

def log_validation_unet(pipeline, args, accelerator, global_step):

    sample_rate = pipeline.config["model_params"]["sample_rate"]
    if args.seed is None: seed = 42
    else: seed = args.seed

    output_path = os.path.join(args.output_dir, "output")
    os.makedirs(output_path, exist_ok=True)
    
    samples = []
    for i in range(args.num_validation_samples):
        logger.info(f"Generating sample {i+1}/{args.num_validation_samples}...")

        generator = torch.Generator(device=accelerator.device).manual_seed(seed)
        with torch.autocast("cuda"):
            sample = pipeline(steps=args.num_validation_steps,
                              seed=generator).cpu().squeeze(0)
            sample_filename = f"step_{global_step}_{args.num_validation_steps}_s{seed}.flac"
            samples.append((sample, sample_filename))

            sample_output_path = os.path.join(output_path, sample_filename)
            save_audio(sample, sample_rate, sample_output_path)
            logger.info(f"Saved sample to to {sample_output_path}")

        seed += 1

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for sample, sample_filename in samples:
                sample = normalize_lufs(sample.mean(dim=0), sample_rate).clip(min=-1, max=1)
                tracker.writer.add_audio(os.path.splitext(sample_filename)[0],
                                         sample,
                                         global_step,
                                         sample_rate=sample_rate)
        else:
            logger.warn(f"audio logging not implemented for {tracker.name}")

def log_validation_vae(pipeline, args, accelerator, global_step):

    raise NotImplementedError()

    sample_rate = pipeline.config["model_params"]["sample_rate"]
    if args.seed is None: seed = 42
    else: seed = args.seed

    output_path = os.path.join(args.output_dir, "output")
    os.makedirs(output_path, exist_ok=True)
    
    samples = []
    for i in range(args.num_validation_samples):
        logger.info(f"Generating sample {i+1}/{args.num_validation_samples}...")

        generator = torch.Generator(device=accelerator.device).manual_seed(seed)
        with torch.autocast("cuda"):
            sample = pipeline(steps=args.num_validation_steps,
                              seed=generator).cpu()
            sample_filename = f"step_{global_step}_{args.num_validation_steps}_s{seed}.flac"
            samples.append((sample, sample_filename))

            sample_output_path = os.path.join(output_path, sample_filename)
            torchaudio.save(sample_output_path, sample, sample_rate, bits_per_sample=16)
            logger.info(f"Saved sample to to {sample_output_path}")

        seed += 1

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for sample, sample_filename in samples:
                tracker.writer.add_audio(os.path.splitext(sample_filename)[0],
                                         sample,
                                         global_step,
                                         sample_rate=sample_rate)
        else:
            logger.warn(f"audio logging not implemented for {tracker.name}")

def main():
    args = parse_args()

    accelerator = init_accelerator(args.output_dir,
                                   args.gradient_accumulation_steps,
                                   args.mixed_precision,
                                   args.logging_dir,
                                   args.report_to,
                                   args.tracker_project_name)
    
    init_logging(accelerator, args.logging_dir, args.module, args.report_to)

    logger.info(accelerator.state, main_process_only=False, in_order=True)
    accelerator.wait_for_everyone()

    if args.seed is not None:
        set_seed(args.seed)
        logger.info(f"Using random seed {args.seed}")
    else:
        logger.info("Using random seed from system - Training may not be reproducible")

    if args.enable_anomaly_detection:
        torch.autograd.set_detect_anomaly(True)
        logger.info("Pytorch anomaly detection enabled")
    else:
        logger.info("Pytorch anomaly detection disabled")

    pipeline, module, module_class, vae = init_module_pipeline(args.pretrained_model_name_or_path,
                                                               args.module,
                                                               args.vae_encode_batch_size,
                                                               accelerator.device)

    if args.use_ema:
        ema_module = init_ema_module(args.pretrained_model_name_or_path,
                                     args.module,
                                     module_class,
                                     args.revision,
                                     args.ema_min_decay,
                                     args.ema_max_decay,
                                     args.ema_inv_gamma,
                                     args.ema_power,
                                     accelerator.device)
    else:
        ema_module = None
        logger.info("Not using EMA model")

    init_accelerator_loadsave_hooks(accelerator, args.module, module_class, ema_module)

    if args.gradient_checkpointing:
        module.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing enabled")
    else:
        logger.info("Gradient checkpointing disabled")

    optimizer = init_optimizer(args.use_8bit_adam,
                               args.learning_rate,
                               args.adam_beta1,
                               args.adam_beta2,
                               args.adam_weight_decay,
                               args.adam_epsilon,
                               module)

    train_dataset, train_dataloader = init_dataloader(accelerator,
                                                      args.dataset_name,
                                                      args.hf_token,
                                                      args.train_data_dir,
                                                      args.cache_dir,
                                                      args.train_batch_size,
                                                      args.dataloader_num_workers,
                                                      args.max_train_samples,
                                                      args.train_data_format,
                                                      args.train_data_raw_format,
                                                      args.train_data_num_channels,
                                                      pipeline.config["model_params"]["sample_rate"],
                                                      pipeline.format.get_sample_crop_width())

    num_update_steps_per_epoch = math.floor(len(train_dataloader) / accelerator.num_processes)
    num_update_steps_per_epoch = math.ceil(num_update_steps_per_epoch / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    if args.checkpointing_steps is None: args.checkpointing_steps = num_update_steps_per_epoch*3
    logger.info(f"Saving checkpoints every {args.checkpointing_steps} steps")
    
    lr_scheduler = init_lr_scheduler(args.lr_scheduler,
                                     optimizer,
                                     args.lr_warmup_steps,
                                     max_train_steps,
                                     accelerator.num_processes)
    
    logger.info(f"Using learning rate schedule {args.lr_scheduler} with warmup steps {args.lr_warmup_steps}", main_process_only=True)

    module, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        module, optimizer, train_dataloader, lr_scheduler
    )

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    logger.info(f"  Path to save/load checkpoints = {args.output_dir}")

    global_step, resume_step, first_epoch = load_checkpoint(args.resume_from_checkpoint,
                                                            args.output_dir,
                                                            args.module,
                                                            accelerator,
                                                            optimizer,
                                                            lr_scheduler,
                                                            args.learning_rate,
                                                            args.adam_weight_decay,
                                                            args.gradient_accumulation_steps,
                                                            num_update_steps_per_epoch)
    
    do_training_loop(args,
                     accelerator,
                     module,
                     ema_module,
                     pipeline,
                     vae,
                     lr_scheduler,
                     optimizer,
                     first_epoch,
                     global_step,
                     resume_step,
                     num_update_steps_per_epoch,
                     max_train_steps,
                     train_dataloader)
    
if __name__ == "__main__":

    init_cuda()
    load_dotenv(override=True)
    main()