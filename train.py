#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import os
import random
import shutil
import subprocess
import atexit
from pathlib import Path
from datetime import datetime
from glob import glob

import ffmpeg

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset, Audio
from packaging import version
from tqdm.auto import tqdm

import diffusers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_tensorboard_available
from diffusers.utils.import_utils import is_xformers_available

from unet2d_dual import UNet2DDualModel
from dual_diffusion_pipeline import DualDiffusionPipeline

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
#check_min_version("0.21.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def log_validation(pipeline, args, accelerator, weight_dtype, global_step):
    logger.info("Running validation... ")

    pipeline.set_progress_bar_config(disable=True)

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
                              seed=generator,
                              scheduler=args.validation_scheduler)
            samples.append(sample)

            sample_output_path = os.path.join(output_path, f"step_{global_step}_{args.validation_scheduler}{args.num_validation_steps}_s{seed}.raw")
            sample.cpu().numpy().tofile(sample_output_path)
            output_flac_file = os.path.splitext(sample_output_path)[0] + '.flac'
            ffmpeg.input(sample_output_path, f="f32le", ac=2, ar=sample_rate).output(output_flac_file).run(quiet=True)
            logger.info(f"Saved sample to to {output_flac_file}")
            os.remove(sample_output_path)

        seed += 1

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            i = 0
            for sample in samples:
                tracker.writer.add_audio(f"step_{global_step}__{i}", sample.real.type(torch.float32).cpu(), global_step, sample_rate=sample_rate)
                i += 1
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="DualDiffusion training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
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
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
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
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
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
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
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
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
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
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
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
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--num_validation_samples",
        type=int,
        default=5,
        help="Number of samples to generate for validation.",
    )
    parser.add_argument(
        "--num_validation_steps",
        type=int,
        default=100,
        help="Number of steps to use when creating validation samples.",
    )
    parser.add_argument(
        "--validation_scheduler",
        type=str,
        default="dpms++",
        help="The scheduler type to use for creating validation samples.",
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

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def main():
    args = parse_args()

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.report_to == "tensorboard":
        if not is_tensorboard_available():
            raise ImportError("Make sure to install tensorboard if you want to use it for logging during training.")
        port = 6006
        tensorboard_monitor_process = subprocess.Popen(['tensorboard', '--logdir', logging_dir, '--bind_all', '--port', f'{port}'])

        def cleanup_process():
            try:
                tensorboard_monitor_process.terminate()
            except Exception:
                logger.warn("Failed to terminate tensorboard process")
                pass

        atexit.register(cleanup_process)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Initialize the model
    pipeline = DualDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path)#.to("cuda")
    unet = pipeline.unet
    noise_scheduler = pipeline.scheduler
    model_params = pipeline.config["model_params"]
    sample_crop_width = DualDiffusionPipeline.get_sample_crop_width(model_params)
    freq_embedding_dim = model_params["freq_embedding_dim"]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = UNet2DDualModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
        )
        ema_unet = EMAModel(ema_unet.parameters(),
                            model_cls=UNet2DDualModel,
                            model_config=ema_unet.config,
                            min_decay=args.ema_min_decay,
                            decay=args.ema_max_decay,
                            inv_gamma=args.ema_inv_gamma,
                            power=args.ema_power)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            pipeline.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    def compute_snr(timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                if not os.path.exists(os.path.join(input_dir, "unet_ema")):
                    logger.info("EMA model in checkpoint not found, using new ema model")
                else:
                    load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DDualModel)
                    ema_unet.load_state_dict(load_model.state_dict())
                    ema_unet.to(accelerator.device)
                    del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DDualModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            data_dir=args.train_data_dir,
        ).cast_column("audio", Audio(decode=False))
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "audiofolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        ).cast_column("audio", Audio(decode=False))

    logger.info(f"Dataset size: {len(dataset)}")

    def transform_samples(examples):
        
        samples = []
        for audio in examples["audio"]:

            if audio["bytes"] is not None:
                sample_len = len(audio["bytes"]) // 2
                crop_offset = np.random.randint(0, sample_len - sample_crop_width)
                sample = np.frombuffer(audio["bytes"], dtype=np.int16, count=sample_crop_width, offset=crop_offset * 2)
            else:
                sample_len = os.path.getsize(audio["path"]) // 2
                crop_offset = np.random.randint(0, sample_len - sample_crop_width)
                sample = np.fromfile(audio["path"], dtype=np.int16, count=sample_crop_width, offset=crop_offset * 2)

            sample = sample.astype(np.float32) / 32768.
            samples.append(torch.from_numpy(sample).to("cuda"))

        return {"input": samples}

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(transform_samples)

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        if args.tracker_project_name is None:
            args.tracker_project_name = os.path.basename(args.output_dir)
        start_timestamp = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        args.tracker_project_name = f"{args.tracker_project_name}_{start_timestamp}"

        """
        tracker_config = dict(vars(args))
        for k, v in unet.config.items():
            if not isinstance(v, (int, float, str, bool)): v = f"{v}"
            tracker_config[f"unet_{k}"] = v

        for k, v in noise_scheduler.config.items():
            if not isinstance(v, (int, float, str, bool)): v = f"{v}"
            tracker_config[f"scheduler_{k}"] = v

        for k, v in model_params.items():
            if not isinstance(v, (int, float, str, bool)): v = f"{v}"
            tracker_config[f"model_{k}"] = v
        """
        tracker_config = None
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

        # copy all source code / scripts to model folder for posterity
        
        source_src_path = os.path.dirname(__file__)
        target_src_path = os.path.join(args.output_dir, "src")
        logger.info(f"Copying source code at '{source_src_path}' to model folder '{target_src_path}'")

        os.makedirs(target_src_path, exist_ok=True)
        src_file_types = ["py", "cmd", "yml", "sh"]
        src_files = []
        for file_type in src_file_types:
            src_files += glob(f"*.{file_type}")
        for src_file in src_files:
            shutil.copy(src_file, os.path.join(target_src_path, os.path.basename(src_file)))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0
    grad_norm = 0.
    debug_written = False

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0

        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        checkpoint_saved_this_epoch = False
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):

                raw_samples = batch["input"]
                samples = DualDiffusionPipeline.raw_to_freq(raw_samples, model_params)

                noise = torch.randn_like(samples)
                if args.input_perturbation > 0:
                    new_noise = noise + args.input_perturbation * torch.randn_like(noise)

                if not debug_written:
                    logger.info(f"Samples mean: {samples.mean()} - Samples std: {samples.std()}")
                    logger.info(f"Samples shape: {samples.shape} - Samples std (with noise): {(samples + noise).std()}")
                    samples.cpu().numpy().tofile("output/debug_samples.raw")
                    raw_samples.cpu().numpy().tofile("output/debug_raw_samples.raw")
                    #raw_samples = LGDiffusionPipeline.freq_to_raw(samples)
                    #raw_samples.cpu().numpy().tofile("output/debug_reconstructed_raw_samples.raw")
                    debug_written = True
                    
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (samples.shape[0],), device=samples.device).long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if args.input_perturbation > 0:
                    model_input = noise_scheduler.add_noise(samples, new_noise, timesteps)
                else:
                    model_input = noise_scheduler.add_noise(samples, noise, timesteps)
                if freq_embedding_dim > 0:
                    model_input = DualDiffusionPipeline.add_freq_embedding(model_input, freq_embedding_dim)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(samples, noise, timesteps)
                elif noise_scheduler.config.prediction_type == "sample":
                    target = samples
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the target and compute loss
                model_output = unet(model_input, timesteps).sample

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_output.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(timesteps)
                    if noise_scheduler.config.prediction_type == "v_prediction": # correction as per the above paper
                        snr += 1.                                                # for v-prediction

                    mse_loss_weights = (
                        torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )
                    # We first calculate the original loss. Then we mean over the non-batch dimensions and
                    # rebalance the sample-wise losses with their respective loss weights.
                    # Finally, we take the mean of the rebalanced loss.
                    loss = F.mse_loss(model_output.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm).item()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1

                logs = {"loss": train_loss,
                        "lr": lr_scheduler.get_last_lr()[0],
                        "step": global_step,
                        "grad_norm": grad_norm}
                if args.use_ema:
                    logs["ema_decay"] = ema_unet.cur_decay_value    
                accelerator.log(logs, step=global_step)
                progress_bar.set_postfix(**logs)

                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        checkpoint_saved_this_epoch = True

            if global_step >= args.max_train_steps:
                break
        
        progress_bar.close()

        # Create the pipeline using the trained modules and save it.
        accelerator.wait_for_everyone()
        if accelerator.is_main_process and checkpoint_saved_this_epoch:
            logger.info(f"Saving model to {args.output_dir}")

            unet = accelerator.unwrap_model(unet)
            if args.use_ema:
                ema_unet.store(unet.parameters())
                ema_unet.copy_to(unet.parameters())

            pipeline.unet = unet
            pipeline.config["model_params"]["last_global_step"] = global_step
            pipeline.save_pretrained(args.output_dir, safe_serialization=True)

            if args.num_validation_samples > 0:
                try:
                    log_validation(
                        pipeline,
                        args,
                        accelerator,
                        weight_dtype,
                        global_step,
                    )
                except Exception as e:
                    logger.error(f"Error running validation: {e}")
                    pipeline.set_tiling_mode(False)
                    
            if args.use_ema:
                ema_unet.restore(unet.parameters())
                
    accelerator.end_training()


if __name__ == "__main__":

    if torch.cuda.is_available():
        torch.backends.cuda.cufft_plan_cache[0].max_size = 8 # stupid cufft memory leak
    else:
        print("Error: PyTorch not compiled with CUDA support or CUDA unavailable")
        exit(1)

    main()
