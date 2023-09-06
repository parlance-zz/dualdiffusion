from typing import Literal, Union

import numpy as np
import torch

from diffusers.schedulers import DPMSolverMultistepScheduler, DDIMScheduler
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from unet1d_dual import UNet1DDualModel

class DualDiffusionPipeline1D(DiffusionPipeline):

    def __init__(
        self,
        unet: UNet1DDualModel,
        scheduler: DDIMScheduler,
        model_params: dict = None,
    ):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        
        if model_params is not None:
            self.config["model_params"] = model_params

    @staticmethod
    def create_new(model_params, save_model_path):
        
        num_input_channels, num_output_channels = DualDiffusionPipeline1D.get_num_channels(model_params)

        unet = UNet1DDualModel(
            act_fn="silu",
            attention_head_dim=8,
            flip_sin_to_cos=True,
            freq_shift=0,
            mid_block_scale_factor=1,
            norm_eps=1e-05,
            norm_num_groups=32,
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            layers_per_block=2,
            block_out_channels=(32, 64, 96, 128, 160, 192, 224, 256),
            down_block_types=(
                "DualDownBlock1D",
                "DualDownBlock1D",
                "DualDownBlock1D",
                "DualDownBlock1D",
                "DualDownBlock1D",
                "DualDownBlock1D",
                "DualDownBlock1D",
                "DualDownBlock1D",
            ),
            up_block_types=(
                "DualUpBlock1D",
                "DualUpBlock1D",
                "DualUpBlock1D",
                "DualUpBlock1D",
                "DualUpBlock1D",
                "DualUpBlock1D",
                "DualUpBlock1D",
                "DualUpBlock1D",
            ),
        )

        beta_schedule = model_params["beta_schedule"]
        beta_start = model_params["beta_start"]
        beta_end = model_params["beta_end"]
        prediction_type = model_params["prediction_type"]
        scheduler = DDIMScheduler(clip_sample_range=20.,
                                  prediction_type=prediction_type,
                                  beta_schedule=beta_schedule,
                                  beta_start=beta_start,
                                  beta_end=beta_end)

        pipeline = DualDiffusionPipeline1D(unet, scheduler, model_params)
        pipeline.save_pretrained(save_model_path, safe_serialization=True)
        return pipeline

    @staticmethod
    def get_sample_crop_width(model_params):
        return model_params["sample_raw_length"]
    
    @staticmethod
    def get_num_channels(model_params):
        return (2, 2)

    @staticmethod
    def get_window(window_len):
        x = torch.arange(0, window_len, device="cuda") + 0.5
        return (torch.ones(window_len, device="cuda") + torch.cos(x / window_len * 2. * np.pi - np.pi)) * 0.5

    @staticmethod
    @torch.no_grad()
    def raw_to_sample(raw_samples, model_params, format_override=None):

        raw_samples = raw_samples.clone()
        raw_samples /= raw_samples.std(dim=1, keepdim=True)

        noise_floor = model_params["noise_floor"]
        if noise_floor > 0:
            raw_samples += torch.randn_like(raw_samples) * noise_floor

        spatial_window_len = model_params["spatial_window_length"]
        if spatial_window_len > 0:
            spatial_window = DualDiffusionPipeline1D.get_window(spatial_window_len).square_()
            raw_samples[:, :spatial_window_len//2]  *= spatial_window[:spatial_window_len//2]
            raw_samples[:, -spatial_window_len//2:] *= spatial_window[-spatial_window_len//2:]

        fft_samples = torch.fft.fft(raw_samples, norm="ortho")
        fft_samples[:, raw_samples.shape[1]//2:] = 0.
        ifft_samples = torch.fft.ifft(fft_samples, norm="ortho")

        format = model_params["format"] if format_override is None else format_override
        if format == "complex":
            spatial_samples = ifft_samples.unsqueeze(1)
            spatial_samples /= spatial_samples.std(dim=(1, 2), keepdim=True)
        elif format == "complex_2channels":
            spatial_samples = torch.view_as_real(ifft_samples).permute(0, 2, 1).contiguous()
            spatial_samples /= spatial_samples.std(dim=(1, 2), keepdim=True)
        else:
            raise ValueError(f"Unknown format '{format}'")

        return spatial_samples
    
    @staticmethod
    @torch.no_grad()
    def sample_to_raw(spatial_samples, model_params):
        
        format = model_params["format"]
        if format == "complex_2channels":
            raw_samples = torch.view_as_complex(spatial_samples.permute(0, 2, 1).contiguous())
        else:
            raise ValueError(f"Unknown format '{format}'")
        
        return raw_samples / raw_samples.std(dim=1, keepdim=True) * 0.18215

    @staticmethod
    @torch.no_grad()
    def raw_to_log_scale(samples, u=255.):
        return torch.sgn(samples) * torch.log(1. + 255 * samples.abs()) / torch.log(1 + u)

    @staticmethod
    @torch.no_grad() 
    def log_scale_to_raw(samples, u=255.):
        return torch.sgn(samples) * ((1 + u) ** samples.abs() - 1) / u

    @torch.no_grad()
    def __call__(
        self,
        steps: int = 100,
        scheduler="dpms++",
        seed: Union[int, torch.Generator]=None,
        loops: int = 0,
        batch_size: int = 1,
        length: int = 1,
        renormalize_sample: bool = False,
        rebalance_mean: bool = False,
    ):
        if (steps <= 0) or (steps > 1000):
            raise ValueError(f"Steps must be between 1 and 1000, got {steps}")
        if loops < 0:
            raise ValueError(f"Loops must be greater than or equal to 0, got {loops}")
        if length <= 0:
            raise ValueError(f"Length must be greater than or equal to 1, got {length}")

        #if loops > 0: self.set_tiling_mode("x")
        #else: self.set_tiling_mode(False)

        if scheduler == "ddim":
            noise_scheduler = self.scheduler
        elif scheduler == "dpms++":
            prediction_type = self.scheduler.config["prediction_type"]
            beta_schedule = self.scheduler.config["beta_schedule"]
            noise_scheduler = DPMSolverMultistepScheduler(prediction_type=prediction_type, solver_order=3, beta_schedule=beta_schedule)
        else:
            raise ValueError(f"Unknown scheduler '{scheduler}'")
        noise_scheduler.set_timesteps(steps)
        timesteps = noise_scheduler.timesteps

        if isinstance(seed, int):
            if seed == 0: seed = np.random.randint(100000,999999)
            generator = torch.Generator(device=self.device).manual_seed(seed)
        elif isinstance(seed, torch.Generator):
            generator = seed

        model_params = self.config["model_params"]
        sample_crop_width = DualDiffusionPipeline1D.get_sample_crop_width(model_params)
        num_input_channels, num_output_channels = DualDiffusionPipeline1D.get_num_channels(model_params)

        noise = torch.randn((batch_size, num_output_channels, sample_crop_width,),
                            device=self.device,
                            generator=generator)
        sample = noise #; print(f"Sample shape: {sample.shape}")

        for step, t in enumerate(self.progress_bar(timesteps)):
            
            model_input = sample
            model_input = noise_scheduler.scale_model_input(model_input, t)
            model_output = self.unet(model_input, t).sample
            
            #if step == 0:
            #    print(f"Model output shape: {model_output.shape}")
            #    model_output.float().cpu().numpy().tofile("./output/debug_model_output.raw")

            sample = noise_scheduler.step(
                model_output=model_output,
                timestep=t,
                sample=sample,
                generator=generator,
            )["prev_sample"]

            if rebalance_mean:
                sample -= sample.mean(dim=(1,2), keepdim=True)
            if renormalize_sample:
                sample /= sample.std(dim=(1,2), keepdim=True)

        print("Sample std: ", sample.std(dim=(1,2)).item())

        #sample = sample.type(torch.float32)
        #sample.cpu().numpy().tofile("./output/debug_sample.raw")

        raw_sample = DualDiffusionPipeline1D.sample_to_raw(sample, model_params)
        #if loops > 0: raw_sample = raw_sample.repeat(1, loops+1)
        #else: self.set_tiling_mode(False)

        return raw_sample
    
    def set_module_tiling(self, module, tiling):

        F = torch.nn.functional
        padding_modeX = "circular" if tiling == "x" else "constant"

        def _conv_forward(self, input, weight, bias):
            
            if padding_modeX == "circular":
                padded_shape = list(input.shape)
                padded_shape[-1] += 2
                padded = torch.zeros(padded_shape, device=input.device, dtype=input.dtype)
                padded[..., 1:-1] = input
                padded[..., 0] = input[..., -1]
                padded[..., -1] = input[..., 0]
            else:
                padded = F.pad(input, pad=(1,1), mode="constant")

            return F.conv1d(
                padded, weight, bias, self.stride, 0, self.dilation, self.groups
            )

        module._conv_forward = _conv_forward.__get__(module)

    def remove_module_tiling(self, module):
        try:
            del module._conv_forward
        except AttributeError:
            pass

    def set_tiling_mode(self, tiling: bool | Literal["x"] = True):

        module_names, _, _ = self.extract_init_dict(dict(self.config))
        #modules = [getattr(self, name) for name in module_names.keys()]
        modules = [self.unet]
        modules = filter(lambda module: isinstance(module, torch.nn.Module), modules)

        for module in modules:
            for submodule in module.modules():
                if isinstance(submodule, torch.nn.Conv1d | torch.nn.ConvTranspose1d):
                    if isinstance(submodule, torch.nn.ConvTranspose1d):
                        raise NotImplementedError(
                            "Assymetric tiling doesn't support this module"
                        )

                    if tiling is False:
                        self.remove_module_tiling(submodule)
                    else:
                        self.set_module_tiling(submodule, tiling)