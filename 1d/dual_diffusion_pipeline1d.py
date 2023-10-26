from typing import Literal, Union

import numpy as np
import torch

from diffusers.schedulers import DPMSolverMultistepScheduler, DDIMScheduler
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from unet1d_dual import UNet1DDualModel

def raw_to_log_scale(samples, u=255.):
    return torch.sgn(samples) * torch.log(1. + u * samples.abs()) / np.log(1. + u)

def log_scale_to_raw(samples, u=255.):
    return torch.sgn(samples) * ((1. + u) ** samples.abs() - 1.) / u

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
            #act_fn="silu",
            act_fn="mish",
            #act_fn="relu",
            attention_head_dim=32,
            flip_sin_to_cos=True,
            freq_shift=0,
            mid_block_scale_factor=1,
            norm_eps=1e-05,
            norm_num_groups=32,
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            #layers_per_block=2,
            layers_per_block=1,
            conv_size=3,
            #block_out_channels=(384, 544, 768, 1088, 1536),
            block_out_channels=(512, 736, 1024, 1440, 2048),
            add_attention=True,
            use_fft=False,
            #upsample_type="resnet",
            #downsample_type="resnet",
            upsample_type="kernel",
            downsample_type="kernel",
            dropout=0.3,
            down_block_types=(
                "DualAttnDownBlock1D",
                "DualAttnDownBlock1D",
                "DualAttnDownBlock1D",
                "DualAttnDownBlock1D",
                "DualAttnDownBlock1D",
            ),
            up_block_types=(
                "DualAttnUpBlock1D",
                "DualAttnUpBlock1D",
                "DualAttnUpBlock1D",
                "DualAttnUpBlock1D",
                "DualAttnUpBlock1D",
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
        num_chunks = model_params["num_chunks"]
        #return (num_chunks * 2, num_chunks * 2)
        return (num_chunks, num_chunks)

    @staticmethod
    def get_window(window_len):
        x = torch.arange(0, window_len, device="cuda") + 0.5
        return (torch.ones(window_len, device="cuda") + torch.cos(x / window_len * 2. * np.pi - np.pi)) * 0.5

    @staticmethod
    @torch.no_grad()
    def raw_to_sample(raw_samples, model_params):

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

        fft_samples_positive_frequencies_only = torch.fft.fft(raw_samples, norm="ortho")[:, :raw_samples.shape[1]//2]
        fft_samples_positive_frequencies_only[:, 0] = 0. # explicitly remove dc component, otherwise the first chunk has significant non-zero mean
        num_chunks = model_params["num_chunks"]
        chunk_len = fft_samples_positive_frequencies_only.shape[1] // num_chunks
        fft_samples_chunks = fft_samples_positive_frequencies_only.view(fft_samples_positive_frequencies_only.shape[0], -1, chunk_len)
        fft_chunk_ffts = torch.fft.fft(fft_samples_chunks, norm="ortho")

        spatial_samples = torch.view_as_real(fft_chunk_ffts)
        #spatial_samples = spatial_samples.permute(0, 1, 3, 2).contiguous()
        #spatial_samples = spatial_samples.view(spatial_samples.shape[0], spatial_samples.shape[1]*2, chunk_len)
        spatial_samples = spatial_samples.view(spatial_samples.shape[0], spatial_samples.shape[1], chunk_len*2)

        spatial_samples /= spatial_samples.std(dim=(1, 2), keepdim=True)
        return spatial_samples
    
    @staticmethod
    @torch.no_grad()
    def sample_to_raw(spatial_samples, model_params):
        
        spatial_samples = spatial_samples.squeeze(1)
        return spatial_samples / spatial_samples.std(dim=1, keepdim=True) * 0.18215
    
        format = model_params["format"]
        if format == "complex_2channels":
            raw_samples = torch.view_as_complex(spatial_samples.permute(0, 2, 1).contiguous())
        else:
            raise ValueError(f"Unknown format '{format}'")
        
        #avg_std = model_params["avg_std"]
        #raw_samples = log_scale_to_raw(raw_samples * avg_std)
        return raw_samples / raw_samples.std(dim=1, keepdim=True) * 0.18215

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

        noise = torch.randn((batch_size, num_output_channels, sample_crop_width // num_output_channels,),
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

        sample = sample.type(torch.float32)
        sample.cpu().numpy().tofile("./output/debug_sample.raw")

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