from typing import Literal, Union

import numpy as np
import torch

from diffusers.schedulers import DPMSolverMultistepScheduler, DDIMScheduler
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from unet2d_dual import UNet2DDualModel
from unet1d_dual import UNet1DDualModel
class DualDiffusionPipeline(DiffusionPipeline):

    def __init__(
        self,
        unet: UNet2DDualModel,
        scheduler: DDIMScheduler,
        model_params: dict = None,
    ):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        
        if model_params is not None:
            self.config["model_params"] = model_params

    @staticmethod
    def create_new(model_params, save_model_path):
        
        num_input_channels, num_output_channels = DualDiffusionPipeline.get_num_channels(model_params)

        unet = UNet2DDualModel(
            act_fn="silu",
            attention_head_dim=8,
            #attention_head_dim=32,
            #attention_head_dim=(16, 32, 64, 128),
            #attention_head_dim=(8, 16, 32, 64),
            #separate_attn_dim=(3,2),
            separate_attn_dim=3,
            reverse_separate_attn_dim=False,
            #separate_attn_dim=(3,2),
            #reverse_separate_attn_dim=True,
            center_input_sample=False,
            downsample_padding=1,
            flip_sin_to_cos=True,
            freq_shift=0,
            mid_block_scale_factor=1,
            norm_eps=1e-05,
            norm_num_groups=32,
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 512, 1024), #, 1024),
            down_block_types=(
                "SeparableAttnDownBlock2D",
                "SeparableAttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                #"SeparableAttnDownBlock2D",
                #"AttnDownBlock2D",
            ),
            up_block_types=(
                #"AttnUpBlock2D",
                #"SeparableAttnUpBlock2D",
                #"SeparableAttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "SeparableAttnUpBlock2D",
                "SeparableAttnUpBlock2D",
            ),
            #downsample_type="resnet",
            #upsample_type="resnet",
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

        pipeline = DualDiffusionPipeline(unet, scheduler, model_params)
        pipeline.save_pretrained(save_model_path, safe_serialization=True)
        return pipeline

    @staticmethod
    def get_sample_crop_width(model_params):
        return model_params["sample_raw_length"]
    
    @staticmethod
    def get_num_channels(model_params):
        format = model_params["format"]
        freq_embedding_dim = model_params["freq_embedding_dim"]
        if format == "complex":
            return (1 + freq_embedding_dim, 1)
        elif format == "complex_2channels":
            return (2 + freq_embedding_dim, 2)
        elif format == "complex_1channel":
            return (1 + freq_embedding_dim, 1)
        else:
            raise ValueError(f"Unknown format '{format}'")

    @staticmethod
    def get_window(window_len):
        x = torch.arange(0, window_len, device="cuda") + 0.5
        return (torch.ones(window_len, device="cuda") + torch.cos(x / window_len * 2. * np.pi - np.pi)) * 0.5

    @staticmethod
    @torch.no_grad()
    def get_positional_embedding(positions, embedding_dim):
        positions = positions.unsqueeze(0)
        indices = (torch.arange(0, embedding_dim, step=2, device=positions.device) / embedding_dim).unsqueeze(1)
        return torch.cat((torch.sin(positions / (10000. ** indices)), torch.cos(positions / (10000. ** indices))), dim=0)
    
    @staticmethod
    @torch.no_grad()
    def add_freq_embedding(freq_samples, freq_embedding_dim):
        if freq_embedding_dim == 0: return freq_samples

        ln_freqs = ((torch.arange(0, freq_samples.shape[2], device=freq_samples.device) + 0.5) / freq_samples.shape[2]).log_()
        
        if freq_embedding_dim > 1:
            ln_freqs *= freq_samples.shape[2] / ln_freqs[0].item()
            freq_embeddings = DualDiffusionPipeline.get_positional_embedding(ln_freqs, freq_embedding_dim).type(freq_samples.dtype)
            freq_embeddings = freq_embeddings.view(1, freq_embedding_dim, freq_samples.shape[2], 1).repeat(freq_samples.shape[0], 1, 1, freq_samples.shape[3])
        else:
            ln_freqs /= ln_freqs[0].item()
            ln_freqs = ln_freqs.type(freq_samples.dtype)
            freq_embeddings = ln_freqs.view(1, 1, freq_samples.shape[2], 1).repeat(freq_samples.shape[0], 1, 1, freq_samples.shape[3])

        return torch.cat((freq_samples, freq_embeddings), dim=1)

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
            spatial_window = DualDiffusionPipeline.get_window(spatial_window_len).square_()
            raw_samples[:, :spatial_window_len//2]  *= spatial_window[:spatial_window_len//2]
            raw_samples[:, -spatial_window_len//2:] *= spatial_window[-spatial_window_len//2:]

        fft_samples_positive_frequencies_only = torch.fft.fft(raw_samples, norm="ortho")[:, :raw_samples.shape[1]//2]
        #fft_samples_positive_frequencies_only[:, 0] = 0. # explicitly remove dc component, otherwise the first chunk has significant non-zero mean
        num_chunks = model_params["num_chunks"]
        chunk_len = fft_samples_positive_frequencies_only.shape[1] // num_chunks
        fft_samples_chunks = fft_samples_positive_frequencies_only.view(fft_samples_positive_frequencies_only.shape[0], -1, chunk_len)
        fft_chunk_ffts = torch.fft.fft(fft_samples_chunks, norm="ortho")

        format = model_params["format"] if format_override is None else format_override
        if format == "complex":
            freq_samples = fft_chunk_ffts.unsqueeze(1)
            freq_samples /= fft_chunk_ffts.std(dim=(1, 2, 3), keepdim=True)
        elif format == "complex_2channels":
            freq_samples = torch.view_as_real(fft_chunk_ffts).permute(0, 3, 1, 2).contiguous()
            freq_samples /= freq_samples.std(dim=(1, 2, 3), keepdim=True)
        elif format == "complex_1channel":
            freq_samples = torch.view_as_real(fft_chunk_ffts).view(fft_chunk_ffts.shape[0], fft_chunk_ffts.shape[1], -1).unsqueeze(1)
            freq_samples /= freq_samples.std(dim=(1, 2, 3), keepdim=True)
        else:
            raise ValueError(f"Unknown format '{format}'")

        return freq_samples
    
    @staticmethod
    @torch.no_grad()
    def sample_to_raw(freq_samples, model_params):
        
        format = model_params["format"]
        if format == "complex_1channel":
            fft_samples_chunks = torch.view_as_complex(freq_samples.view(freq_samples.shape[0], freq_samples.shape[2], -1, 2))
        elif format == "complex_2channels":
            fft_samples_chunks = torch.view_as_complex(freq_samples.permute(0, 2, 3, 1).contiguous())
        else:
            raise ValueError(f"Unknown format '{format}'")
        
        fft_samples = torch.fft.ifft(fft_samples_chunks, norm="ortho")
        fft_samples = fft_samples.view(fft_samples.shape[0], -1)

        fft_samples = torch.cat((fft_samples, torch.zeros_like(fft_samples)), dim=1)
        raw_samples = torch.fft.ifft(fft_samples, norm="ortho")
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

        if loops > 0: self.set_tiling_mode("x")
        else: self.set_tiling_mode(False)

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
        sample_crop_width = DualDiffusionPipeline.get_sample_crop_width(model_params)
        num_chunks = model_params["num_chunks"]
        default_length = (sample_crop_width // 2) // num_chunks
        num_input_channels, num_output_channels = DualDiffusionPipeline.get_num_channels(model_params)

        noise = torch.randn((batch_size, num_output_channels, num_chunks, default_length*length,),
                            device=self.device,
                            generator=generator)
        sample = noise #; print(f"Sample shape: {sample.shape}")
        freq_embedding_dim = model_params["freq_embedding_dim"]
        
        for step, t in enumerate(self.progress_bar(timesteps)):
            
            model_input = sample
            if freq_embedding_dim > 0:
                model_input = DualDiffusionPipeline.add_freq_embedding(model_input, freq_embedding_dim)
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
                sample -= sample.mean(dim=(1,2,3), keepdim=True)
            if renormalize_sample:
                sample /= sample.std(dim=(1,2,3), keepdim=True)

        print("Sample std: ", sample.std(dim=(1,2,3)).item())

        #sample = sample.type(torch.float32)
        #sample.cpu().numpy().tofile("./output/debug_sample.raw")

        raw_sample = DualDiffusionPipeline.sample_to_raw(sample, model_params)
        if loops > 0: raw_sample = raw_sample.repeat(1, loops+1)
        else: self.set_tiling_mode(False)

        return raw_sample
    
    def set_module_tiling(self, module, tiling):
        F, _pair = torch.nn.functional, torch.nn.modules.utils._pair

        padding_modeX = "circular" if tiling != "y" else "constant"
        padding_modeY = "circular" if tiling != "x" else "constant"

        rprt = module._reversed_padding_repeated_twice
        paddingX = (rprt[0], rprt[1], 0, 0)
        paddingY = (0, 0, rprt[2], rprt[3])

        def _conv_forward(self, input, weight, bias):
            padded = F.pad(input, paddingX, mode=padding_modeX)
            padded = F.pad(padded, paddingY, mode=padding_modeY)
            return F.conv2d(
                padded, weight, bias, self.stride, _pair(0), self.dilation, self.groups
            )

        module._conv_forward = _conv_forward.__get__(module)

    def remove_module_tiling(self, module):
        try:
            del module._conv_forward
        except AttributeError:
            pass

    def set_tiling_mode(self, tiling: bool | Literal["x", "y", "xy"] = True):

        module_names, _, _ = self.extract_init_dict(dict(self.config))
        #modules = [getattr(self, name) for name in module_names.keys()]
        modules = [self.unet]
        modules = filter(lambda module: isinstance(module, torch.nn.Module), modules)

        for module in modules:
            for submodule in module.modules():
                if isinstance(submodule, torch.nn.Conv2d | torch.nn.ConvTranspose2d):
                    if isinstance(submodule, torch.nn.ConvTranspose2d):
                        raise NotImplementedError(
                            "Assymetric tiling doesn't support this module"
                        )

                    if tiling is False:
                        self.remove_module_tiling(submodule)
                    else:
                        self.set_module_tiling(submodule, tiling)




# "img2img" test
"""
crop_width = LGDiffusionPipeline.get_sample_crop_width(model_params)
raw_sample = np.fromfile("./dataset/samples/600.raw", dtype=np.int16, count=crop_width) / 32768.
raw_sample = torch.from_numpy(raw_sample).unsqueeze(0).to("cuda").type(torch.float32)
freq_sample = LGDiffusionPipeline.raw_to_freq(raw_sample, model_params).type(torch.float16)
strength = 0.75
#sample = (freq_sample.type(torch.float16) * (1 - strength) + noise * strength) / 2#4
step_ratio = int(noise_scheduler.config.num_train_timesteps * strength) // steps
timesteps = torch.from_numpy((np.arange(1, steps+1) * step_ratio).round()[::-1].copy().astype(np.int64))
sample = noise_scheduler.add_noise(freq_sample, noise, timesteps[1])
"""