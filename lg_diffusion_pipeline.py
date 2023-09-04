from typing import Tuple, Literal

import numpy as np
import torch

from diffusers.models import UNet2DModel
from diffusers.schedulers import DPMSolverMultistepScheduler, DDIMScheduler
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

def compute_snr(noise_scheduler, timesteps):
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

class LGDiffusionPipeline(DiffusionPipeline):

    def __init__(
        self,
        unet: UNet2DModel,
        scheduler: DPMSolverMultistepScheduler,
        model_params: dict = None,
    ):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        
        if model_params is not None:
            self.config["model_params"] = model_params

    @staticmethod
    def create_new(model_params, save_model_path):
        
        # 7 layer - static heads and groups
        """
        unet = UNet2DModel(
            sample_size=sample_size,
            act_fn="silu",
            attention_head_dim=8,
            center_input_sample=False,
            downsample_padding=1,
            flip_sin_to_cos=True,
            freq_shift=0,
            mid_block_scale_factor=1,
            norm_eps=1e-05,
            norm_num_groups=32,
            in_channels=2,
            out_channels=2,
            layers_per_block=2,
            block_out_channels=(32, 64, 96, 192, 384, 768, 1536),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
        """
        num_input_channels, num_output_channels = LGDiffusionPipeline.get_num_channels(model_params)
        
        """
        unet = UNet2DModel(
            act_fn="silu",
            attention_head_dim=8,
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
            block_out_channels=(32, 64, 128, 256, 512, 1024),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
        """

        unet = UNet2DModel(
            act_fn="silu",
            attention_head_dim=16,
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
            block_out_channels=(128, 192, 288, 448, 672, 1024),
            down_block_types=(
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
            ),
        )

        def enforce_zero_terminal_snr(betas):
            # Convert betas to alphas_bar_sqrt
            alphas = 1 - betas
            alphas_bar = alphas.cumprod(0)
            alphas_bar_sqrt = alphas_bar.sqrt()

            # Store old values.
            alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
            alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
            # Shift so last timestep is zero.
            alphas_bar_sqrt -= alphas_bar_sqrt_T
            # Scale so first timestep is back to old value.
            alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

            # Convert alphas_bar_sqrt to betas
            alphas_bar = alphas_bar_sqrt ** 2
            alphas = alphas_bar[1:] / alphas_bar[:-1]
            alphas = torch.cat([alphas_bar[0:1], alphas])
            betas = 1 - alphas
            return betas
        
        beta_schedule = model_params["beta_schedule"]
        beta_start = model_params["beta_start"]
        beta_end = model_params["beta_end"]
        num_train_timesteps = 1000

        if beta_schedule == "log_linear":
            # freq14 beta_start = 0.0001, beta_end = 0.0115
            #ln_freq = ((torch.arange(0, num_train_timesteps, dtype=torch.float32) + 0.5) / num_train_timesteps).log()
            #ln_freq -= ln_freq[0].item()
            #ln_freq /= ln_freq[-1].item()
            #trained_betas = (ln_freq * (beta_end - beta_start) + beta_start).tolist()

            # freq 15 beta_start = beta_end = 0.01
            #trained_betas = (beta_start + beta_end) / 2.
            #trained_betas = [trained_betas] * num_train_timesteps

            # freq 16 beta_start = 0.01, beta_end = 0.018
            #ln_freq = (((torch.arange(0, num_train_timesteps, dtype=torch.float32) + 0.5) / num_train_timesteps)+1.).log()
            #ln_freq /= ln_freq[-1].item()
            #trained_betas = (ln_freq * beta_end).tolist()

            def alpha_bar(time_step):
                #return np.cos((time_step + 0.008) / 1.008 * np.pi / 2) ** 2
                return ( (np.tanh(4*(-time_step + 0.5)) + 1.) / 2. )
                #return 1-time_step


            max_beta = 0.9999999
            trained_betas = []
            for i in range(num_train_timesteps):
                t1 = i / num_train_timesteps
                t2 = (i + 1) / num_train_timesteps
                trained_betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))

            #enforce_zero_terminal_snr(trained_betas) # you also need to change the scheduler to actually use the last timestep
        else:
            trained_betas = None

        prediction_type = model_params["prediction_type"]
        scheduler = DDIMScheduler(clip_sample_range=20.,
                                  prediction_type=prediction_type,
                                  beta_schedule=beta_schedule,
                                  trained_betas=trained_betas,
                                  beta_start=beta_start,
                                  beta_end=beta_end)

        scheduler.set_timesteps(num_train_timesteps)
        log_snrs = compute_snr(scheduler, scheduler.timesteps).log()
        log_snrs.cpu().numpy().tofile("./output/debug_log_snrs.raw")

        pipeline = LGDiffusionPipeline(unet, scheduler, model_params)
        pipeline.save_pretrained(save_model_path, safe_serialization=True)
        return pipeline

    @staticmethod
    def get_sample_crop_width(model_params):
        return model_params["sample_raw_length"] + model_params["overlapped"] * (model_params["sample_raw_length"] // model_params["num_chunks"])
    
    @staticmethod
    def get_num_channels(model_params):
        format = model_params["format"]
        freq_embedding_dim = model_params["freq_embedding_dim"]
        if format == "abs_ln":
            return (1 + freq_embedding_dim, 1)
        elif format == "abs":
            return (1 + freq_embedding_dim, 1)
        elif format == "complex":
            return (1 + freq_embedding_dim, 1)
        elif format == "complex_2channels":
            return (2 + freq_embedding_dim, 2)
        elif format == "complex_1channel":
            return (1 + freq_embedding_dim, 1)
        else:
            raise ValueError(f"Unknown format '{format}'")

    @staticmethod
    def get_chunk_offsets(sample_len, chunk_len): # e.g., 65536+128 is a valid input size for chunk_len 256
        assert(chunk_len % 2 == 0)
        #sample_len -= chunk_len // 2
        assert(sample_len % chunk_len == 0)
        return torch.arange(0, sample_len, step=chunk_len // 2).to("cuda")
    
    @staticmethod
    def get_chunk_indices(sample_len, chunk_len):
        chunk_offsets = LGDiffusionPipeline.get_chunk_offsets(sample_len, chunk_len)
        return torch.arange(0, chunk_len, device="cuda").view(1, -1) + chunk_offsets.view(-1, 1)
    
    @staticmethod
    def get_chunks(samples, chunk_len, overlap=True):
        if overlap:
            chunk_indices = LGDiffusionPipeline.get_chunk_indices(samples.shape[1], chunk_len)
            return samples[:, chunk_indices]
        else:
            return samples.view(samples.shape[0], -1, chunk_len)
    
    @staticmethod
    def get_window(window_len):
        x = torch.arange(0, window_len, device="cuda") + 0.5
        return (torch.ones(window_len, device="cuda") + torch.cos(x / window_len * 2. * np.pi - np.pi)) * 0.5

    #"""
    # freq9
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
            freq_embeddings = LGDiffusionPipeline.get_positional_embedding(ln_freqs, freq_embedding_dim).type(freq_samples.dtype)
            freq_embeddings = freq_embeddings.view(1, freq_embedding_dim, freq_samples.shape[2], 1).repeat(freq_samples.shape[0], 1, 1, freq_samples.shape[3])
        else:
            ln_freqs /= ln_freqs[0].item()
            ln_freqs = ln_freqs.type(freq_samples.dtype)
            freq_embeddings = ln_freqs.view(1, 1, freq_samples.shape[2], 1).repeat(freq_samples.shape[0], 1, 1, freq_samples.shape[3])

        return torch.cat((freq_samples, freq_embeddings), dim=1)
    #"""

    @staticmethod
    @torch.no_grad()
    def raw_to_freq(raw_samples, model_params, format_override=None):

        raw_samples = raw_samples.clone()
        raw_samples /= raw_samples.std(dim=1, keepdim=True)

        noise_floor = model_params["noise_floor"]
        if noise_floor > 0:
            raw_samples += torch.randn_like(raw_samples) * noise_floor

        spatial_window_len = model_params["spatial_window_length"]
        if spatial_window_len > 0:
            spatial_window = LGDiffusionPipeline.get_window(spatial_window_len).square_()
            raw_samples[:, :spatial_window_len//2]  *= spatial_window[:spatial_window_len//2]
            raw_samples[:, -spatial_window_len//2:] *= spatial_window[-spatial_window_len//2:]

        fft_samples_positive_frequencies_only = torch.fft.fft(raw_samples, norm="ortho")[:, :raw_samples.shape[1]//2]
        #fft_samples_positive_frequencies_only[:, 0] = 0. # explicitly remove dc component, otherwise the first chunk has significant non-zero mean
        num_chunks = model_params["num_chunks"]
        chunk_len = fft_samples_positive_frequencies_only.shape[1] // num_chunks
        overlapped = model_params["overlapped"]
        fft_samples_chunks = LGDiffusionPipeline.get_chunks(fft_samples_positive_frequencies_only, chunk_len, overlapped)
        #fft_samples_chunks[:, :, 0] = 0. # explicitly remove dc component, otherwise chunks have significant non-zero mean

        window_type = model_params["window_type"]
        if overlapped:
            if window_type == "sqrt_hann":
                windowed_fft_chunks = fft_samples_chunks * LGDiffusionPipeline.get_window(chunk_len).sqrt_().view(1, 1,-1)
            elif window_type == "hann":
                windowed_fft_chunks = fft_samples_chunks * LGDiffusionPipeline.get_window(chunk_len).view(1, 1,-1)
            elif window_type == "none":
                pass
            else:
                raise ValueError(f"Unknown window type '{window_type}'")
        else:
            windowed_fft_chunks = fft_samples_chunks

        windowed_chunk_ffts = torch.fft.fft(windowed_fft_chunks, norm="ortho")
        
        format = model_params["format"] if format_override is None else format_override
        if format == "abs_ln":
            freq_samples = windowed_chunk_ffts.abs().log_().unsqueeze(1)
            freq_samples -= freq_samples.mean(dim=(1, 2, 3), keepdim=True)
            avg_ln_std = model_params["avg_std"]
            if avg_ln_std > 0: freq_samples /= avg_ln_std
        elif format == "abs":
            freq_samples = windowed_chunk_ffts.abs().unsqueeze(1)
            freq_samples /= freq_samples.std(dim=(1, 2, 3), keepdim=True)
        elif format == "complex":
            freq_samples = windowed_chunk_ffts.unsqueeze(1)
            freq_samples /= freq_samples.std(dim=(1, 2, 3), keepdim=True)
        elif format == "complex_2channels":
            freq_samples = torch.view_as_real(windowed_chunk_ffts).permute(0, 3, 1, 2).contiguous()
            freq_samples /= freq_samples.std(dim=(1, 2, 3), keepdim=True)
        elif format == "complex_1channel":
            freq_samples = torch.view_as_real(windowed_chunk_ffts).view(windowed_chunk_ffts.shape[0], windowed_chunk_ffts.shape[1], -1).unsqueeze(1)
            freq_samples /= freq_samples.std(dim=(1, 2, 3), keepdim=True)
        else:
            raise ValueError(f"Unknown format '{format}'")

        return freq_samples
    
    @staticmethod
    @torch.no_grad()
    def freq_to_raw(freq_samples, model_params, phases=None):
        
        fft_sample_len = LGDiffusionPipeline.get_sample_crop_width(model_params) // 2
        chunk_len = fft_sample_len // model_params["num_chunks"]
        num_chunks = model_params["num_chunks"]

        format = model_params["format"]

        if format == "complex_1channel":
            windowed_chunk_ffts = torch.view_as_complex(freq_samples.view(freq_samples.shape[0], freq_samples.shape[2], -1, 2))
        elif format == "complex_2channels":
            windowed_chunk_ffts = torch.view_as_complex(freq_samples.permute(0, 2, 3, 1).contiguous())
        elif format == "abs":
            if phases is None:
                spatial = torch.randn((freq_samples.shape[0], fft_sample_len*2), device=freq_samples.device, dtype=freq_samples.dtype)
                phases = LGDiffusionPipeline.raw_to_freq(spatial, model_params, format_override="complex")
            windowed_chunk_ffts = phases / phases.abs() * freq_samples
        else:
            raise ValueError(f"Unknown format '{format}'")
        
        windowed_fft_chunks = torch.fft.ifft(windowed_chunk_ffts, norm="ortho")
        fft_samples = windowed_fft_chunks.view(windowed_fft_chunks.shape[0], -1)

        #chunk_indices = LGDiffusionPipeline.get_chunk_indices(fft_sample_len, chunk_len)
        #fft_samples = torch.zeros((freq_samples.shape[0], fft_sample_len), device=windowed_fft_chunks.device, dtype=windowed_fft_chunks.dtype)
        #fft_samples[:, chunk_indices[0::2]]  = windowed_fft_chunks[:, 0::2, :]
        #fft_samples[:, chunk_indices[1::2]] += windowed_fft_chunks[:, 1::2, :]
        
        fft_samples = torch.cat((fft_samples, torch.zeros_like(fft_samples)), dim=1)
        raw_samples = torch.fft.ifft(fft_samples, norm="ortho")
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
        seed=0,
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
        
        model_params = self.config["model_params"]
        freq_embedding_dim = model_params["freq_embedding_dim"]

        if loops > 0:
            self.set_tiling_mode("x")
        else:
            self.set_tiling_mode(False)

        prediction_type = self.scheduler.config["prediction_type"]
        beta_schedule = self.scheduler.config["beta_schedule"]
        if scheduler == "ddim":
            noise_scheduler = self.scheduler
        elif scheduler == "dpms++":
            if beta_schedule == "log_linear":
                noise_scheduler = DPMSolverMultistepScheduler(prediction_type=prediction_type, solver_order=3, trained_betas=self.scheduler.trained_betas)
            else:
                noise_scheduler = DPMSolverMultistepScheduler(prediction_type=prediction_type, solver_order=3, beta_schedule=beta_schedule)
        else:
            raise ValueError(f"Unknown scheduler '{scheduler}'")
        
        if seed == 0: seed = np.random.randint(10000000,
                                               99999999)
        torch.manual_seed(seed)

        num_input_channels, num_output_channels = LGDiffusionPipeline.get_num_channels(model_params)
        noise = torch.randn((batch_size, num_output_channels, model_params["num_chunks"], 256*length,), device=self.device)#, dtype=torch.float16)
        sample = noise; print(f"Sample shape: {sample.shape}")

        noise_scheduler.set_timesteps(steps)
        timesteps = noise_scheduler.timesteps

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

        for step, t in enumerate(self.progress_bar(timesteps)):
            
            model_input = sample
            if freq_embedding_dim > 0:
                model_input = LGDiffusionPipeline.add_freq_embedding(model_input, freq_embedding_dim)
            model_input = noise_scheduler.scale_model_input(model_input, t)
            model_output = self.unet(model_input, t)["sample"]
            
            if step == 0:
                print(f"Model output shape: {model_output.shape}")
                model_output.float().cpu().numpy().tofile("./output/debug_model_output.raw")

            sample = noise_scheduler.step(
                model_output=model_output,
                timestep=t,
                sample=sample,
            )["prev_sample"]

            #sample -= sample.amin(dim=(1,2,3), keepdim=True)
            #sample = sample.clip(min=0.)
            #sample /= sample.std(dim=(1,2,3), keepdim=True)

            if rebalance_mean:
                sample -= sample.mean(dim=(1,2,3), keepdim=True)
            if renormalize_sample:
                sample /= sample.std(dim=(1,2,3), keepdim=True)

        if model_params["format"] == "abs":
            sample = sample.clip(min=0.)
        elif model_params["format"] == "abs_ln":
            sample -= sample.mean(dim=(1,2,3), keepdim=True)

        print("Sample std: ", sample.std(dim=(1,2,3)).item())

        sample = sample.type(torch.float32)
        sample.cpu().numpy().tofile("./output/debug_sample.raw")

        raw_sample = LGDiffusionPipeline.freq_to_raw(sample, model_params)
        if loops > 0: raw_sample = raw_sample.repeat(1, loops+1)
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