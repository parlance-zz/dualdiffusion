from typing import Tuple, Literal

import numpy as np
import torch

from diffusers.models import UNet2DModel
from diffusers.schedulers import DPMSolverMultistepScheduler, DDIMScheduler
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

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

        #scheduler = DPMSolverMultistepScheduler(prediction_type="v_prediction", solver_order=3, beta_schedule="squaredcos_cap_v2")
        scheduler = DDIMScheduler(clip_sample_range=20., prediction_type="v_prediction", beta_schedule="squaredcos_cap_v2")

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
        ln_freqs *= freq_samples.shape[2] / ln_freqs[0]

        freq_embeddings = LGDiffusionPipeline.get_positional_embedding(ln_freqs, freq_embedding_dim)
        freq_embeddings = freq_embeddings.view(1, freq_embedding_dim, freq_samples.shape[2], 1).repeat(freq_samples.shape[0], 1, 1, freq_samples.shape[3])
        return torch.cat((freq_samples, freq_embeddings), dim=1)
    
    @staticmethod
    @torch.no_grad()
    def raw_to_freq(raw_samples, model_params):

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
        num_chunks = model_params["num_chunks"]
        chunk_len = fft_samples_positive_frequencies_only.shape[1] // num_chunks
        overlapped = model_params["overlapped"]
        fft_samples_chunks = LGDiffusionPipeline.get_chunks(fft_samples_positive_frequencies_only, chunk_len, overlapped)
        
        format = model_params["format"]
        if overlapped:
            if format == "abs_ln":
                windowed_fft_chunks = fft_samples_chunks * LGDiffusionPipeline.get_window(chunk_len).sqrt_().view(1, 1,-1)
            else:
                windowed_fft_chunks = fft_samples_chunks * LGDiffusionPipeline.get_window(chunk_len).view(1, 1,-1)
        else:
            windowed_fft_chunks = fft_samples_chunks

        windowed_chunk_ffts = torch.fft.fft(windowed_fft_chunks, norm="ortho")
        
        if format == "abs_ln":
            freq_samples = windowed_chunk_ffts.abs_().log_().unsqueeze(1)
            freq_samples -= freq_samples.mean(dim=(1, 2, 3), keepdim=True)
            avg_ln_std = model_params["avg_std"]
            if avg_ln_std > 0: freq_samples /= avg_ln_std
        elif format == "abs":
            freq_samples = windowed_chunk_ffts.abs_().unsqueeze(1)
            freq_samples /= freq_samples.std(dim=(1, 2, 3), keepdim=True)
        elif format == "complex":
            freq_samples = freq_samples.unsqueeze(1)
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
    def freq_to_raw(freq_samples, model_params):
        
        fft_sample_len = LGDiffusionPipeline.get_sample_crop_width(model_params) // 2
        chunk_len = fft_sample_len // model_params["num_chunks"]
        num_chunks = model_params["num_chunks"]

        windowed_chunk_ffts = torch.view_as_complex(freq_samples.view(freq_samples.shape[0], freq_samples.shape[2], -1, 2))
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
        steps: int = 50,
        scheduler="dpms++",
        seed=0,
        loops: int = 1,
        batch_size: int = 1,
    ):
        if (steps <= 0) or (steps > 1000):
            raise ValueError(f"Steps must be between 1 and 1000, got {steps}")
        if loops < 0:
            raise ValueError(f"Loops must be greater than or equal to 0, got {loops}")
        
        model_params = self.config["model_params"]
        freq_embedding_dim = model_params["freq_embedding_dim"]

        if loops > 0:
            self.set_tiling_mode("x")
        else:
            self.set_tiling_mode(False)

        if scheduler == "ddim":
            self.scheduler = DDIMScheduler(clip_sample_range=20., prediction_type="v_prediction", beta_schedule="squaredcos_cap_v2")
        elif scheduler == "dpms++":
            self.scheduler = DPMSolverMultistepScheduler(prediction_type="v_prediction", solver_order=3, beta_schedule="squaredcos_cap_v2")
        else:
            raise ValueError(f"Unknown scheduler '{scheduler}'")
        
        if seed == 0: seed = np.random.randint(10000000,
                                               99999999)
        torch.manual_seed(seed)

        num_input_channels, num_output_channels = LGDiffusionPipeline.get_num_channels(model_params)
        noise = torch.randn((batch_size, num_output_channels, model_params["num_chunks"], 256,), device=self.device, dtype=torch.float16)
        sample = noise; print(f"Sample shape: {sample.shape}")

        self.scheduler.set_timesteps(steps)
        for step, t in enumerate(self.progress_bar(self.scheduler.timesteps)):

            model_input = sample
            if freq_embedding_dim > 0:
                model_input = LGDiffusionPipeline.add_freq_embedding(model_input.float(), freq_embedding_dim).type(torch.float16)
            model_input = self.scheduler.scale_model_input(model_input, t)
            model_output = self.unet(model_input, t)["sample"]
            
            if step == 0:
                print(f"Model input shape: {model_input.shape}")
                print(f"Model output shape: {model_output.shape}")
                model_input.float().cpu().numpy().tofile("./output/debug_model_input.raw")
                model_output.float().cpu().numpy().tofile("./output/debug_model_output.raw")

            sample = self.scheduler.step(
                model_output=model_output,
                timestep=t,
                sample=sample,
            )["prev_sample"]

            #sample -= sample.mean(dim=(1,2,3), keepdim=True)
            #sample /= sample.std(dim=(1,2,3), keepdim=True)

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


"""

def get_chunk_offsets(sample_len, chunk_len): # e.g., 65536+128 is a valid input size for chunk_len 256
    #assert(chunk_len % 2 == 0)
    #sample_len -= chunk_len // 2
    #assert(sample_len % chunk_len == 0)
    #return torch.arange(0, sample_len, step=chunk_len // 2).to("cuda")
    overlap = 2
    step = int(chunk_len / overlap + 0.5)

    window_offsets = torch.arange(0, sample_len-chunk_len, step=step)
    if (window_offsets[-1] + chunk_len) < sample_len:
        window_offsets = torch.cat((window_offsets, torch.tensor([sample_len-chunk_len])))
    
    return window_offsets.to("cuda")


def get_chunk_indices(sample_len, chunk_len):
    chunk_offsets = LGDiffusionPipeline.get_chunk_offsets(sample_len, chunk_len)
    return torch.arange(0, chunk_len, device="cuda").view(1, -1) + chunk_offsets.view(-1, 1)

def get_chunks(samples, chunk_len):
    chunk_indices = LGDiffusionPipeline.get_chunk_indices(samples.shape[1], chunk_len)
    return samples[:, chunk_indices]

def get_window(window_len):
    x = torch.arange(0, window_len, device="cuda") + 0.5
    return (torch.ones(window_len, device="cuda") + torch.cos(x / window_len * 2. * np.pi - np.pi)) * 0.5

@torch.no_grad()
def raw_to_freq(raw_samples, chunk_len):
    
    spatial_window = LGDiffusionPipeline.get_window(chunk_len)
    raw_samples = raw_samples.clone()
    raw_samples[:, :chunk_len//2]  *= spatial_window[:chunk_len//2]
    raw_samples[:, -chunk_len//2:] *= spatial_window[-chunk_len//2:]

    half_chunk_len = chunk_len // 2

    fft_samples_positive_frequencies_only = torch.fft.fft(raw_samples, norm="ortho")[:, :raw_samples.shape[1]//2]
    fft_samples_chunks = LGDiffusionPipeline.get_chunks(fft_samples_positive_frequencies_only, half_chunk_len)
    windowed_fft_chunks = fft_samples_chunks * get_window(half_chunk_len).view(1, 1,-1) ** 0.5
    balanced_windowed_fft_chunks = windowed_fft_chunks# * (torch.arange(1, windowed_fft_chunks.shape[1]+1, device="cuda") / windowed_fft_chunks.shape[1]).view(1,-1, 1)
    windowed_chunk_ffts = torch.fft.fft(balanced_windowed_fft_chunks, norm="ortho")
    
    freq_samples = windowed_chunk_ffts.unsqueeze(1)
    return freq_samples / freq_samples.std(dim=(1, 2, 3), keepdim=True)

@torch.no_grad()
def freq_to_raw(freq_samples):
    freq_samples = freq_samples.squeeze(1)

    half_chunk_len = freq_samples.shape[2]
    fft_sample_len = freq_samples.shape[1] // 2 * half_chunk_len + half_chunk_len // 2
    
    windowed_chunk_ffts = freq_samples
    windowed_fft_chunks = torch.fft.ifft(windowed_chunk_ffts, norm="ortho")
    unbalanced_windowed_fft_chunks = windowed_fft_chunks# / (torch.arange(1, windowed_fft_chunks.shape[1]+1, device="cuda") / windowed_fft_chunks.shape[1]).view(1,-1, 1)

    chunk_indices = LGDiffusionPipeline.get_chunk_indices(fft_sample_len, half_chunk_len)
    fft_samples = torch.zeros((freq_samples.shape[0], fft_sample_len), device=unbalanced_windowed_fft_chunks.device, dtype=unbalanced_windowed_fft_chunks.dtype)
    #fft_samples[:, chunk_indices[0::2]]  = unbalanced_windowed_fft_chunks[:, 0::2, :]
    #fft_samples[:, chunk_indices[1::2]] += unbalanced_windowed_fft_chunks[:, 1::2, :]
    
    overlap = 2
    for i in range(overlap):
        fft_samples[:, chunk_indices[i::overlap]] += unbalanced_windowed_fft_chunks[:, i::overlap, :]
        
    fft_samples = torch.cat((fft_samples, torch.zeros_like(fft_samples)), dim=1)
    raw_samples = torch.fft.ifft(fft_samples, norm="ortho")
    return raw_samples / raw_samples.std(dim=1, keepdim=True) * 0.18215

@torch.no_grad()
def phase_reconstruct(x, chunk_size, avg_std):
    
    crop_width = x.shape[3] * x.shape[2] + x.shape[3]
    y = torch.randn((1, crop_width), device="cuda")

    for i in range(128):
        y = raw_to_freq(y, chunk_size)
        y = y / y.abs() * torch.exp(x * avg_std)
        y = freq_to_raw(y)
    
    return y

"""