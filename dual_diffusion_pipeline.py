from typing import Literal, Union

import numpy as np
import torch

from diffusers.schedulers import DPMSolverMultistepScheduler, DDIMScheduler
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from unet2d_dual import UNet2DDualModel

def to_freq(x):
    x = x.permute(0, 2, 3, 1).contiguous().float()
    x = torch.fft.fft2(torch.view_as_complex(x), norm="ortho")
    x = torch.view_as_real(x)
    return x.permute(0, 3, 1, 2).contiguous()

def to_spatial(x):
    x = x.permute(0, 2, 3, 1).contiguous().float()
    x = torch.fft.ifft2(torch.view_as_complex(x), norm="ortho")
    x = torch.view_as_real(x)
    return x.permute(0, 3, 1, 2).contiguous()

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

        self.tiling_mode = False

    @staticmethod
    def create_new(model_params, save_model_path):
        
        num_input_channels, num_output_channels = DualDiffusionPipeline.get_num_channels(model_params)

        unet = UNet2DDualModel(
            #dropout=0.1,
            dropout=0.0,
            #act_fn="mish",
            act_fn="silu",
            #attention_head_dim=(16, 32, 64),
            attention_head_dim=16,
            #separate_attn_dim=(3,2),
            separate_attn_dim=(2,3),
            #positional_coding_dims=(3,), 
            positional_coding_dims=(),
            #reverse_separate_attn_dim=True,
            reverse_separate_attn_dim=False,
            #double_attention=True,
            double_attention=False,
            #add_attention=True,
            add_attention=False,
            downsample_padding=1,
            flip_sin_to_cos=True,
            freq_shift=0,
            mid_block_scale_factor=1,
            norm_eps=1e-05,
            norm_num_groups=32,
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            layers_per_block=2,
            conv_size=(3,3),
            #downsample_type="resnet",
            #upsample_type="resnet",

            #block_out_channels=(32, 64, 128),
            #down_block_types=(
            #    "SeparableAttnDownBlock2D",
            #    "SeparableAttnDownBlock2D",
            #    "SeparableAttnDownBlock2D",
            #),
            #up_block_types=(
            #    "SeparableAttnUpBlock2D",
            #    "SeparableAttnUpBlock2D",
            #    "SeparableAttnUpBlock2D",
            #),
            block_out_channels=(32, 32, 64, 128, 256),
            down_block_types=(
                "SeparableAttnDownBlock2D",
                "SeparableAttnDownBlock2D",
                "SeparableAttnDownBlock2D",
                "SeparableAttnDownBlock2D",
                "SeparableAttnDownBlock2D",
            ),
            up_block_types=(
                "SeparableAttnUpBlock2D",
                "SeparableAttnUpBlock2D",
                "SeparableAttnUpBlock2D",
                "SeparableAttnUpBlock2D",
                "SeparableAttnUpBlock2D",
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
                                  beta_end=beta_end,
                                  rescale_betas_zero_snr=True)

        pipeline = DualDiffusionPipeline(unet, scheduler, model_params)
        pipeline.save_pretrained(save_model_path, safe_serialization=True)
        return pipeline

    @staticmethod
    def get_sample_crop_width(model_params):
        return model_params["sample_raw_length"]
    
    @staticmethod
    def get_num_channels(model_params):
        freq_embedding_dim = model_params["freq_embedding_dim"]
        channels = model_params["channels"]
        return (channels + freq_embedding_dim, channels)

    @staticmethod
    @torch.no_grad()
    def get_window(window_len):
        x = torch.arange(0, window_len, device="cuda") / window_len
        return (1 + torch.cos(x * 2.*np.pi - np.pi)) * 0.5

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
    def raw_to_sample(raw_samples, model_params, window=None):
        channels = model_params["channels"]
        num_chunks = model_params["num_chunks"]
        sample_len = model_params["sample_raw_length"]
        half_sample_len = sample_len // 2
        chunk_len = half_sample_len // num_chunks
        half_chunk_len = chunk_len // 2
        bsz = raw_samples.shape[0]

        fft = torch.fft.fft(raw_samples, norm="ortho")[:, :half_sample_len + half_chunk_len]
        fft[:, half_sample_len:] = 0.

        slices_1 = fft[:, :half_sample_len].view(bsz, num_chunks, chunk_len)
        slices_2 = fft[:,  half_chunk_len:].view(bsz, num_chunks, chunk_len)

        if window is None:
            window = DualDiffusionPipeline.get_window(chunk_len)

        samples = torch.cat((slices_1, slices_2), dim=2).view(bsz, num_chunks*2, chunk_len) * window

        if channels == 2:
            samples = torch.view_as_real(torch.fft.fft(samples, norm="ortho")).permute(0, 3, 1, 2).contiguous()
        elif channels == 1:
            samples = torch.view_as_real(torch.fft.fft(samples, norm="ortho")).view(bsz, 1, num_chunks*2, chunk_len*2)
        else:
            raise ValueError(f"Invalid number of channels '{channels}', must be 1 or 2")
        
        samples /= samples.std(dim=(1, 2, 3), keepdim=True)
        return samples, window
    
    @staticmethod
    @torch.no_grad()
    def sample_to_raw(samples):    
        channels = samples.shape[1]
        num_chunks = samples.shape[2] // 2
        chunk_len = samples.shape[3] // 2 * channels
        half_chunk_len = chunk_len // 2
        sample_len = samples.shape[2] * chunk_len
        half_sample_len = sample_len // 2
        
        bsz = samples.shape[0]
        
        if channels == 2:
            samples = torch.view_as_complex(samples.permute(0, 2, 3, 1).contiguous())
        elif channels == 1:
            samples = torch.view_as_complex(samples.view(bsz, 1, num_chunks*2, chunk_len, 2)).squeeze(1)
        else:
            raise ValueError(f"Invalid number of channels '{channels}', must be 1 or 2")
        
        samples = torch.fft.ifft(samples, norm="ortho")

        slices_1 = samples[:, 0::2, :]
        slices_2 = samples[:, 1::2, :]

        fft = torch.zeros((bsz, sample_len), dtype=torch.complex64, device="cuda")
        fft[:, :half_sample_len] = slices_1.reshape(bsz, -1)
        fft[:,  half_chunk_len:half_sample_len+half_chunk_len] += slices_2.reshape(bsz, -1)
        fft[:, half_sample_len:half_sample_len+half_chunk_len] = 0.
        
        return torch.fft.ifft(fft, norm="ortho") * 2.

    @staticmethod
    @torch.no_grad()
    def save_sample_img(sample, img_path):

        import cv2

        sample = sample.abs()
        sample /= torch.max(sample)
        sample = (sample.cpu().numpy()*255.).astype(np.uint8)
        sample_img = sample
        sample_img = cv2.applyColorMap(sample, cv2.COLORMAP_JET)
        cv2.imwrite(img_path, sample_img)

    @torch.no_grad()
    def __call__(
        self,
        steps: int = 100,
        scheduler="dpms++",
        seed: Union[int, torch.Generator]=None,
        loops: int = 0,
        batch_size: int = 1,
        length: int = 1,
    ):
        if (steps <= 0) or (steps > 1000):
            raise ValueError(f"Steps must be between 1 and 1000, got {steps}")
        if loops < 0:
            raise ValueError(f"Loops must be greater than or equal to 0, got {loops}")
        if length <= 0:
            raise ValueError(f"Length must be greater than or equal to 1, got {length}")

        self.set_tiling_mode(loops > 0)

        if scheduler == "ddim":
            noise_scheduler = self.scheduler
        elif scheduler == "dpms++":
            prediction_type = self.scheduler.config["prediction_type"]
            beta_schedule = self.scheduler.config["beta_schedule"]
            noise_scheduler = DPMSolverMultistepScheduler(prediction_type=prediction_type,
                                                          solver_order=3,
                                                          beta_schedule=beta_schedule)
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
        channels = model_params["channels"]
        default_length = sample_crop_width // num_chunks // channels
        num_input_channels, num_output_channels = DualDiffusionPipeline.get_num_channels(model_params)

        noise = torch.randn((batch_size, num_output_channels, num_chunks*2, default_length*length,),
                            device=self.device,
                            generator=generator)
        sample = noise
        freq_embedding_dim = model_params["freq_embedding_dim"]

        print(f"Sample shape: {sample.shape}")

        # terminal timestep imposed mean test
        """
        raw_sample = np.fromfile("D:/dualdiffusion/dataset/samples/400.raw", dtype=np.int16, count=sample_crop_width) / 32768.
        raw_sample = torch.from_numpy(raw_sample).unsqueeze(0).to("cuda").type(torch.float32)
        freq_sample = DualDiffusionPipeline.raw_to_freq(raw_sample, model_params)
        #freq_sample_std = freq_sample.std(dim=(1,3), keepdim=True)
        #noise2 = torch.randn_like(noise) * freq_sample_std
        freq_sample_mean = freq_sample.mean(dim=3, keepdim=True)
        noise2 = freq_sample_mean.repeat(1, 1, 1, sample.shape[3])
        sample = noise_scheduler.add_noise(noise2, noise, noise_scheduler.timesteps[12]) # 0-2 gives best results, but 3-10 might be usable
        #sample = noise_scheduler.add_noise(freq_sample, noise, noise_scheduler.timesteps[9]) #0 to ~12 gives best results for audio2audio
        """

        for step, t in enumerate(self.progress_bar(timesteps)):
            
            model_input = sample
            if freq_embedding_dim > 0:
                model_input = DualDiffusionPipeline.add_freq_embedding(model_input, freq_embedding_dim)
            model_input = noise_scheduler.scale_model_input(model_input, t)
            model_output = self.unet(model_input, t).sample
            
            sample = noise_scheduler.step(
                model_output=model_output,
                timestep=t,
                sample=sample,
                generator=generator,
            )["prev_sample"]

            #sample /= sample.std(dim=3, keepdim=True)

        print("Sample std: ", sample.std(dim=(1,2,3)).item())

        sample = sample.type(torch.float32)
        sample.cpu().numpy().tofile("./output/debug_sample.raw")

        raw_sample = DualDiffusionPipeline.sample_to_raw(sample)
        raw_sample *= 0.18215 / raw_sample.std(dim=1, keepdim=True)
        if loops > 0: raw_sample = raw_sample.repeat(1, loops+1)
        return raw_sample
    
    def set_module_tiling(self, module):
        F, _pair = torch.nn.functional, torch.nn.modules.utils._pair

        padding_modeX = "circular"
        padding_modeY = "constant"

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

    def set_tiling_mode(self, tiling: bool):

        if self.tiling_mode == tiling:
            return

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
                        self.set_module_tiling(submodule)

        self.tiling_mode = tiling