import os
import json
import time
import glob
from typing import List, Tuple, Union, Optional

import numpy as np
import torch
import torch.nn.functional as F

from diffusers.models import UNet1DModel
from diffusers.schedulers import DDIMScheduler, DDPMScheduler
from diffusers.pipelines.pipeline_utils import AudioPipelineOutput, DiffusionPipeline
from diffusers.utils import randn_tensor

class DualDiffusionPipeline(DiffusionPipeline):

    def __init__(
        self,
        unet_s: UNet1DModel,
        unet_f: UNet1DModel,
        scheduler_s: Union[DDIMScheduler, DDPMScheduler],
        scheduler_f: Union[DDIMScheduler, DDPMScheduler],
        sample_len: int = None,
        s_resolution: int = None,
        f_resolution: int = None,
        s_avg_std: float = None,
        f_avg_std: float = None,
    ):
        super().__init__()
        self.register_modules(unet_s=unet_s, unet_f=unet_f, scheduler_s=scheduler_s, scheduler_f=scheduler_f)

        #if s_filter_params is not None: self.config["s_filter_params"] = s_filter_params
        if sample_len is not None: self.config["sample_len"] = sample_len
        if s_resolution is not None: self.config["s_resolution"] = s_resolution
        if f_resolution is not None: self.config["f_resolution"] = f_resolution
        if s_avg_std is not None: self.config["s_avg_std"] = s_avg_std
        if f_avg_std is not None: self.config["f_avg_std"] = f_avg_std

    def get_default_steps(self) -> int:
        return 50 if isinstance(self.scheduler, DDIMScheduler) else 1000

    @staticmethod
    def create_new(data_cfg_path, save_model_path):

        with open(data_cfg_path, "rb") as f:
            dataset_cfg = json.load(f)

        scheduler_s = DDIMScheduler(clip_sample_range=10000.)
        scheduler_f = DDIMScheduler(clip_sample_range=10000.)
        unet_s = UNet1DModel(sample_size=dataset_cfg["s_resolution"],
                             in_channels=2,
                             out_channels=2,
                             layers_per_block=2,
                             block_out_channels=(128, 128, 256, 256, 512, 512),
                             down_block_types=(
                                 "DownBlock1D",
                                 "DownBlock1D",
                                 "DownBlock1D",
                                 "DownBlock1D",
                                 "AttnDownBlock1D",
                                 "DownBlock1D"),
                             up_block_types=(
                                 "UpBlock1D",
                                 "AttnUpBlock1D",
                                 "UpBlock1D",
                                 "UpBlock1D",
                                 "UpBlock1D",
                                 "UpBlock1D")
                             )
        
        unet_f = UNet1DModel(sample_size=dataset_cfg["f_resolution"],
                             in_channels=2,
                             out_channels=2,
                             layers_per_block=2,
                             block_out_channels=(128, 128, 256, 256, 512, 512),
                             down_block_types=(
                                 "DownBlock1D",
                                 "DownBlock1D",
                                 "DownBlock1D",
                                 "DownBlock1D",
                                 "AttnDownBlock1D",
                                 "DownBlock1D"),
                             up_block_types=(
                                 "UpBlock1D",
                                 "AttnUpBlock1D",
                                 "UpBlock1D",
                                 "UpBlock1D",
                                 "UpBlock1D",
                                 "UpBlock1D")
                             )

        pipeline = DualDiffusionPipeline(unet_s,
                                         unet_f,
                                         scheduler_s,
                                         scheduler_f,
                                         sample_len=dataset_cfg["sample_len"],
                                         s_resolution=dataset_cfg["s_resolution"],
                                         f_resolution=dataset_cfg["f_resolution"],
                                         s_avg_std = dataset_cfg["s_avg_std"],
                                         f_avg_std = dataset_cfg["f_avg_std"],
                                         )
        
        pipeline.save_pretrained(save_model_path, safe_serialization=True)
        return pipeline

    @staticmethod
    def get_window_offsets(resolution, overlap, sample_len):
            
            step = int(resolution / overlap + 0.5)

            window_offsets = torch.arange(0, sample_len-resolution, step=step)
            if (window_offsets[-1] + resolution) < sample_len:
                window_offsets = torch.cat((window_offsets, torch.tensor([sample_len-resolution])))
            
            return window_offsets.to("cuda")

    @staticmethod
    def get_window(resolution):
        return((torch.ones(resolution, device="cuda") + torch.cos(torch.arange(0, resolution, device="cuda") / resolution * 2. * np.pi - np.pi)) * 0.5)

    @staticmethod
    def get_s_samples(raw_input, s_resolution):

        window_offsets = DualDiffusionPipeline.get_window_offsets(s_resolution, 2, len(raw_input))

        response = torch.zeros((window_offsets.shape[0], s_resolution), dtype=torch.complex64, device="cuda")
        s_window = DualDiffusionPipeline.get_window(s_resolution)

        response_indices = torch.arange(0, s_resolution, device="cuda").view(1, -1) + window_offsets.view(-1, 1)
        response = torch.fft.fft(raw_input[response_indices] * s_window.view(1, -1), norm="ortho")
        response = response[:, :s_resolution//2]
        response[:, ::2] *= -1. # 180 degree phase shift

        return torch.view_as_real(response).permute(0, 2, 1)
    
    @staticmethod
    def invert_s_samples(s_response, raw_input):

        s_resolution = s_response.shape[2] * 2
        window_offsets = DualDiffusionPipeline.get_window_offsets(s_resolution, 2, len(raw_input))

        s_response = torch.view_as_complex(s_response.permute(0, 2, 1))
        s_response[:, ::2] *= -1. # undo 180 degree phase shift
        s_response = F.pad(s_response, (0, 1))
        
        window_offsets_even = window_offsets[::2]
        s_response_even = s_response[::2]
        response_indices_even = torch.arange(0, s_resolution, device="cuda").view(1, -1) + window_offsets_even.view(-1, 1)
        raw_input[response_indices_even] = torch.fft.irfft(s_response_even, norm="ortho")

        window_offsets_odd = window_offsets[1::2]
        s_response_odd = s_response[1::2]
        response_indices_odd = torch.arange(0, s_resolution, device="cuda").view(1, -1) + window_offsets_odd.view(-1, 1)
        raw_input[response_indices_odd] += torch.fft.irfft(s_response_odd, norm="ortho")

        return raw_input

    @staticmethod
    def get_f_samples(fft_input, f_resolution):

        window_offsets = DualDiffusionPipeline.get_window_offsets(f_resolution, 2, len(fft_input))

        response = torch.zeros((window_offsets.shape[0], f_resolution), dtype=torch.complex64)
        f_window = DualDiffusionPipeline.get_window(f_resolution)

        response_indices = torch.arange(0, f_resolution).to("cuda").view(1, -1) + window_offsets.view(-1, 1)
        response = torch.fft.ifft(fft_input[response_indices] * f_window.view(1, -1), norm="ortho")
        response[:, ::2] *= -1. # 180 degree phase shift

        return torch.view_as_real(response).permute(0, 2, 1)
    
    @staticmethod
    def invert_f_samples(f_response, fft_input):

        f_resolution = f_response.shape[2]
        window_offsets = DualDiffusionPipeline.get_window_offsets(f_resolution, 2, len(fft_input))

        f_response = torch.view_as_complex(f_response.permute(0, 2, 1))
        f_response[:, ::2] *= -1. # undo 180 degree phase shift

        window_offsets_even = window_offsets[::2]
        f_response_even = f_response[::2]
        response_indices_even = torch.arange(0, f_resolution, device="cuda").view(1, -1) + window_offsets_even.view(-1, 1)
        fft_input[response_indices_even] = torch.fft.fft(f_response_even, norm="ortho")

        window_offsets_odd = window_offsets[1::2]
        f_response_odd = f_response[1::2]
        response_indices_odd = torch.arange(0, f_resolution, device="cuda").view(1, -1) + window_offsets_odd.view(-1, 1)
        fft_input[response_indices_odd] += torch.fft.fft(f_response_odd, norm="ortho")

        return fft_input

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        length: int = 0,
        start_step: int = 0,
        steps: int = None,
        generator: torch.Generator = None,
        step_generator: torch.Generator = None,
        eta: float = 0,
        noise: torch.Tensor = None,
    ) -> Union[
        AudioPipelineOutput,
        Tuple[int, List[np.ndarray]],
    ]:
        """

        Args:
            start_step (int): step to start from
            steps (`int`): number of de-noising steps (defaults to 50 for DDIM, 1000 for DDPM)
            generator (`torch.Generator`): random number generator or None
            step_generator (`torch.Generator`): random number generator used to de-noise or None
            eta (`float`): parameter between 0 and 1 used with DDIM scheduler
            noise (`torch.Tensor`): noise tensor of shape (batch_size, in_channels, sample_size) or None
            return_dict (`bool`): if True return AudioPipelineOutput, ImagePipelineOutput else Tuple

        Returns:
            `List[PIL Image]`: mel spectrograms (`float`, `List[np.ndarray]`): sample rate and raw audios
        """

        steps = steps or self.get_default_steps()
        self.scheduler_s.set_timesteps(steps)
        self.scheduler_f.set_timesteps(steps)
        step_generator = step_generator or generator

        if length == 0: length = self.config["sample_len"]

        if noise is None:
            noise = randn_tensor((length,), generator=generator, device=self.device) * self.config["s_avg_std"]

        raw_input = noise

        for step, t in enumerate(self.progress_bar(self.scheduler_f.timesteps[start_step:])):
            
            fft_input = torch.fft.fft(raw_input, norm="ortho")[:len(raw_input)//2]
            f_response = DualDiffusionPipeline.get_f_samples(fft_input, self.config["f_resolution"])
            #f_response.cpu().numpy().tofile("./f_unet_input.raw")
            f_response = f_response.type(torch.float16)
            model_output = torch.zeros_like(f_response)

            for i in range(0, f_response.shape[0], batch_size):
                next_batch_size = min(batch_size, f_response.shape[0] - i)
                model_output[i:i+next_batch_size] = self.unet_f(f_response[i:i+next_batch_size], t)["sample"]
            #model_output.type(torch.float32).cpu().numpy().tofile("./f_unet_output.raw")

            f_reconstruction = torch.zeros_like(fft_input)
            DualDiffusionPipeline.invert_f_samples(model_output.type(torch.float32), f_reconstruction)

            fft_input = torch.view_as_real(fft_input).type(torch.float16)
            f_reconstruction = torch.view_as_real(f_reconstruction).type(torch.float16)
            
            if isinstance(self.scheduler_f, DDIMScheduler):
                fft_input = self.scheduler_f.step(
                    model_output=f_reconstruction,
                    timestep=t,
                    sample=fft_input,
                    eta=eta,
                    generator=step_generator,
                )["prev_sample"]
            else:
                fft_input = self.scheduler_f.step(
                    model_output=f_reconstruction,
                    timestep=t,
                    sample=fft_input,
                    generator=step_generator,
                )["prev_sample"]

            fft_input = torch.view_as_complex(fft_input.type(torch.float32))
            fft_input = F.pad(fft_input, (0, 1))
            raw_input = torch.fft.irfft(fft_input, norm="ortho")
            s_response = DualDiffusionPipeline.get_s_samples(raw_input, self.config["s_resolution"])
            #s_response.cpu().numpy().tofile("./s_unet_input.raw")
            s_response = s_response.type(torch.float16)
            model_output = torch.zeros_like(s_response)

            for i in range(0, s_response.shape[0], batch_size*2):
                next_batch_size = min(batch_size*2, s_response.shape[0] - i)
                model_output[i:i+next_batch_size] = self.unet_s(s_response[i:i+next_batch_size], t)["sample"]
            #model_output.type(torch.float32).cpu().numpy().tofile("./s_unet_output.raw")

            s_reconstruction = torch.zeros_like(raw_input)
            DualDiffusionPipeline.invert_s_samples(model_output.type(torch.float32), s_reconstruction)

            raw_input = raw_input.type(torch.float16)
            s_reconstruction = s_reconstruction.type(torch.float16)

            if isinstance(self.scheduler_f, DDIMScheduler):
                raw_input = self.scheduler_f.step(
                    model_output=s_reconstruction,
                    timestep=t,
                    sample=raw_input,
                    eta=eta,
                    generator=step_generator,
                )["prev_sample"]
            else:
                raw_input = self.scheduler_f.step(
                    model_output=s_reconstruction,
                    timestep=t,
                    sample=raw_input,
                    generator=step_generator,
                )["prev_sample"]

            raw_input = raw_input.type(torch.float32)

        return raw_input

if __name__ == "__main__":
    
    if not torch.cuda.is_available():
        print("Error: PyTorch not compiled with CUDA support or CUDA unavailable")
        exit(1)

    model_path = "./models/dualdiffusion"
    print(f"Loading DualDiffusion model from '{model_path}'...")
    my_pipeline = DualDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda")
    
    #noise = np.fromfile("./dataset/dual/01 - front line base.raw", dtype=np.float32, count=2048*1024)
    #noise = torch.from_numpy(noise).to("cuda")
    
    start = time.time(); output = my_pipeline(batch_size=32, steps=100)#, noise=noise)
    print(f"Time taken: {time.time()-start}")

    existing_output_count = len(glob.glob("./output/*.raw"))
    test_output_path = f"./output/test_output_{existing_output_count}.raw"
    output.cpu().numpy().tofile(test_output_path)
    print(f"Saved output to {test_output_path}")