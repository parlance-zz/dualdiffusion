import json
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from diffusers.models import UNet1DModel
from diffusers.schedulers import PNDMScheduler, DDIMScheduler, DDIMInverseScheduler
from diffusers.pipelines.pipeline_utils import AudioPipelineOutput, DiffusionPipeline
from diffusers.utils import randn_tensor
    
class DualDiffusionPipeline(DiffusionPipeline):

    def __init__(
        self,
        unet_s: UNet1DModel,
        unet_f: UNet1DModel,
        scheduler_s: PNDMScheduler,
        scheduler_f: PNDMScheduler,
        sample_len: int = None,
        s_resolution: int = None,
        f_resolution: int = None,
    ):
        super().__init__()
        self.register_modules(unet_s=unet_s, unet_f=unet_f, scheduler_s=scheduler_s, scheduler_f=scheduler_f)

        if sample_len is not None: self.config["sample_len"] = sample_len
        if s_resolution is not None: self.config["s_resolution"] = s_resolution
        if f_resolution is not None: self.config["f_resolution"] = f_resolution

    def get_default_steps(self) -> int:
        return 50

    @staticmethod
    def create_new(data_cfg_path, save_model_path):

        with open(data_cfg_path, "rb") as f:
            dataset_cfg = json.load(f)

        # model hyper-params, todo: no clue if these are optimal, or even good

        scheduler_s = PNDMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            skip_prk_steps=True,
            steps_offset=1,
        )
        scheduler_f = PNDMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            skip_prk_steps=True,
            steps_offset=1,
        )

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
                            in_channels=1,
                            out_channels=1,
        #                     in_channels=2,
        #                     out_channels=2,        
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
        return (torch.ones(resolution, device="cuda") + torch.cos(torch.arange(0, resolution, device="cuda") / resolution * 2. * np.pi - np.pi)) * 0.5
    
    @staticmethod
    def get_windows(window_offsets, resolution):
        windows = DualDiffusionPipeline.get_window(resolution).unsqueeze(0).expand(window_offsets.shape[0], resolution).clone()
        windows[ 0,:resolution//2]  = 1. # left edge
        windows[-1,-resolution//2:] = 1. # right edge

        return windows

    @staticmethod
    def get_s_samples(raw_input, s_resolution):
        window_offsets = DualDiffusionPipeline.get_window_offsets(s_resolution, 2, len(raw_input))
        response_indices = torch.arange(0, s_resolution, device="cuda").view(1, -1) + window_offsets.view(-1, 1)

        response = raw_input[response_indices]
        response -= response.mean(dim=-1, keepdim=True) # each individual response should have zero mean
        return torch.view_as_real(response).permute(0, 2, 1)
    
    @staticmethod
    def invert_s_samples(s_response, raw_input):
        s_resolution = s_response.shape[2]
        window_offsets = DualDiffusionPipeline.get_window_offsets(s_resolution, 2, len(raw_input))
        windows = DualDiffusionPipeline.get_windows(window_offsets, s_resolution)
        s_response = torch.view_as_complex(s_response.permute(0, 2, 1))

        window_offsets_even = window_offsets[::2]
        s_response_even = s_response[::2]
        windows_even = windows[::2]
        response_indices_even = torch.arange(0, s_resolution, device="cuda").view(1, -1) + window_offsets_even.view(-1, 1)

        window_offsets_odd = window_offsets[1::2]
        s_response_odd = s_response[1::2]
        windows_odd = windows[1::2]
        response_indices_odd = torch.arange(0, s_resolution, device="cuda").view(1, -1) + window_offsets_odd.view(-1, 1)

        raw_input[response_indices_even] = s_response_even * windows_even
        raw_input[response_indices_odd] += s_response_odd * windows_odd

        return raw_input

    @staticmethod
    def get_f_samples(fft_input, f_resolution, abs=True):
        window_offsets = DualDiffusionPipeline.get_window_offsets(f_resolution, 2, len(fft_input))

        response_indices = torch.arange(0, f_resolution, device="cuda").view(1, -1) + window_offsets.view(-1, 1)
        #response = torch.fft.ifft(fft_input[response_indices], norm="ortho")
        #response = fft_input[response_indices]
        #return torch.view_as_real(response).permute(0, 2, 1)

        #response = fft_input[response_indices]
        #response[:, response.shape[1]//2:] = 0.
        #response = torch.fft.ifft(response, norm="ortho")
        #response -= response.mean(dim=-1, keepdim=True)
        #return torch.view_as_real(response).permute(0, 2, 1)
        #"""
        if abs:
            response = torch.abs(torch.fft.ifft(fft_input[response_indices], norm="ortho"))
            #response[::2, ::2] *= -1.
            #response[1::2, 1::2] *= -1.
            response -= response.mean(dim=-1, keepdim=True)
            return response.unsqueeze(1)
        else:
            return torch.fft.ifft(fft_input[response_indices], norm="ortho")
        #"""

    @staticmethod
    def invert_f_samples(f_response, fft_input):

        fft_input_response = DualDiffusionPipeline.get_f_samples(fft_input, f_response.shape[2], abs=False)
        
        f_resolution = f_response.shape[2]
        window_offsets = DualDiffusionPipeline.get_window_offsets(f_resolution, 2, len(fft_input))
        windows = DualDiffusionPipeline.get_windows(window_offsets, f_resolution)
        #f_response = torch.view_as_complex(f_response.permute(0, 2, 1))
        
        #f_response = torch.clip(f_response.squeeze(1), min=0., max=None)
        #f_response = torch.abs(f_response.squeeze(1))
        f_response = f_response.squeeze(1)
        f_response = f_response - torch.min(f_response, dim=-1, keepdim=True)[0]

        window_offsets_even = window_offsets[::2]
        f_response_even = f_response[::2]
        windows_even = windows[::2]
        response_indices_even = torch.arange(0, f_resolution, device="cuda").view(1, -1) + window_offsets_even.view(-1, 1)
        fft_input_even = torch.fft.fft(f_response_even, norm="ortho")
        #fft_input_even[:, fft_input_even.shape[1]//2:] = 0.
        fft_input_even = torch.fft.fft(fft_input_response[::2] / torch.abs(fft_input_response[::2]) * f_response_even, norm="ortho")

        window_offsets_odd = window_offsets[1::2]
        f_response_odd = f_response[1::2]
        windows_odd = windows[1::2]
        response_indices_odd = torch.arange(0, f_resolution, device="cuda").view(1, -1) + window_offsets_odd.view(-1, 1)
        fft_input_odd = torch.fft.fft(f_response_odd, norm="ortho")
        #fft_input_odd[:, fft_input_odd.shape[1]//2:] = 0.
        fft_input_odd = torch.fft.fft(fft_input_response[1::2] / torch.abs(fft_input_response[1::2]) * f_response_odd, norm="ortho")

        fft_input[response_indices_even] = fft_input_even * windows_even
        fft_input[response_indices_odd] += fft_input_odd * windows_odd
        #fft_input[response_indices_even] = f_response_even * windows_even
        #fft_input[response_indices_odd] += f_response_odd * windows_odd

        return fft_input

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        length: int = 0,
        start_step: int = 1,
        steps: int = None,
        generator: torch.Generator = None,
        step_generator: torch.Generator = None,
        noise: torch.Tensor = None,
    ) -> Union[
        AudioPipelineOutput,
        Tuple[int, List[np.ndarray]],
    ]:
        self.scheduler_s = DDIMScheduler(clip_sample_range=10000.)
        self.scheduler_f = DDIMScheduler(clip_sample_range=10000.)

        steps = steps or self.get_default_steps()
        self.scheduler_s.set_timesteps(steps)
        self.scheduler_f.set_timesteps(steps)
        step_generator = step_generator or generator

        if length == 0: length = self.config["sample_len"]

        if noise is None:
            assert(False) # todo
        
        raw_input = noise[:length]
        noise.cpu().numpy().tofile(f"./output/noise_input.raw")
        
        f_batch_size = int(self.config["s_resolution"] / self.config["f_resolution"] * batch_size + 0.5)
        f_floor = int(len(raw_input) / self.config["s_resolution"] * 2 + 0.5)

        for step, t in enumerate(self.progress_bar(self.scheduler_f.timesteps[start_step:])):

            # frequency domain
            
            fft_input = torch.fft.fft(raw_input, norm="ortho")[:len(raw_input)//2]
            #"""
            f_response = DualDiffusionPipeline.get_f_samples(fft_input, self.config["f_resolution"])
            #f_response.cpu().numpy().tofile(f"./output/f_unet_input_{step}.raw")
            f_response = f_response.type(torch.float16)

            for i in range(0, f_response.shape[0], f_batch_size):
                next_batch_size = min(f_batch_size, f_response.shape[0] - i)

                model_input = f_response[i:i+next_batch_size]
                model_output = self.unet_f(model_input, t)["sample"]
                
                f_response[i:i+next_batch_size] = self.scheduler_f.step(
                    model_output=model_output,
                    timestep=t,
                    sample=model_input,
                )["prev_sample"]

            f_response = f_response.type(torch.float32)
            #f_response.cpu().numpy().tofile(f"./output/s_unet_output_f_response_{step}.raw")
            fft_input = DualDiffusionPipeline.invert_f_samples(f_response, fft_input)
            #fft_input.cpu().numpy().tofile(f"./output/fft_input_{step}.raw")
            #"""       
            # spatial domain

            fft_input[:f_floor] = 0.
            raw_input = torch.fft.ifft(F.pad(fft_input, (0, len(fft_input))), norm="ortho")
            raw_input = raw_input / raw_input.std() * noise.std()
            #"""
            s_response = DualDiffusionPipeline.get_s_samples(raw_input, self.config["s_resolution"])
            #s_response.cpu().numpy().tofile(f"./output/s_unet_input_{step}.raw")
            s_response = s_response.type(torch.float16)

            for i in range(0, s_response.shape[0], batch_size):
                next_batch_size = min(batch_size, s_response.shape[0] - i)

                model_input = s_response[i:i+next_batch_size]
                model_output = self.unet_s(model_input, t)["sample"]
                
                s_response[i:i+next_batch_size] = self.scheduler_s.step(
                    model_output=model_output,
                    timestep=t,
                    sample=model_input,
                )["prev_sample"]
            
            s_response = s_response.type(torch.float32)
            #s_response.cpu().numpy().tofile(f"./output/s_unet_output_s_response_{step}.raw")
            raw_input = DualDiffusionPipeline.invert_s_samples(s_response, raw_input)
            raw_input = raw_input / raw_input.std() * noise.std()
            #raw_input.cpu().numpy().tofile(f"./output/raw_input_{step}.raw")
            #"""

        return raw_input