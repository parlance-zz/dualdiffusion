import json
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from diffusers.models import UNet1DModel, UNet2DModel
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

        """
        unet_s = UNet2DModel(
            sample_size=dataset_cfg["s_resolution"],
            in_channels=2,
            out_channels=2,
            layers_per_block=2,
            attention_head_dim=8,
            block_out_channels=(256, 384, 512, 768, 1024),
            down_block_types=(
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
            ),
        )
        unet_f = UNet2DModel(
            sample_size=dataset_cfg["f_resolution"],
            in_channels=2,
            out_channels=2,
            layers_per_block=2,
            attention_head_dim=8,
            block_out_channels=(256, 384, 512, 768, 1024),
            down_block_types=(
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
            ),
        )
        """

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
        windows[ 0,:resolution//2]  = 1.
        windows[-1,-resolution//2:] = 1.

        return windows
    
    """
    @staticmethod
    def complex2color(complex):

        phase = torch.angle(complex)
        magnitude = torch.abs(complex)

        r_phase = phase + 0.
        g_phase = phase + 2. * np.pi / 3.
        b_phase = phase + 4. * np.pi / 3.

        color_tensor = torch.zeros((complex.shape[0], 3, complex.shape[1]), device="cuda")
        color_tensor[:, 0, :] = magnitude * (torch.cos(r_phase) + 1.) * 0.5
        color_tensor[:, 1, :] = magnitude * (torch.cos(g_phase) + 1.) * 0.5
        color_tensor[:, 2, :] = magnitude * (torch.cos(b_phase) + 1.) * 0.5

        return color_tensor
    """

    @staticmethod
    def get_s_samples(raw_input, s_resolution):

        #window_offsets = DualDiffusionPipeline.get_window_offsets(s_resolution*2, 2, len(raw_input))
        #response_indices = torch.arange(0, s_resolution*2, device="cuda").view(1, -1) + window_offsets.view(-1, 1)
        #response = torch.fft.fft(raw_input[response_indices], norm="ortho")[:, :s_resolution]
        #response[:, 0] = 0.

        #return torch.view_as_real(response).permute(0, 2, 1)

        window_offsets = DualDiffusionPipeline.get_window_offsets(s_resolution, 2, len(raw_input))
        response_indices = torch.arange(0, s_resolution, device="cuda").view(1, -1) + window_offsets.view(-1, 1)
        response = raw_input[response_indices]
        response -= response.mean(dim=-1, keepdim=True) # each individual response should have zero mean

        return torch.view_as_real(response).permute(0, 2, 1)
    
    @staticmethod
    def invert_s_samples(s_response, raw_input):

        s_resolution = s_response.shape[2]
        #window_offsets = DualDiffusionPipeline.get_window_offsets(s_resolution*2, 2, len(raw_input))
        #windows = DualDiffusionPipeline.get_windows(window_offsets, s_resolution*2)
        window_offsets = DualDiffusionPipeline.get_window_offsets(s_resolution, 2, len(raw_input))
        windows = DualDiffusionPipeline.get_windows(window_offsets, s_resolution)

        s_response = torch.view_as_complex(s_response.permute(0, 2, 1))
        #s_response = torch.cat((s_response, torch.zeros_like(s_response)), dim=-1)

        window_offsets_even = window_offsets[::2]
        s_response_even = s_response[::2]
        windows_even = windows[::2]
        #response_indices_even = torch.arange(0, s_resolution*2, device="cuda").view(1, -1) + window_offsets_even.view(-1, 1)
        response_indices_even = torch.arange(0, s_resolution, device="cuda").view(1, -1) + window_offsets_even.view(-1, 1)
        #raw_input_ifft_even = torch.fft.ifft(s_response_even, norm="ortho").real

        window_offsets_odd = window_offsets[1::2]
        s_response_odd = s_response[1::2]
        windows_odd = windows[1::2]
        #response_indices_odd = torch.arange(0, s_resolution*2, device="cuda").view(1, -1) + window_offsets_odd.view(-1, 1)
        response_indices_odd = torch.arange(0, s_resolution, device="cuda").view(1, -1) + window_offsets_odd.view(-1, 1)
        #raw_input_ifft_odd = torch.fft.ifft(s_response_odd, norm="ortho").real

        #raw_input[response_indices_even] = raw_input_ifft_even * windows_even
        #raw_input[response_indices_odd] += raw_input_ifft_odd * windows_odd
        raw_input[response_indices_even] = s_response_even * windows_even
        raw_input[response_indices_odd] += s_response_odd * windows_odd

        return raw_input

    @staticmethod
    def get_f_samples(fft_input, f_resolution):

        window_offsets = DualDiffusionPipeline.get_window_offsets(f_resolution, 2, len(fft_input))
        response_indices = torch.arange(0, f_resolution, device="cuda").view(1, -1) + window_offsets.view(-1, 1)
        response = torch.fft.ifft(fft_input[response_indices], norm="ortho")

        return torch.view_as_real(response).permute(0, 2, 1)
    
    @staticmethod
    def invert_f_samples(f_response, fft_input):

        f_resolution = f_response.shape[2]
        window_offsets = DualDiffusionPipeline.get_window_offsets(f_resolution, 2, len(fft_input))
        windows = DualDiffusionPipeline.get_windows(window_offsets, f_resolution)

        f_response = torch.view_as_complex(f_response.permute(0, 2, 1))

        window_offsets_even = window_offsets[::2]
        f_response_even = f_response[::2]
        windows_even = windows[::2]
        response_indices_even = torch.arange(0, f_resolution, device="cuda").view(1, -1) + window_offsets_even.view(-1, 1)
        fft_input_even = torch.fft.fft(f_response_even, norm="ortho")

        window_offsets_odd = window_offsets[1::2]
        f_response_odd = f_response[1::2]
        windows_odd = windows[1::2]
        response_indices_odd = torch.arange(0, f_resolution, device="cuda").view(1, -1) + window_offsets_odd.view(-1, 1)
        fft_input_odd = torch.fft.fft(f_response_odd, norm="ortho")

        fft_input[response_indices_even] = fft_input_even * windows_even
        fft_input[response_indices_odd] += fft_input_odd * windows_odd

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
        self.scheduler_s = DDIMScheduler(clip_sample_range=10000.)
        self.scheduler_f = DDIMScheduler(clip_sample_range=10000.)

        steps = steps or self.get_default_steps()
        self.scheduler_s.set_timesteps(steps)
        self.scheduler_f.set_timesteps(steps)
        step_generator = step_generator or generator

        if length == 0: length = self.config["sample_len"]

        if noise is None:
            noise = randn_tensor((length,), generator=generator, device=self.device)
            noise_fft = torch.fft.fft(noise, norm="ortho")
            noise_fft[:len(noise)//2] /= torch.arange(len(noise)//2, device=noise.device)
            noise_fft[0] = 0.; noise_fft[len(noise)//2:] = 0.
            noise = torch.fft.ifft(noise_fft, norm="ortho")
            noise /= noise.std()
        
        raw_input = noise
        noise.cpu().numpy().tofile(f"./output/noise_input.raw")

        for step, t in enumerate(self.progress_bar(self.scheduler_f.timesteps[start_step:])):
            
            fft_input = torch.fft.fft(raw_input, norm="ortho")[:len(raw_input)//2]

            """
            if step > 0 and (1==2):
                
                f_response = DualDiffusionPipeline.get_f_samples(fft_input, self.config["f_resolution"])
                #f_response.cpu().numpy().tofile("./output/f_unet_input.raw")
                f_response_std = f_response.std(dim=(1, 2))
                f_response /= f_response_std.view(-1, 1, 1)
                f_response = f_response.type(torch.float16)
                model_output = torch.zeros_like(f_response)

                for i in range(0, f_response.shape[0], batch_size):
                    next_batch_size = min(batch_size, f_response.shape[0] - i)
                    model_output[i:i+next_batch_size] = self.unet_f(f_response[i:i+next_batch_size], t)["sample"]
                #model_output.type(torch.float32).cpu().numpy().tofile("./output/f_unet_output.raw")

                model_output *= f_response_std.view(-1, 1, 1)
                f_reconstruction = torch.zeros_like(fft_input)
                f_reconstruction = DualDiffusionPipeline.invert_f_samples(model_output.type(torch.float32), f_reconstruction)

                fft_input = torch.view_as_real(fft_input).type(torch.float16)
                f_reconstruction = torch.view_as_real(f_reconstruction).type(torch.float16)


                fft_input = self.scheduler_f.step(
                    model_output=f_reconstruction,
                    timestep=t,
                    sample=fft_input,
                    generator=step_generator,
                )["prev_sample"]
                
                fft_input = torch.view_as_complex(fft_input.type(torch.float32))
            """

            #fft_input.cpu().numpy().tofile(f"./output/fft_input_{step}.raw")
            raw_input = torch.fft.ifft(F.pad(fft_input, (0, len(fft_input))), norm="ortho")
            #raw_input -= raw_input.mean()
            #raw_input /= raw_input.std()

            s_response = DualDiffusionPipeline.get_s_samples(raw_input, self.config["s_resolution"])
            s_response.cpu().numpy().tofile(f"./output/s_unet_input_{step}.raw")
            s_response = s_response.type(torch.float16)

            for i in range(0, s_response.shape[0], batch_size):
                
                next_batch_size = min(batch_size, s_response.shape[0] - i)

                model_input = s_response[i:i+next_batch_size]
                model_output = self.unet_s(model_input, t)["sample"]
                
                model_input = self.scheduler_s.step(
                    model_output=model_output,
                    timestep=t,
                    sample=model_input,
                )["prev_sample"]

                s_response[i:i+next_batch_size] = model_input
            
            s_response = s_response.type(torch.float32)
            s_response.cpu().numpy().tofile(f"./output/s_unet_output_s_response_{step}.raw")

            raw_input = DualDiffusionPipeline.invert_s_samples(s_response, raw_input)
            raw_input.cpu().numpy().tofile(f"./output/raw_input_{step}.raw")

        return raw_input