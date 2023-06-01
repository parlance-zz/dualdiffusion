from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from diffusers.models import UNet1DModel
from diffusers.schedulers import PNDMScheduler, DDIMScheduler
from diffusers.pipelines.pipeline_utils import AudioPipelineOutput, DiffusionPipeline
    
class SingleDiffusionPipeline(DiffusionPipeline):

    def __init__(
        self,
        unet: UNet1DModel,
        scheduler: PNDMScheduler,
        sample_size: int = None,
        sample_rate: int = None,
        #compress_factor: float = 64.,
    ):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

        if sample_size is not None: self.config["sample_size"] = sample_size
        if sample_rate is not None: self.config["sample_rate"] = sample_rate
        #if compress_factor is not None: self.config["compress_factor"] = compress_factor

    def get_default_steps(self) -> int:
        return 50

    @staticmethod
    def create_new(sample_size, sample_rate, save_model_path):

        # model hyper-params, todo: no clue if these are optimal, or even good

        scheduler = PNDMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            skip_prk_steps=True,
            steps_offset=1,
        )

        unet = UNet1DModel(sample_size=sample_size,
                           in_channels=4,
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
        
        pipeline = SingleDiffusionPipeline(unet,
                                           scheduler,
                                           sample_size=sample_size,
                                           sample_rate=sample_rate,
                                          )

        pipeline.save_pretrained(save_model_path, safe_serialization=True)
        return pipeline

    @staticmethod
    def resample_audio(raw_input, sample_rate, new_sample_rate):
        fft = torch.fft.fft(raw_input, norm="ortho")[:len(raw_input)//2]
        cutoff = int(new_sample_rate / sample_rate * len(fft) + 0.5)
        if cutoff > len(fft):
            fft = torch.cat((fft, torch.zeros(cutoff - len(fft), device=fft.device)))
        else:
            fft = fft[:cutoff]
        fft = torch.cat((fft, torch.zeros(len(fft), device=fft.device)))
        return torch.fft.ifft(fft, norm="ortho")

    @staticmethod
    def get_analytic_audio(raw_input):
        raw_input_max = torch.max(torch.abs(raw_input))
        fft = torch.fft.fft(raw_input, norm="ortho")[:len(raw_input)//2]
        analytic = torch.cat((fft, torch.zeros(len(fft), device=fft.device)))
        analytic = torch.fft.ifft(analytic, norm="ortho")
        analytic /= torch.max(torch.abs(analytic.real))
        analytic *= raw_input_max
        return analytic
    
    """
    @staticmethod
    def compress_audio(raw_input, compress_factor=255.):
        if compress_factor == 0.: return raw_input
        return torch.sign(raw_input) * torch.log(1. + compress_factor * torch.abs(raw_input)) / np.log(1. + compress_factor)
    
    @staticmethod
    def decompress_audio(compressed_input, compress_factor=128.):
        if compress_factor == 0.: return compressed_input
        return (torch.sign(compressed_input) * torch.pow(1. + compress_factor, torch.abs(compressed_input)) - 1.) / compress_factor
    """

    @staticmethod
    def get_noise_from_example(example, sample_size, batch_size):
        assert(len(example) % 2 == 0)
        #example_std = example.std()

        fft = torch.fft.fft(example, norm="ortho")[:len(example)//2]
        noise = torch.exp(torch.rand(len(fft), device=fft.device) * 1j * 2 * np.pi) * torch.abs(fft)
        noise = torch.cat((noise, torch.zeros(len(noise), device=noise.device, dtype=torch.complex64)))
        noise[0] = 0.; noise = torch.fft.ifft(noise, norm="ortho")
        #noise = noise / noise.std() * example_std

        random_indices = torch.randint(0, len(noise) - sample_size, (batch_size, 1)) + torch.arange(0, sample_size).view(1, -1)
        noise = noise[random_indices]
        noise -= noise.mean(dim=-1, keepdim=True)

        return torch.view_as_real(noise).permute(0, 2, 1)

    @torch.no_grad()
    def get_training_sample(self, clean_image, batch_size):
        sample_size = self.config["sample_size"]
        random_indices = torch.randint(0, len(clean_image) - sample_size*2, (batch_size, 1)) + torch.arange(0, sample_size*2).view(1, -1)

        chunk = clean_image[random_indices]
        chunk -= chunk.mean(dim=-1, keepdim=True)

        last_output = torch.view_as_real(chunk[:, :sample_size]).permute(0, 2, 1)
        sample = torch.view_as_real(chunk[:, sample_size:sample_size*2]).permute(0, 2, 1)
        noise = SingleDiffusionPipeline.get_noise_from_example(clean_image, sample_size, batch_size)

        return last_output, sample, noise

    @torch.no_grad()
    def __call__(
        self,
        example: torch.Tensor,
        length: int = 30,
        start_step: int = 1,
        steps: int = 50,
    ) -> Union[
        AudioPipelineOutput,
        Tuple[int, List[np.ndarray]],
    ]:
        #self.scheduler = DDIMScheduler(clip_sample_range=10000.)

        example = example.to(self.device)
        sample_size = self.config["sample_size"]
        output = torch.zeros(length * sample_size, device=self.device, dtype=torch.complex64)
        
        output[:sample_size] = example[32768:32768+sample_size] * 3.14 # ?

        #last_output, sample, noise = self.get_training_sample(example, 1)
        #torch.view_as_complex(last_output.squeeze(0).permute(1, 0)).cpu().numpy().tofile("last_output.raw")
        #torch.view_as_complex(sample.squeeze(0).permute(1, 0)).cpu().numpy().tofile("sample.raw")
        #torch.view_as_complex(noise.squeeze(0).permute(1, 0)).cpu().numpy().tofile("noise.raw")
        #exit(0)

        for chunk in range(1, length):
            
            self.scheduler.set_timesteps(steps)
            
            sample = SingleDiffusionPipeline.get_noise_from_example(example, sample_size, 1).squeeze(0)
            last_output = torch.view_as_real(output[(chunk - 1) * sample_size:chunk * sample_size]).permute(1, 0)

            for step, t in enumerate(self.progress_bar(self.scheduler.timesteps[start_step:])):

                model_input = torch.cat((sample, last_output), dim=0).unsqueeze(0)
                model_output = self.unet(model_input, t)["sample"]
                
                sample = self.scheduler.step(
                    model_output=model_output,
                    timestep=t,
                    sample=sample.unsqueeze(0),
                )["prev_sample"].squeeze(0)

            output[chunk * sample_size:(chunk + 1) * sample_size] = torch.view_as_complex(sample.permute(1, 0))

        return output