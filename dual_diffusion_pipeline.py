import os
from typing import Union
import numpy as np
import torch
import cv2

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
    def create_new(model_params, unet_params, save_model_path):
        
        unet = UNet2DDualModel(**unet_params)

        scheduler = DDIMScheduler(clip_sample_range=20.,
                                  prediction_type=model_params["prediction_type"],
                                  beta_schedule=model_params["beta_schedule"],
                                  beta_start=model_params["beta_start"],
                                  beta_end=model_params["beta_end"],
                                  rescale_betas_zero_snr=model_params["rescale_betas_zero_snr"],)

        pipeline = DualDiffusionPipeline(unet, scheduler, model_params)
        pipeline.save_pretrained(save_model_path, safe_serialization=True)
        return pipeline

    @staticmethod
    def get_sample_crop_width(model_params):
        return model_params["sample_raw_length"]
    
    @staticmethod
    def get_num_channels(model_params):
        freq_embedding_dim = model_params["freq_embedding_dim"]
        channels = model_params["sample_raw_channels"] * 2
        return (channels + freq_embedding_dim, channels)

    @staticmethod
    @torch.no_grad()
    def get_window(window_len):
        x = torch.arange(0, window_len, device="cuda") / (window_len - 1)
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

        num_chunks = model_params["num_chunks"]
        sample_len = model_params["sample_raw_length"]
        half_sample_len = sample_len // 2
        chunk_len = half_sample_len // num_chunks
        bsz = raw_samples.shape[0]

        if window is None:
            window = DualDiffusionPipeline.get_window(chunk_len*2) # this might need to go back to just chunk_len
        raw_samples = raw_samples.clone()
        raw_samples[:, :chunk_len]  *= window[:chunk_len]
        raw_samples[:, -chunk_len:] *= window[chunk_len:]

        fft = torch.fft.fft(raw_samples, norm="ortho")[:, :half_sample_len].view(bsz, num_chunks, chunk_len)
        fft[:, 0, 0] = 0; fft[:, 1:, 0] -= fft[:, 1:, 0].mean()  # remove annoying clicking due to lack of windowing
        samples = torch.view_as_real(torch.fft.fft(fft, norm="ortho")).permute(0, 3, 1, 2).contiguous()
        
        samples /= samples.std(dim=(1, 2, 3), keepdim=True).clip(1e-5)
        return samples, window

    @staticmethod
    @torch.no_grad()
    def sample_to_raw(samples):

        num_chunks = samples.shape[2]
        chunk_len = samples.shape[3]
        sample_len = num_chunks * chunk_len * 2
        half_sample_len = sample_len // 2
        bsz = samples.shape[0]

        samples = torch.view_as_complex(samples.permute(0, 2, 3, 1).contiguous())
        samples = torch.fft.ifft(samples, norm="ortho")
        samples[:, 0, 0] = 0; samples[:, 1:, 0] -= samples[:, 1:, 0].mean() # remove annoying clicking due to lack of windowing
        #a = torch.arange(1, chunk_len, device="cuda") # overkill, apparently. 99% similar to the above result
        #samples[:, :, a] -= samples[:, :, a].mean()
        samples = samples.view(bsz, half_sample_len)
        samples = torch.cat((samples, torch.zeros_like(samples)), dim=1)

        return torch.fft.ifft(samples, norm="ortho") * 2.

    @staticmethod
    @torch.no_grad()
    def save_sample_img(sample, img_path, include_phase=False):
        
        num_chunks = sample.shape[2]
        chunk_len = sample.shape[3]
        bsz = sample.shape[0]

        sample = torch.view_as_complex(sample.clone().detach().permute(0, 2, 3, 1).contiguous().float())
        sample *= torch.arange(num_chunks//8, num_chunks+num_chunks//8).to(sample.device).view(1, num_chunks, 1)
        sample = sample.view(bsz * num_chunks, chunk_len)
        amp = sample.abs(); amp = (amp / torch.max(amp)).cpu().numpy()
        
        if include_phase:
            phase = sample.angle().cpu().numpy()
            cv2_img = np.zeros((bsz * num_chunks, chunk_len, 3), dtype=np.uint8)
            cv2_img[:, :, 0] = (np.sin(phase) + 1) * 255/2 * amp
            cv2_img[:, :, 1] = (np.sin(phase + 2*np.pi/3) + 1) * 255/2 * amp
            cv2_img[:, :, 2] = (np.sin(phase + 4*np.pi/3) + 1) * 255/2 * amp
        else:
            cv2_img = (amp * 255).astype(np.uint8)
            cv2_img = cv2.applyColorMap(cv2_img, cv2.COLORMAP_JET)

        cv2.imwrite(img_path, cv2.flip(cv2_img, -1))

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
        _, num_output_channels = DualDiffusionPipeline.get_num_channels(model_params)
        num_chunks = model_params["num_chunks"]
        default_length = model_params["sample_raw_length"] // num_chunks // 2

        noise = torch.randn((batch_size, num_output_channels, num_chunks, default_length*length,),
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
        
        debug_path = os.environ.get("DEBUG_PATH", None)
        if debug_path is not None:
            print("Sample std: ", sample.std(dim=(1,2,3)).item())
            os.makedirs(debug_path, exist_ok=True)
            sample.cpu().numpy().tofile(os.path.join(debug_path, "debug_sample.raw"))
        
        raw_sample = DualDiffusionPipeline.sample_to_raw(sample)
        raw_sample *= 0.18215 / raw_sample.std(dim=1, keepdim=True).clip(1e-5)
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