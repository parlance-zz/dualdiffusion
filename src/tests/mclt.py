# MIT License
#
# Copyright (c) 2023 Christopher Friesen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import utils.config as config

from dataclasses import dataclass
import os
import random

import torch

from modules.formats.mclt import DualMCLTFormat, DualMCLTFormatConfig
from modules.formats.spectrogram import SpectrogramFormat, SpectrogramFormatConfig
from modules.mp_tools import wavelet_decompose, wavelet_recompose
from training.sigma_sampler import SigmaSampler, SigmaSamplerConfig
from utils.dual_diffusion_utils import (
    load_audio, tensor_info_str, save_audio, init_cuda, save_tensor_raw,
    tensor_to_img, save_img
)


@dataclass
class MCLT_TestConfig:
    device: str
    format_config: DualMCLTFormatConfig
    sigma_config: SigmaSamplerConfig
    add_noise: float
    save_output: bool
    add_random_test_samples: int
    test_samples: list[str]

@torch.inference_mode()
def mclt_test() -> None:
    
    cfg: MCLT_TestConfig = config.load_config(
        MCLT_TestConfig, os.path.join(config.CONFIG_PATH, "tests", "mclt.json"))

    mclt_format: DualMCLTFormat = DualMCLTFormat(cfg.format_config).to(device=cfg.device)
    spectrogram_format: SpectrogramFormat = SpectrogramFormat(SpectrogramFormatConfig(abs_exponent=0.5, raw_to_sample_scale=1, sample_to_raw_scale=1)).to(device=cfg.device)
    crop_width = spectrogram_format.sample_raw_crop_width(length=cfg.format_config.sample_raw_length)
    test_output_path = os.path.join(config.DEBUG_PATH, "mclt_test")

    if cfg.add_random_test_samples > 0:
        train_samples = config.load_json(os.path.join(config.DATASET_PATH, "train.jsonl"))
        cfg.test_samples += [sample["file_name"] for sample in
                             random.sample(train_samples, cfg.add_random_test_samples)]

    mclt_psd = None
    mclt_avg_std = 0
    spectrogram_psd = None
    spectrogram_avg_std = 0
    import numpy as np
    ln_psd_mclt_raw = np.fromfile(os.path.join(config.DEBUG_PATH, "mclt_test", "ln_psd_mclt.raw"), dtype=np.float32)
    recorded_psd = torch.Tensor(ln_psd_mclt_raw).exp().to(device=cfg.device).view(1, 1,-1, 1)
    #wavelet_avg_std = torch.zeros(4, device=cfg.device, dtype=torch.float64)

    for i, sample_filename in enumerate(cfg.test_samples):
        base_filename = os.path.splitext(os.path.basename(sample_filename))[0]

        input_raw_sample, sample_rate = load_audio(os.path.join(config.DATASET_PATH, sample_filename),
                            start=0, count=crop_width, device=cfg.device, return_sample_rate=True)
        input_raw_sample.unsqueeze_(0)
        assert sample_rate == mclt_format.config.sample_rate
        assert input_raw_sample.shape[1] == mclt_format.config.sample_raw_channels

        print(f"({i}/{len(cfg.test_samples)}) {sample_filename}:")
        print("  input_raw_sample: ", tensor_info_str(input_raw_sample))

        mclt_sample = mclt_format.raw_to_sample(input_raw_sample)
        mclt_sample /= recorded_psd
        print("  mlct_sample:", tensor_info_str(mclt_sample))
        output_raw_sample = mclt_format.sample_to_raw(mclt_sample + torch.randn_like(mclt_sample) * cfg.add_noise)
        print("  output_raw_sample:", tensor_info_str(output_raw_sample))

        mclt_avg_std += mclt_sample.std()
        if mclt_psd is None: mclt_psd = mclt_sample.abs().square().to(dtype=torch.float64)
        else: mclt_psd += mclt_sample.abs().square().to(dtype=torch.float64)

        if cfg.save_output == True:
            output_flac_file_path = os.path.join(test_output_path, f"{base_filename}_original.flac")
            save_audio(input_raw_sample.squeeze(0), mclt_format.config.sample_rate, output_flac_file_path, target_lufs=None)
            print(f"  Saved flac output to {output_flac_file_path}")

            output_flac_file_path = os.path.join(test_output_path, f"{base_filename}_mclt_reconstructed.flac")
            save_audio(output_raw_sample.squeeze(0), mclt_format.config.sample_rate, output_flac_file_path, target_lufs=None)
            print(f"  Saved flac output to {output_flac_file_path}")

        
        spectrogram_sample = spectrogram_format.raw_to_sample(input_raw_sample)
        print("  spectrogram_sample:", tensor_info_str(spectrogram_sample))

        spectrogram_avg_std += spectrogram_sample.std()
        if spectrogram_psd is None: spectrogram_psd = spectrogram_sample.square().to(dtype=torch.float64)
        else: spectrogram_psd += spectrogram_sample.square().to(dtype=torch.float64)

        """
        wavelets = wavelet_decompose(spectrogram_sample)
        for i, wavelet in enumerate(wavelets):
            wavelet_avg_std[i] += wavelet.std().to(dtype=torch.float64)
                
        if cfg.save_output == True:
            #spectrogram_sample = spectrogram_format.convert_to_abs_exp1(spectrogram_sample)
            reconstructed = wavelet_recompose(wavelets)

            save_img(tensor_to_img(spectrogram_sample, flip_y=True), os.path.join(test_output_path, f"{base_filename}_spectrogram.png"))
            save_img(tensor_to_img(reconstructed, flip_y=True), os.path.join(test_output_path, f"{base_filename}_wavelet_reconstructed.png"))
            for i, wavelet in enumerate(wavelets):
                save_img(tensor_to_img(wavelet, flip_y=True), os.path.join(test_output_path, f"{base_filename}_wavelet_{i}.png"))

            output_raw_sample = spectrogram_format.sample_to_raw(reconstructed + torch.randn_like(reconstructed) * cfg.add_noise)
            print("  output_raw_sample:", tensor_info_str(output_raw_sample))
            output_flac_file_path = os.path.join(test_output_path, f"{base_filename}_spectrogram_reconstructed.flac")
            save_audio(output_raw_sample.squeeze(0), spectrogram_format.config.sample_rate, output_flac_file_path, target_lufs=None)
            print(f"  Saved flac output to {output_flac_file_path}")
        """

        print("")

    print(f"avg mdct std: {mclt_avg_std / len(cfg.test_samples)}")
    print(f"avg spectrogram std: {spectrogram_avg_std / len(cfg.test_samples)}")
    #for i in range(wavelet_avg_std.shape[0]):
    #    print(f"avg wavelet_{i} std: {wavelet_avg_std[i] / len(cfg.test_samples)}")
    #exit()

    mclt_psd = (mclt_psd / len(cfg.test_samples)).mean(dim=(1,3)).sqrt().flatten()
    #spectrogram_psd = (spectrogram_psd / len(cfg.test_samples)).mean(dim=(1,3)).sqrt().flatten()
    sigma_sampler = SigmaSampler(cfg.sigma_config)
    sigma_psd = sigma_sampler.sample(mclt_psd.shape[0], "cpu")

    save_tensor_raw(mclt_psd.log().float(), os.path.join(test_output_path, "ln_psd_mclt.raw"))
    save_tensor_raw(spectrogram_psd.log().float(), os.path.join(test_output_path, "ln_psd_spectrogram.raw"))
    save_tensor_raw(sigma_psd.log(), os.path.join(test_output_path, "ln_psd_sigma.raw"))

    print(f"avg power spectral density mclt:  {mclt_psd.square().mean().sqrt().item()}")
    #print(f"avg power spectral density spectrogram:  {spectrogram_psd.square().mean().sqrt().item()}")
    print(f"avg power spectral density sigma: {sigma_psd.square().mean().sqrt().item()}")
    print("median power spectral density mclt:", mclt_psd[mclt_psd.shape[0]//2].item())
    #print("median power spectral density spectrogram:", spectrogram_psd[spectrogram_psd.shape[0]//2].item())
    print("median power spectral density sigma:", sigma_psd[mclt_psd.shape[0]//2].item())

    print(mclt_psd)


if __name__ == "__main__":

    init_cuda()
    mclt_test()