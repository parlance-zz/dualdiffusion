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
from training.sigma_sampler import SigmaSampler, SigmaSamplerConfig
from utils.dual_diffusion_utils import (
    load_audio, tensor_info_str, save_audio, init_cuda, save_tensor_raw
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

    crop_width = mclt_format.sample_raw_crop_width()
    test_output_path = os.path.join(config.DEBUG_PATH, "mclt_test")

    if cfg.add_random_test_samples > 0:
        train_samples = config.load_json(os.path.join(config.DATASET_PATH, "train.jsonl"))
        cfg.test_samples += [sample["file_name"] for sample in
                             random.sample(train_samples, cfg.add_random_test_samples)]

    psd = None
    avg_std = 0

    for i, sample_filename in enumerate(cfg.test_samples):
        base_filename = os.path.splitext(os.path.basename(sample_filename))[0]

        input_raw_sample, sample_rate = load_audio(os.path.join(config.DATASET_PATH, sample_filename),
                            start=0, count=crop_width, device=cfg.device, return_sample_rate=True)
        input_raw_sample.unsqueeze_(0)
        assert sample_rate == mclt_format.config.sample_rate
        assert input_raw_sample.shape[1] == mclt_format.config.sample_raw_channels

        mclt_sample = mclt_format.raw_to_sample(input_raw_sample)
        output_raw_sample = mclt_format.sample_to_raw(mclt_sample + torch.randn_like(mclt_sample) * cfg.add_noise)

        avg_std += mclt_sample.std()
        if psd is None: psd = mclt_sample.abs().square().to(dtype=torch.float64)# * (torch.arange(0, 256, device=mclt_sample.device).view(1, 1,-1, 1)+0.5) * (70/256)
        else: psd += mclt_sample.abs().square().to(dtype=torch.float64) #* (torch.arange(0, 256, device=mclt_sample.device).view(1, 1,-1, 1)+0.5) * (70/256)

        print(f"({i}/{len(cfg.test_samples)}) {sample_filename}:")
        print("  mlct_sample:", tensor_info_str(mclt_sample))
        print("  input_raw_sample: ", tensor_info_str(input_raw_sample))
        print("  output_raw_sample:", tensor_info_str(output_raw_sample))

        if cfg.save_output == True:
            output_flac_file_path = os.path.join(test_output_path, f"{base_filename}_original.flac")
            save_audio(input_raw_sample.squeeze(0), mclt_format.config.sample_rate, output_flac_file_path, target_lufs=None)
            print(f"  Saved flac output to {output_flac_file_path}")

            output_flac_file_path = os.path.join(test_output_path, f"{base_filename}_reconstructed.flac")
            save_audio(output_raw_sample.squeeze(0), mclt_format.config.sample_rate, output_flac_file_path, target_lufs=None)
            print(f"  Saved flac output to {output_flac_file_path}")

        print("")

    print(f"avg mdct std: {avg_std / len(cfg.test_samples)}")
    #exit()

    psd = (psd / len(cfg.test_samples)).mean(dim=(1,3)).sqrt().flatten()
    sigma_sampler = SigmaSampler(cfg.sigma_config)
    sigma_psd = sigma_sampler.sample(psd.shape[0], "cpu")

    save_tensor_raw(psd.log().float(), os.path.join(test_output_path, "ln_psd_mclt.raw"))
    save_tensor_raw(sigma_psd.log(), os.path.join(test_output_path, "ln_psd_sigma.raw"))
    save_tensor_raw(sigma_psd.log() - psd.log().cpu().float(), os.path.join(test_output_path, "ln_psd_diff.raw"))

    print(f"avg power spectral density mclt:  {psd.square().mean().sqrt().item()}")
    print(f"avg power spectral density sigma: {sigma_psd.square().mean().sqrt().item()}")
    print("median power spectral density mclt:", psd[psd.shape[0]//2].item())
    print("median power spectral density sigma:", sigma_psd[psd.shape[0]//2].item())
    
    masking = torch.zeros(psd.shape[0])

    for i in range(psd.shape[0]):
        hz = (0.5 + i) / psd.shape[0] * mclt_format.config.sample_rate / 2
        masking[i] = (sigma_psd[i] > psd).float().mean()
        print(f"  {hz}hz  mclt: {psd[i].item():.4f}  sigma: {sigma_psd[i].item():.4f}  "
              f"diff: {(sigma_psd[i]-psd[i]).item():.4f}  masking: {(masking[i]).item():.4f}")
        
    save_tensor_raw(masking, os.path.join(test_output_path, "masking.raw"))


if __name__ == "__main__":

    init_cuda()
    mclt_test()