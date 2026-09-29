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

from utils import config

from dataclasses import dataclass
import os
import random

import torch

from modules.formats.ms_mdct_dual_6 import MS_MDCT_DualFormat, MS_MDCT_DualFormatConfig
from training.trainer import TrainLogger as StatLogger
from utils.dual_diffusion_utils import (
    init_cuda, save_audio, load_audio, tensor_to_img,
    save_img, get_audio_info, dict_str, tensor_info_str
)


@dataclass
class MS_MDCT_DualFormat_TestConfig:

    device: str
    save_output: bool
    test_sample_verbose: bool

    add_random_test_samples: int
    test_samples: list[str]

    format_config: MS_MDCT_DualFormatConfig

@torch.inference_mode()
def ms_mdct_dual_format_test() -> None:

    torch.manual_seed(0)
    random.seed()

    cfg: MS_MDCT_DualFormat_TestConfig = config.load_config(MS_MDCT_DualFormat_TestConfig,
        os.path.join(config.CONFIG_PATH, "tests", "ms_mdct_dual_format_6.json"))
    
    output_path = os.path.join(config.DEBUG_PATH, "ms_mdct_dual_format_6_test")
    os.makedirs(output_path, exist_ok=True)

    format_load_path = os.path.join(output_path, "format")
    format_loaded = False
    if os.path.isdir(format_load_path):
        if input(f"\nLoad saved format? ({format_load_path}) (y/n): ") == "y":
            format: MS_MDCT_DualFormat = MS_MDCT_DualFormat.from_pretrained(output_path, subfolder="format", device=cfg.device)
            format_loaded = True

    if format_loaded == False:
        format: MS_MDCT_DualFormat = MS_MDCT_DualFormat(cfg.format_config).to(cfg.device)

    print("Format config:")
    print(dict_str(format.config.__dict__))
    
    dataset_path = config.DATASET_PATH
    test_samples = cfg.test_samples

    if cfg.add_random_test_samples > 0:
        train_samples = config.load_json(os.path.join(config.DATASET_PATH, "train.jsonl"))
        test_samples += [sample["file_name"] for sample in random.sample(train_samples, cfg.add_random_test_samples)]

    mdct_phase_avg_bin_msq = torch.zeros(format.config.mdct_num_frequencies, dtype=torch.float64, device=cfg.device).view(1, 1,-1, 1)
    mdct_psd_avg_bin_msq = torch.zeros(format.config.mdct_num_frequencies, dtype=torch.float64, device=cfg.device).view(1, 1,-1, 1)
    mdct_psd_avg_bin_mean = torch.zeros(format.config.mdct_num_frequencies, dtype=torch.float64, device=cfg.device).view(1, 1,-1, 1)

    ms_psd_avg_bin_msqs: list[torch.Tensor] = []; ms_psd_avg_bin_means: list[torch.Tensor] = []
    for i in range(format.config.num_ms_psds):
        num_ms_psd_freqs = format.config.ms_psds[i].num_frequencies
        ms_psd_avg_bin_msqs.append( torch.zeros(num_ms_psd_freqs, dtype=torch.float64, device=cfg.device).view(1, 1,-1, 1))
        ms_psd_avg_bin_means.append(torch.zeros(num_ms_psd_freqs, dtype=torch.float64, device=cfg.device).view(1, 1,-1, 1))

    r_dims = (0, 1, 3) if format.config.use_per_freq_preconditioning == True else (0, 1, 2, 3)

    stat_logger = StatLogger()
    print(f"\nNum test_samples: {len(test_samples)}\n")

    for i, filename in enumerate(test_samples):
        
        print(f"pass 1 - file {i+1}/{len(test_samples)}: {filename}")

        file_path = os.path.join(dataset_path, filename)
        if os.path.isfile(file_path) == False:
            file_path = os.path.join(config.DEBUG_PATH, filename)

        raw_length = min(get_audio_info(file_path).frames, cfg.format_config.default_raw_length)
        crop_width = format.get_raw_crop_width(raw_length)

        start = -1 if cfg.save_output == False else 0
        raw_sample = load_audio(file_path, start=start, count=crop_width).unsqueeze(0).to(cfg.device)

        ms_psds = format.raw_to_ms_psd(raw_sample, level=-1)

        mdct_phase_psd = format.raw_to_mdct_phase_psd(raw_sample)
        raw_sample_recon = format.mdct_phase_psd_to_raw(mdct_phase_psd)

        mdct_phase, mdct_psd = mdct_phase_psd.to(dtype=torch.float64).chunk(2, dim=1)
        mdct_phase_avg_bin_msq += mdct_phase.pow(2).mean(dim=r_dims, keepdim=True) / len(test_samples)
        mdct_psd_avg_bin_msq  += mdct_psd.pow(2).mean(dim=r_dims, keepdim=True) / len(test_samples)
        mdct_psd_avg_bin_mean += mdct_psd.mean(dim=r_dims, keepdim=True) / len(test_samples)
    
        for i in range(format.config.num_ms_psds):

            _ms_psd = ms_psds[i].to(dtype=torch.float64)
            _ms_psd_msq = _ms_psd.pow(2).mean(dim=r_dims, keepdim=True)
            _ms_psd_mean = _ms_psd.mean(dim=r_dims, keepdim=True)

            ms_psd_avg_bin_msqs[i]  += _ms_psd_msq / len(test_samples)
            ms_psd_avg_bin_means[i] += _ms_psd_mean / len(test_samples)
            stat_logger.add_logs({f"ms_psd_{i}_msq": _ms_psd_msq.mean()})
            stat_logger.add_logs({f"ms_psd_{i}_mean": _ms_psd_mean.mean()})
            
        stat_logger.add_logs({
            "raw_sample_msq": raw_sample.pow(2).mean(),
            "raw_sample_recon_msq": raw_sample_recon.pow(2).mean(),
            "recon_error_rms": (raw_sample - raw_sample_recon).pow(2).mean().pow(0.5),

            "mdct_phase_psd_msq": mdct_phase_psd.pow(2).mean(),
            "mdct_phase_psd_mean": mdct_phase_psd.mean(),
            "mdct_phase_msq": mdct_phase.pow(2).mean(),
            "mdct_psd_msq": mdct_psd.pow(2).mean(),
            "mdct_psd_mean": mdct_psd.mean(),
        })

        if cfg.test_sample_verbose == True:
            print("raw_sample:", tensor_info_str(raw_sample))
            print("raw_sample_recon:", tensor_info_str(raw_sample_recon), "\n")
            print("mdct_phase_psd:", tensor_info_str(mdct_phase_psd), f"(target shape: {format.get_mdct_phase_psd_shape(raw_length=raw_length)}")

            for i in range(format.config.num_ms_psds):
                print(f"ms_psd_{i}:", tensor_info_str(ms_psds[i]))

        if cfg.save_output == False:
            continue

        filename = os.path.splitext(os.path.basename(filename))[0]

        raw_sample_output_path = os.path.join(output_path, f"{filename}.flac")
        save_audio(raw_sample.squeeze(0), cfg.format_config.sample_rate, raw_sample_output_path, target_lufs=None)
        print(f"Saved raw_sample to {raw_sample_output_path}")

        mdct_psd_output_path = os.path.join(output_path, f"{filename}_mdct_psd.png")
        save_img(tensor_to_img(mdct_psd, flip_y=True), mdct_psd_output_path)
        print(f"Saved mdct_psd img to {mdct_psd_output_path}")

        recon_output_path = os.path.join(output_path, f"{filename}_recon.flac")
        save_audio(raw_sample_recon.squeeze(0), cfg.format_config.sample_rate, recon_output_path, target_lufs=None)
        print(f"Saved raw_sample_recon to {recon_output_path}")

        for i in range(format.config.num_ms_psds):
            ms_psd_output_path = os.path.join(output_path, f"{filename}_ms_psd_{i}.png")
            save_img(format.ms_psd_to_img(ms_psds[i]), ms_psd_output_path)
            print(f"Saved ms_psd_{i} img to {ms_psd_output_path}")

    if format_loaded == False:
        
        # copy our calculated means for mdct/ms psd to format buffers, zero the psd msqs for recalculation in 2nd pass
        format.mdct_phase_scale.copy_(mdct_phase_avg_bin_msq.pow(0.5).float())
        format.mdct_psd_offset.copy_(-mdct_psd_avg_bin_mean.float())
        mdct_psd_avg_bin_msq.zero_()
        
        for i in range(format.config.num_ms_psds):
            getattr(format, f"ms_psd_offset_{i}").copy_(-ms_psd_avg_bin_means[i].float())
            ms_psd_avg_bin_msqs[i].zero_()

    else:

        stat_logs = stat_logger.get_logs()
        print("\nAverage stats:")
        print(dict_str(stat_logs))

        #mdct_phase_avg_bin_msq.pow(0.5).float().cpu().numpy().tofile(os.path.join(output_path, "mdct_phase_scale_loaded.raw"))

        #for i in range(format.config.num_ms_psds):
        #    ms_psd_avg_bin_msqs[i].pow(0.5).float().cpu().numpy().tofile(os.path.join(output_path, f"mdct_psd_scale_{i}_loaded.raw"))
        #    (-ms_psd_avg_bin_means[i].float()).cpu().numpy().tofile(os.path.join(output_path, f"mdct_psd_offset_{i}_loaded.raw"))
            
        return
    
    for i, filename in enumerate(test_samples):
        
        print(f"pass 2 - file {i+1}/{len(test_samples)}: {filename}")

        file_path = os.path.join(dataset_path, filename)
        if os.path.isfile(file_path) == False:
            file_path = os.path.join(config.DEBUG_PATH, filename)

        raw_length = min(get_audio_info(file_path).frames, cfg.format_config.default_raw_length)
        crop_width = format.get_raw_crop_width(raw_length)

        start = -1 if cfg.save_output == False else 0
        raw_sample = load_audio(file_path, start=start, count=crop_width).unsqueeze(0).to(cfg.device)
        
        mdct_phase_psd = format.raw_to_mdct_phase_psd(raw_sample)
        mdct_phase, mdct_psd = mdct_phase_psd.to(dtype=torch.float64).chunk(2, dim=1)
        mdct_psd_avg_bin_msq += mdct_psd.pow(2).mean(dim=r_dims, keepdim=True) / len(test_samples)

        ms_psds = format.raw_to_ms_psd(raw_sample, level=-1)
        for i in range(format.config.num_ms_psds):
            _ms_psd_msq = ms_psds[i].to(dtype=torch.float64).pow(2).mean(dim=r_dims, keepdim=True)
            ms_psd_avg_bin_msqs[i] += _ms_psd_msq / len(test_samples)

    format.mdct_psd_scale.copy_(mdct_psd_avg_bin_msq.pow(0.5).float())

    for i in range(format.config.num_ms_psds):
        getattr(format, f"ms_psd_scale_{i}").copy_(ms_psd_avg_bin_msqs[i].pow(0.5).float())
    
    if format_loaded == False:
        if input("\nSave format? (y/n): ") == "y":
            format_output_path = os.path.join(output_path, "format")
            format.save_pretrained(output_path, subfolder="format")
            print(f"Saved format to {format_output_path}")


if __name__ == "__main__":

    init_cuda()
    ms_mdct_dual_format_test()