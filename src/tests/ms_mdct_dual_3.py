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

from modules.formats.ms_mdct_dual_3 import MS_MDCT_DualFormat, MS_MDCT_DualFormatConfig
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
        os.path.join(config.CONFIG_PATH, "tests", "ms_mdct_dual_format_3.json"))
    format: MS_MDCT_DualFormat = MS_MDCT_DualFormat(cfg.format_config).to(cfg.device)

    print("Format config:")
    print(dict_str(format.config.__dict__))

    dataset_path = config.DATASET_PATH
    test_samples = cfg.test_samples

    if cfg.add_random_test_samples > 0:
        train_samples = config.load_json(os.path.join(config.DATASET_PATH, "train.jsonl"))
        test_samples += [sample["file_name"] for sample in random.sample(train_samples, cfg.add_random_test_samples)]

    output_path = os.path.join(config.DEBUG_PATH, "ms_mdct_dual_format_3_test")
    os.makedirs(output_path, exist_ok=True)

    format.ms_filters.transpose(0, 1).cpu().numpy().tofile(os.path.join(output_path, "ms_filters.raw"))
    format.ms_windows.transpose(0, 1).cpu().numpy().tofile(os.path.join(output_path, "ms_windows.raw"))
    mdct_phase_avg_bin_var = torch.zeros_like(format.mdct_mel_density.flatten())

    stat_logger = StatLogger()
    print(f"\nNum test_samples: {len(test_samples)}\n")

    for i, filename in enumerate(test_samples):
        
        print(f"file {i+1}/{len(test_samples)}: {filename}")

        file_path = os.path.join(dataset_path, filename)
        if os.path.isfile(file_path) == False:
            file_path = os.path.join(config.DEBUG_PATH, filename)

        raw_length = min(get_audio_info(file_path).frames, cfg.format_config.default_raw_length)
        crop_width = format.get_raw_crop_width(raw_length)

        start = -1 if cfg.save_output == False else 0
        raw_sample = load_audio(file_path, start=start, count=crop_width).unsqueeze(0).to(cfg.device)
        mel_spec = format.raw_to_mel_spec(raw_sample)
        linear_psd = format.mel_spec_to_linear_psd(mel_spec)

        mdct = format.raw_to_mdct(raw_sample)
        mdct_phase = format.raw_to_mdct_phase(raw_sample)
        #raw_sample_mdct = format.mdct_to_raw(mdct)
        raw_sample_mdct = format.mdct_phase_to_raw(mdct_phase)

        mdct_phase_avg_bin_var += mdct_phase.var(dim=(0,1,3)) / len(test_samples)

        stat_logger.add_logs({
            "raw_sample_var": raw_sample.var(),
            "raw_sample_mdct_var": raw_sample_mdct.var(),
            "mel_spec_var": mel_spec.var(),
            "mel_spec_mean": mel_spec.mean(),
            "mdct_var": mdct.var(),
            "mdct_phase_var": mdct_phase.var(),
            "linear_psd_mean": linear_psd.mean(),
            "linear_psd_var": linear_psd.var()
        })

        if cfg.test_sample_verbose == True:
            print("raw_sample:", tensor_info_str(raw_sample))
            print("mel_spec:", tensor_info_str(mel_spec), f"(target shape: {format.get_mel_spec_shape(raw_length=raw_length)}")
            print("mdct_phase:", tensor_info_str(mdct_phase), f"(target shape: {format.get_mdct_shape(raw_length=raw_length)}")
            print("linear_psd:", tensor_info_str(linear_psd))
            print("raw_sample_mdct:", tensor_info_str(raw_sample_mdct), "\n")

        if cfg.save_output == False:
            continue

        filename = os.path.splitext(os.path.basename(filename))[0]

        raw_sample_output_path = os.path.join(output_path, f"{filename}.flac")
        save_audio(raw_sample.squeeze(0), cfg.format_config.sample_rate, raw_sample_output_path, target_lufs=None)
        print(f"Saved raw_sample to {raw_sample_output_path}")

        mel_spec_output_path = os.path.join(output_path, f"{filename}_mel_spec.png")
        save_img(format.mel_spec_to_img(mel_spec), mel_spec_output_path)
        print(f"Saved mel_spec img to {mel_spec_output_path}")

        mdct_output_path = os.path.join(output_path, f"{filename}_mdct.flac")
        save_audio(raw_sample_mdct.squeeze(0), cfg.format_config.sample_rate, mdct_output_path, target_lufs=None)
        print(f"Saved raw_sample_mdct to {mdct_output_path}")

        linear_psd_path = os.path.join(output_path, f"{filename}_linear_psd.png")
        save_img(format.mel_spec_to_img(linear_psd), linear_psd_path)
        print(f"Saved linear_psd img to {linear_psd_path}")

        mdct_psd = format.raw_to_mdct_psd(raw_sample)
        mdct_psd_path = os.path.join(output_path, f"{filename}_mdct_psd.png")
        save_img(tensor_to_img(mdct_psd, flip_y=True), mdct_psd_path)
        print(f"Saved mdct_psd img to {mdct_psd_path}")

        # make sure both of these images are scaled the same for visual comparison
        linear_psd_downscaled = torch.nn.functional.interpolate(linear_psd, mdct_psd.shape[2:], mode="area")

        mdct_psd /= mdct_psd.amax()
        linear_psd_downscaled /= linear_psd_downscaled.amax()
        linear_psd_downscaled.clip_(mdct_psd.amin(), mdct_psd.amax())
        linear_psd_downscaled[0, 0, 0, 0] = mdct_psd.amin(); linear_psd_downscaled[0, 0, 0, 1] = mdct_psd.amax()
    
        linear_psd_downscaled_path = os.path.join(output_path, f"{filename}_mdct_psd_linear_downscaled.png")
        save_img(format.mel_spec_to_img(linear_psd_downscaled), linear_psd_downscaled_path)
        print(f"Saved linear_psd_downscaled img to {linear_psd_downscaled_path}")

    print(f"\nAverage MDCT phase bin scales:")
    print(mdct_phase_avg_bin_var.pow(0.5).cpu().tolist())
    mdct_phase_avg_bin_var.cpu().numpy().tofile(os.path.join(output_path, "mdct_phase_avg_bin_var.raw"))

    print("\nAverage stats:")
    print(dict_str(stat_logger.get_logs()))


if __name__ == "__main__":

    init_cuda()
    ms_mdct_dual_format_test()