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

from modules.formats.mdct_2psd import MDCT_2PSD_Format, MDCT_2PSD_FormatConfig
from training.trainer import TrainLogger as StatLogger
from utils.dual_diffusion_utils import (
    init_cuda, save_audio, load_audio,
    save_img, get_audio_info, dict_str, tensor_info_str
)


@dataclass
class MDCT_2PSD_Format_TestConfig:

    device: str
    save_output: bool
    test_sample_verbose: bool
    add_random_test_samples: int
    test_samples: list[str]

    format_config: MDCT_2PSD_FormatConfig
    
@torch.inference_mode()
def mdct_2psd_format_test() -> None:

    torch.manual_seed(0)
    random.seed()

    cfg: MDCT_2PSD_Format_TestConfig = config.load_config(MDCT_2PSD_Format_TestConfig,
        os.path.join(config.CONFIG_PATH, "tests", "mdct_2psd_format.json"))
    format: MDCT_2PSD_Format = MDCT_2PSD_Format(cfg.format_config).to(cfg.device)

    print("Format config:")
    print(dict_str(format.config.__dict__))

    dataset_path = config.DATASET_PATH
    test_samples = cfg.test_samples

    if cfg.add_random_test_samples > 0:
        train_samples = config.load_json(os.path.join(config.DATASET_PATH, "train.jsonl"))
        test_samples += [sample["file_name"] for sample in random.sample(train_samples, cfg.add_random_test_samples)]

    output_path = os.path.join(config.DEBUG_PATH, "mdct_2psd_format_test")
    os.makedirs(output_path, exist_ok=True)

    stat_logger = StatLogger()
    print(f"\nNum test_samples: {len(test_samples)}\n")

    for i, filename in enumerate(test_samples):
        
        print(f"file {i+1}/{len(test_samples)}: {filename}")

        file_path = os.path.join(dataset_path, filename)
        if os.path.isfile(file_path) == False:
            file_path = os.path.join(config.DEBUG_PATH, filename)

        raw_length = min(get_audio_info(file_path).frames, cfg.format_config.default_raw_length)
        crop_width = format.get_raw_crop_width(raw_length)

        raw_sample = load_audio(file_path, count=crop_width).unsqueeze(0).to(cfg.device)

        mdct0 = format.raw_to_mdct(raw_sample, idx=0)
        raw_sample_mdct0 = format.mdct_to_raw(mdct0, idx=0)
        mdct0_psd = format.raw_to_mdct_psd(raw_sample, idx=0)
        mdct0_scaled = format.scale_mdct_from_psd(mdct0, mdct0_psd, idx=0)
        mdct0_unscaled = format.unscale_mdct_from_psd(mdct0_scaled, mdct0_psd, idx=0)

        mdct1 = format.raw_to_mdct(raw_sample, idx=1)
        raw_sample_mdct1 = format.mdct_to_raw(mdct1, idx=1)
        mdct1_psd = format.raw_to_mdct_psd(raw_sample, idx=1)
        mdct1_scaled = format.scale_mdct_from_psd(mdct1, mdct1_psd, idx=1)
        mdct1_unscaled = format.unscale_mdct_from_psd(mdct1_scaled, mdct1_psd, idx=1)

        stat_logger.add_logs({
            "raw_sample_std": raw_sample.std(),
            "raw_sample_mdct0_std": raw_sample_mdct0.std(),
            "raw_sample_mdct1_std": raw_sample_mdct1.std(),
            "mdct0_std": mdct0.std(),
            "mdct0_psd_std": mdct0_psd.std(),
            "mdct0_psd_mean": mdct0_psd.mean(),
            "mdct0_scaled_std": mdct0_scaled.std(),
            "mdct0_unscaled_std": mdct0_unscaled.std(),
            "mdct1_std": mdct1.std(),
            "mdct1_psd_std": mdct1_psd.std(),
            "mdct1_psd_mean": mdct1_psd.mean(),
            "mdct1_scaled_std": mdct1_scaled.std(),
            "mdct1_unscaled_std": mdct1_unscaled.std(),
        })

        if cfg.test_sample_verbose == True:
            print("raw_sample:", tensor_info_str(raw_sample))
            print("raw_sample_mdct0:", tensor_info_str(raw_sample_mdct0))
            print("raw_sample_mdct1:", tensor_info_str(raw_sample_mdct1))
            print("mdct0:", tensor_info_str(mdct0), f"(target shape: {format.get_mdct_shape(raw_length=raw_length, idx=0)}")
            print("mdct1:", tensor_info_str(mdct1), f"(target shape: {format.get_mdct_shape(raw_length=raw_length, idx=1)}")

        if cfg.save_output == False:
            continue

        filename = os.path.splitext(os.path.basename(filename))[0]

        raw_sample_output_path = os.path.join(output_path, f"{filename}.flac")
        save_audio(raw_sample.squeeze(0), cfg.format_config.sample_rate, raw_sample_output_path, target_lufs=None)
        print(f"Saved raw_sample to {raw_sample_output_path}")

        mdct0_output_path = os.path.join(output_path, f"{filename}_mdct0.flac")
        save_audio(raw_sample_mdct0.squeeze(0), cfg.format_config.sample_rate, mdct0_output_path, target_lufs=None)
        print(f"Saved raw_sample_mdct0 to {mdct0_output_path}")

        mdct1_output_path = os.path.join(output_path, f"{filename}_mdct1.flac")
        save_audio(raw_sample_mdct1.squeeze(0), cfg.format_config.sample_rate, mdct1_output_path, target_lufs=None)
        print(f"Saved raw_sample_mdct1 to {mdct1_output_path}")

        mdct0_psd_output_path = os.path.join(output_path, f"{filename}_mdct0_psd.png")
        save_img(format.psd_to_img(mdct0_psd), mdct0_psd_output_path)
        print(f"Saved mdct0_psd to {mdct0_psd_output_path}")

        mdct1_psd_output_path = os.path.join(output_path, f"{filename}_mdct1_psd.png")
        save_img(format.psd_to_img(mdct1_psd), mdct1_psd_output_path)
        print(f"Saved mdct1_psd to {mdct1_psd_output_path}")

    print("\nAverage stats:")
    print(dict_str(stat_logger.get_logs()))


if __name__ == "__main__":

    init_cuda()
    mdct_2psd_format_test()