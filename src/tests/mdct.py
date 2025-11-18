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

from modules.formats.mdct import MDCT_Format, MDCT_FormatConfig
from training.trainer import TrainLogger as StatLogger
from utils.dual_diffusion_utils import (
    init_cuda, save_audio, load_audio,
    save_img, get_audio_info, dict_str, tensor_info_str
)


@dataclass
class MDCT_Format_TestConfig:

    device: str
    save_output: bool
    test_sample_verbose: bool
    add_random_test_samples: int
    test_samples: list[str]

    format_config: MDCT_FormatConfig

def print_tensor_as_json_table(tensor: torch.Tensor, values_per_row: int) -> None:

    data = tensor.flatten().cpu().numpy()
    print("[")
    for i in range(0, len(data), values_per_row):
        row = data[i:i+values_per_row]

        formatted_row = ",".join(f"{val:6.3f}" for val in row)
        if i + values_per_row >= len(data):
            print("  " + formatted_row)
        else:
            print("  " + formatted_row + ",")
    print("]")
    
@torch.inference_mode()
def mdct_format_test() -> None:

    torch.manual_seed(0)
    random.seed()

    cfg: MDCT_Format_TestConfig = config.load_config(MDCT_Format_TestConfig,
        os.path.join(config.CONFIG_PATH, "tests", "mdct_format.json"))
    format: MDCT_Format = MDCT_Format(cfg.format_config).to(cfg.device)

    print("Format config:")
    print(dict_str(format.config.__dict__))

    dataset_path = config.DATASET_PATH
    test_samples = cfg.test_samples

    if cfg.add_random_test_samples > 0:
        train_samples = config.load_json(os.path.join(config.DATASET_PATH, "train.jsonl"))
        test_samples += [sample["file_name"] for sample in random.sample(train_samples, cfg.add_random_test_samples)]

    output_path = os.path.join(config.DEBUG_PATH, "mdct_format_test")
    os.makedirs(output_path, exist_ok=True)

    mdct_mel_density_scaling = 1 / format.mdct_mel_density.flatten().cpu()
    print("\nMDCT mel-density scaling coefficients:")
    print(mdct_mel_density_scaling)
    mdct_mel_density_scaling.numpy().tofile(os.path.join(output_path, "mdct_mel_density_scaling.raw"))
    mdct_avg_bin_std = torch.zeros_like(format.mdct_mel_density.flatten())

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

        mdct = format.raw_to_mdct(raw_sample)
        mdct_psd = format.raw_to_mdct_psd(raw_sample)
        raw_sample_mdct = format.mdct_to_raw(mdct)

        mdct_avg_bin_std += mdct.std(dim=(0, 2, 3)) / len(test_samples)

        stat_logger.add_logs({
            "raw_sample_std": raw_sample.std(),
            "raw_sample_mdct_std": raw_sample_mdct.std(),
            "mdct_std": mdct.std(),
            "mdct_psd": mdct_psd,
        })

        if cfg.test_sample_verbose == True:
            print("raw_sample:", tensor_info_str(raw_sample))
            print("raw_sample_mdct:", tensor_info_str(raw_sample_mdct))
            print("mdct:", tensor_info_str(mdct), f"(target shape: {format.get_mdct_shape(raw_length=raw_length)}")

        if cfg.save_output == False:
            continue

        filename = os.path.splitext(os.path.basename(filename))[0]

        raw_sample_output_path = os.path.join(output_path, f"{filename}.flac")
        save_audio(raw_sample.squeeze(0), cfg.format_config.sample_rate, raw_sample_output_path, target_lufs=None)
        print(f"Saved raw_sample to {raw_sample_output_path}")

        mdct_output_path = os.path.join(output_path, f"{filename}_mdct.flac")
        save_audio(raw_sample_mdct.squeeze(0), cfg.format_config.sample_rate, mdct_output_path, target_lufs=None)
        print(f"Saved raw_sample_mdct to {mdct_output_path}")

    print(f"\nAverage MDCT bin std (std variance: {mdct_avg_bin_std.var().item()}):")
    print(mdct_avg_bin_std)
    mdct_avg_bin_std.cpu().numpy().tofile(os.path.join(output_path, "mdct_avg_bin_std.raw"))

    print("\nAverage stats:")
    print(dict_str(stat_logger.get_logs()))

if __name__ == "__main__":

    init_cuda()
    mdct_format_test()