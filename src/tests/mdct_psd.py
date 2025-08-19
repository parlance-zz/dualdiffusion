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

from modules.formats.mdct_psd import MDCT_PSD_Format, MDCT_PSD_FormatConfig
from training.trainer import TrainLogger as StatLogger
from utils.dual_diffusion_utils import (
    init_cuda, save_audio, load_audio,
    save_img, get_audio_info, dict_str, tensor_info_str
)


@dataclass
class MDCT_PSD_Format_TestConfig:

    device: str
    save_output: bool
    p2m_img_transposed: bool
    test_sample_verbose: bool
    add_random_test_samples: int
    test_samples: list[str]

    format_config: MDCT_PSD_FormatConfig

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
def mdct_psd_format_test() -> None:

    torch.manual_seed(0)
    random.seed()

    cfg: MDCT_PSD_Format_TestConfig = config.load_config(MDCT_PSD_Format_TestConfig,
        os.path.join(config.CONFIG_PATH, "tests", "mdct_psd_format.json"))
    format: MDCT_PSD_Format = MDCT_PSD_Format(cfg.format_config).to(cfg.device)

    print("Format config:")
    print(dict_str(format.config.__dict__))

    dataset_path = config.DATASET_PATH
    test_samples = cfg.test_samples

    if cfg.add_random_test_samples > 0:
        train_samples = config.load_json(os.path.join(config.DATASET_PATH, "train.jsonl"))
        test_samples += [sample["file_name"] for sample in random.sample(train_samples, cfg.add_random_test_samples)]

    output_path = os.path.join(config.DEBUG_PATH, "mdct_psd_format_test")
    os.makedirs(output_path, exist_ok=True)

    mdct_mel_density_scaling = 1 / format.mdct_mel_density.flatten().cpu()
    print("\nMDCT mel-density scaling coefficients:")
    print(mdct_mel_density_scaling)
    mdct_mel_density_scaling.numpy().tofile(os.path.join(output_path, "mdct_mel_density_scaling.raw"))
    mdct_avg_bin_std = torch.zeros_like(format.mdct_mel_density.flatten())

    p2m_avg_bin_std = torch.zeros(format.config.p2m_num_channels, dtype=torch.float32, device=format.device)
    p2m_avg_bin_mean = torch.zeros(format.config.p2m_num_channels, dtype=torch.float32, device=format.device)

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
        raw_sample_mdct = format.mdct_to_raw(mdct)
        mdct_psd = format.raw_to_mdct_psd(raw_sample)
        mdct_scaled = format.scale_mdct_from_psd(mdct, mdct_psd)
        mdct_unscaled = format.unscale_mdct_from_psd(mdct_scaled, mdct_psd)
        
        p2m = format.mdct_to_p2m(mdct)
        p2m_psd = format.mdct_to_p2m_psd(mdct)
        p2m_scaled = format.scale_p2m_from_psd(p2m, p2m_psd)
        p2m_unscaled = format.unscale_p2m_from_psd(p2m_scaled, p2m_psd)
        mdct_p2m = format.p2m_to_mdct(p2m)
        raw_sample_mdct_p2m = format.mdct_to_raw(mdct_p2m)

        mdct_avg_bin_std += mdct.std(dim=(0, 1, 3)) / len(test_samples)

        p2m_avg_bin_std += p2m.std(dim=(0, 2, 3)) / len(test_samples)
        p2m_avg_bin_mean += p2m.mean(dim=(0, 2, 3)) / len(test_samples)

        stat_logger.add_logs({
            "raw_sample_std": raw_sample.std(),
            "raw_sample_mdct_std": raw_sample_mdct.std(),
            "raw_sample_mdct_p2m_std": raw_sample_mdct_p2m.std(),
            "mdct_std": mdct.std(),
            "mdct_scaled_std": mdct_scaled.std(),
            "mdct_unscaled_std": mdct_unscaled.std(),
            "mdct_p2m_std": mdct_p2m.std(),
            "p2m_mean": p2m.mean(),
            "p2m_std": p2m.std(),
            "p2m_scaled_mean": p2m_scaled.mean(),
            "p2m_scaled_std": p2m_scaled.std(),
            "p2m_unscaled_std": p2m_unscaled.std(),
            "p2m_psd_mean": p2m_psd.mean(),
            "p2m_psd_std": p2m_psd.std(),
        })

        if cfg.test_sample_verbose == True:
            print("raw_sample:", tensor_info_str(raw_sample))
            print("raw_sample_mdct:", tensor_info_str(raw_sample_mdct))
            print("mdct:", tensor_info_str(mdct), f"(target shape: {format.get_mdct_shape(raw_length=raw_length)}")
            print("mdct_p2m:", tensor_info_str(mdct_p2m))
            print("p2m:", tensor_info_str(p2m), f"(target shape: {format.get_p2m_shape(mdct.shape)}")
            print("p2m_psd:", tensor_info_str(p2m_psd), "\n")

        if cfg.save_output == False:
            continue

        filename = os.path.splitext(os.path.basename(filename))[0]

        raw_sample_output_path = os.path.join(output_path, f"{filename}.flac")
        save_audio(raw_sample.squeeze(0), cfg.format_config.sample_rate, raw_sample_output_path, target_lufs=None)
        print(f"Saved raw_sample to {raw_sample_output_path}")

        mdct_output_path = os.path.join(output_path, f"{filename}_mdct.flac")
        save_audio(raw_sample_mdct.squeeze(0), cfg.format_config.sample_rate, mdct_output_path, target_lufs=None)
        print(f"Saved raw_sample_mdct to {mdct_output_path}")

        raw_sample_mdct_p2m_output_path = os.path.join(output_path, f"{filename}_mdct_p2m.flac")
        save_audio(raw_sample_mdct_p2m.squeeze(0), cfg.format_config.sample_rate, raw_sample_mdct_p2m_output_path, target_lufs=None)
        print(f"Saved raw_sample_mdct_p2m to {raw_sample_mdct_p2m_output_path}")

        p2m_output_path = os.path.join(output_path, f"{filename}_p2m.png")
        save_img(format._p2m_to_img(p2m, transposed=cfg.p2m_img_transposed), p2m_output_path)
        print(f"Saved p2m img to {p2m_output_path}")
        
        p2m_psd_output_path = os.path.join(output_path, f"{filename}_p2m_psd.png")
        save_img(format.psd_to_img(p2m_psd, transpose_p2m=cfg.p2m_img_transposed), p2m_psd_output_path)
        print(f"Saved p2m_psd img to {p2m_psd_output_path}\n")

        p2m_scaled_output_path = os.path.join(output_path, f"{filename}_p2m_scaled.png")
        save_img(format._p2m_to_img(p2m_scaled, transposed=cfg.p2m_img_transposed), p2m_scaled_output_path)
        print(f"Saved mdct_psd img to {p2m_scaled_output_path}")

    print(f"\nAverage MDCT bin std (std variance: {mdct_avg_bin_std.var().item()}):")
    print(mdct_avg_bin_std)
    mdct_avg_bin_std.cpu().numpy().tofile(os.path.join(output_path, "mdct_avg_bin_std.raw"))

    print(f"\nAverage P2M bin std:")
    print_tensor_as_json_table(p2m_avg_bin_std, 8)
    print(f"\nAverage P2M bin mean:")
    print_tensor_as_json_table(p2m_avg_bin_mean, 8)

    print("\nAverage stats:")
    print(dict_str(stat_logger.get_logs()))


if __name__ == "__main__":

    init_cuda()
    mdct_psd_format_test()