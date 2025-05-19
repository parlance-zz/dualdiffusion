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

from modules.formats.ms_mdct_dual import MS_MDCT_DualFormat, MS_MDCT_DualFormatConfig
from training.trainer import TrainLogger as StatLogger
from utils.dual_diffusion_utils import (
    init_cuda, save_audio, load_audio,
    save_img, get_audio_info, dict_str, tensor_info_str
)


@dataclass
class MS_MDCT_DualFormat_TestConfig:

    device: str
    save_output: bool
    test_sample_verbose: bool
    downsample_mdct_psd: bool
    add_random_test_samples: int
    test_samples: list[str]

    format_config: MS_MDCT_DualFormatConfig

@torch.inference_mode()
def ms_mdct_dual_format_test() -> None:

    torch.manual_seed(0)

    cfg: MS_MDCT_DualFormat_TestConfig = config.load_config(MS_MDCT_DualFormat_TestConfig,
        os.path.join(config.CONFIG_PATH, "tests", "ms_mdct_dual_format.json"))
    format: MS_MDCT_DualFormat = MS_MDCT_DualFormat(cfg.format_config).to(cfg.device)

    print("Format config:")
    print(dict_str(format.config.__dict__))

    dataset_path = config.DATASET_PATH
    test_samples = cfg.test_samples

    if cfg.add_random_test_samples > 0:
        train_samples = config.load_json(os.path.join(config.DATASET_PATH, "train.jsonl"))
        test_samples += [sample["file_name"] for sample in random.sample(train_samples, cfg.add_random_test_samples)]

    output_path = os.path.join(config.DEBUG_PATH, "ms_mdct_dual_format_test")
    os.makedirs(output_path, exist_ok=True)
    
    # save spectrogram stft windows for both low and high bands
    wkwargs={
        "exponent": cfg.format_config.ms_window_exponent_low,
        "periodic": cfg.format_config.ms_window_periodic,
    }
    window_low = MS_MDCT_DualFormat._hann_power_window(cfg.format_config.ms_win_length, **wkwargs)
    window_low.numpy().tofile(os.path.join(output_path, "window_low.raw"))

    wkwargs={
        "exponent": cfg.format_config.ms_window_exponent_high,
        "periodic": cfg.format_config.ms_window_periodic,
    }
    if cfg.format_config.ms_window_exponent_high is not None:
        window_high = MS_MDCT_DualFormat._hann_power_window(cfg.format_config.ms_win_length, **wkwargs)
        window_high.numpy().tofile(os.path.join(output_path, "window_high.raw"))

    if format.ms_freq_scale_mdct_psd is not None:
        mel_spec_filters = format.ms_freq_scale_mdct_psd.filters
    else:
        mel_spec_filters = format.ms_freq_scale.filters
    mel_spec_filters.T.cpu().numpy().tofile(os.path.join(output_path, "mel_spec_filters.raw"))

    stat_logger = StatLogger()
    print(f"\nNum test_samples: {len(test_samples)}\n")

    for filename in test_samples:   
        
        print(f"file: {filename}")

        file_path = os.path.join(dataset_path, filename)
        if os.path.isfile(file_path) == False:
            file_path = os.path.join(config.DEBUG_PATH, filename)

        raw_length = min(get_audio_info(file_path).frames, cfg.format_config.default_raw_length)
        crop_width = format.get_raw_crop_width(raw_length)

        raw_sample = load_audio(file_path, count=crop_width).unsqueeze(0).to(cfg.device)
        mel_spec = format.raw_to_mel_spec(raw_sample)
        mel_spec_mdct_psd = format.mel_spec_to_mdct_psd(mel_spec)

        if cfg.downsample_mdct_psd == True:
            mel_spec_mdct_psd = torch.nn.functional.interpolate(mel_spec_mdct_psd,
                size=(cfg.format_config.mdct_num_frequencies, mel_spec_mdct_psd.shape[-1]), mode="area")

        mdct = format.raw_to_mdct(raw_sample)
        mdct_psd = format.raw_to_mdct_psd(raw_sample)
        raw_sample_mdct = format.mdct_to_raw(mdct)

        stat_logger.add_logs({
            "raw_sample_std": raw_sample.std(),
            "raw_sample_mdct_std": raw_sample_mdct.std(),
            "mel_spec_std": mel_spec.std(),
            "mel_spec_mdct_psd_std": mel_spec_mdct_psd.std(),
            "mdct_std": mdct.std(),
            "mdct_psd_std": mdct_psd.std(),
        })

        if cfg.test_sample_verbose == True:
            print("raw_sample:", tensor_info_str(raw_sample))
            print("mel_spec:", tensor_info_str(mel_spec), f"(target shape: {format.get_mel_spec_shape(raw_length=raw_length)}")
            print("mdct:", tensor_info_str(mdct), f"(target shape: {format.get_mdct_shape(raw_length=raw_length)}")
            print("mel_spec_mdct_psd:", tensor_info_str(mel_spec_mdct_psd))
            print("mdct_psd:", tensor_info_str(mdct_psd))
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
        
        mel_spec_mdct_psd_output_path = os.path.join(output_path, f"{filename}_mel_spec_mdct_psd.png")
        save_img(format.mdct_psd_to_img(mel_spec_mdct_psd), mel_spec_mdct_psd_output_path)
        print(f"Saved mel_spec_mdct_psd img to {mel_spec_mdct_psd_output_path}")

        mdct_output_path = os.path.join(output_path, f"{filename}_mdct.flac")
        save_audio(raw_sample_mdct.squeeze(0), cfg.format_config.sample_rate, mdct_output_path, target_lufs=None)
        print(f"Saved raw_sample_mdct to {mdct_output_path}")

        mdct_psd_output_path = os.path.join(output_path, f"{filename}_mdct_psd.png")
        save_img(format.mdct_psd_to_img(mdct_psd), mdct_psd_output_path)
        print(f"Saved mdct_psd img to {mdct_psd_output_path}")

    print("\nAverage stats:")
    print(dict_str(stat_logger.get_logs()))

if __name__ == "__main__":

    init_cuda()
    ms_mdct_dual_format_test()