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

import os
import timeit

import torch

from formats.spectrogram import DualSpectrogramFormat
from utils.dual_diffusion_utils import (
    load_audio, tensor_to_img, save_img, save_audio,
    save_tensor_raw, init_cuda, quantize_tensor
)

@torch.inference_mode()
def spectrogram_test() -> None:

    test_params = config.load_json(os.path.join(config.CONFIG_PATH, "tests", "spectrogram_test.json"))
    format_params = config.load_json(os.path.join(config.CONFIG_PATH, "tests", test_params["format_cfg_file"]))

    spectrogram_format = DualSpectrogramFormat(**format_params).to(device=test_params["device"])
    crop_width = spectrogram_format.get_raw_crop_width(length=test_params["audio_len"])
    test_output_path = os.path.join(config.DEBUG_PATH, "spectrogram_test") if config.DEBUG_PATH is not None else None

    audios = []
    for sample_filename in test_params["sample_filenames"]:
        audio = load_audio(os.path.join(config.DATASET_PATH, sample_filename),
                            start=0, count=crop_width, device=test_params["device"])

        if test_output_path is not None:
            base_filename = os.path.splitext(os.path.basename(sample_filename))[0]
            save_audio(audio, spectrogram_format.config["sample_rate"],
                       os.path.join(test_output_path, f"{base_filename}_original.flac"))

        audios.append(audio)
    audio = torch.stack(audios, dim=0)

    start = timeit.default_timer()
    spectrogram = spectrogram_format.raw_to_sample(audio)
    print("Encode time: ", timeit.default_timer() - start)
    
    if test_params["noise_level"] > 0:
        spectrogram += torch.randn_like(spectrogram) * test_params["noise_level"]
    if test_params["quantize_level"] > 0:
        spectrogram = quantize_tensor(spectrogram, test_params["quantize_level"])

    print("Audio shape:", audio.shape, "Spectrogram shape:", spectrogram.shape)
    win_length = spectrogram_format.spectrogram_params.win_length
    hop_length = spectrogram_format.spectrogram_params.hop_length
    print(f"win_length: {win_length}, hop_length: {hop_length}")

    if test_output_path is not None:
        for sample_spectrogram, sample_filename in zip(spectrogram.unbind(0), test_params["sample_filenames"]):
            base_filename = os.path.splitext(os.path.basename(sample_filename))[0]
            spectrogram_img = tensor_to_img(sample_spectrogram, flip_y=True)
            save_img(spectrogram_img, os.path.join(test_output_path, f"{base_filename}_spectrogram.png"))
    
    start = timeit.default_timer()
    audio = spectrogram_format.sample_to_raw(spectrogram)
    print("Decode time: ", timeit.default_timer() - start)

    if test_output_path is not None:
        for audio, sample_filename in zip(audio.unbind(0), test_params["sample_filenames"]):
            base_filename = os.path.splitext(os.path.basename(sample_filename))[0]
            save_audio(audio, spectrogram_format.config["sample_rate"],
                       os.path.join(test_output_path, f"{base_filename}_reconstructed.flac"))

        save_tensor_raw(spectrogram_format.spectrogram_converter.spectrogram_func.window,
                        os.path.join(test_output_path, "phase_reconstruction_window.raw"))

        coverage = spectrogram_format.spectrogram_converter.freq_scale.filters.mean(dim=1); coverage /= coverage.amax()
        save_tensor_raw(coverage, os.path.join(test_output_path, "freqscaler_filter_coverage.raw"))
        save_tensor_raw(spectrogram_format.spectrogram_converter.freq_scale.filters.permute(1, 0),
                        os.path.join(test_output_path, "freqscaler_filters.raw"))

if __name__ == "__main__":

    init_cuda()
    torch.manual_seed(0)

    spectrogram_test()