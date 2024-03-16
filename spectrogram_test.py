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

import os
import timeit
from dotenv import load_dotenv

import torch
import numpy as np

from dual_diffusion_utils import load_audio, save_raw_img, save_audio, save_raw, init_cuda, load_raw, quantize_tensor
from spectrogram import SpectrogramParams, SpectrogramConverter

if __name__ == "__main__":

    init_cuda()
    load_dotenv(override=True)

    audio_len = 32000 * 60
    quantize_level = 0 #16
    noise_level = 0 #0.08
    use_mel_scale = True#False
    device = "cuda"

    dataset_path = os.environ.get("DATASET_PATH", "./dataset/samples")
    dataset_format = os.environ.get("DATASET_FORMAT", ".flac")
    dataset_raw_format = os.environ.get("DATASET_RAW_FORMAT", "int16")
    dataset_sample_rate = int(os.environ.get("DATASET_SAMPLE_RATE", 32000))
    dataset_num_channels = int(os.environ.get("DATASET_NUM_CHANNELS", 2))
    sample_filename = np.random.choice(os.listdir(dataset_path), 1, replace=False)[0]
    
    #sample_filename = "Mario no Super Picross - 109 Mario Puzzle 3.flac"
    #sample_filename = "Mega Man X3 - 09 Blast Hornet.flac"
    #sample_filename = "Vortex - 10 Magmemo.flac"
    #sample_filename = "Ganbare Goemon 4 - Kirakira Douchuu - Boku ga Dancer ni Natta Riyuu - 61 Planet Impact Dam.flac"
    #sample_filename = "Bahamut Lagoon - 09 Materaito.flac"
    #sample_filename = "Kirby Super Star  [Kirby's Fun Pak] - 36 Mine Cart Riding.flac"
    #sample_filename = "Pilotwings - 04 Light Plane.flac"
    #sample_filename = "Terranigma - 36 Call at a Port.flac"
    #sample_filename = "Great Battle Gaiden 2, The - Matsuri da Wasshoi - 31 Epilogue to the Story (part 1).flac"
    sample_filename = "Kirby Super Star  [Kirby's Fun Pak] - 41 Halberd ~ Nightmare Warship.flac"

    spectrogram_params = SpectrogramParams(sample_rate=dataset_sample_rate,
                                           stereo=dataset_num_channels == 2,
                                           use_mel_scale=use_mel_scale)
    spectrogram_converter = SpectrogramConverter(spectrogram_params).to(device)
    crop_width = spectrogram_converter.get_crop_width(audio_len)

    print("Sample filename: ", sample_filename)
    file_ext = os.path.splitext(sample_filename)[1]
    if dataset_format == ".raw":
        audio = load_raw(os.path.join(dataset_path, sample_filename),
                         dtype=dataset_raw_format, start=0, count=crop_width, device=device)
    else:
        audio = load_audio(os.path.join(dataset_path, sample_filename),
                           start=0, count=crop_width, device=device)
    save_audio(audio, dataset_sample_rate, "./debug/original_audio.flac")
    audio = audio.unsqueeze(0)

    start = timeit.default_timer()
    spectrogram = spectrogram_converter.audio_to_spectrogram(audio)
    print("Encode time: ", timeit.default_timer() - start)
    
    if noise_level > 0:
        spectrogram += torch.randn_like(spectrogram) * noise_level * spectrogram.std()
    if quantize_level > 0:
        spectrogram = quantize_tensor(spectrogram, quantize_level)

    print("Audio shape:", audio.shape, "Spectrogram shape:", spectrogram.shape)
    save_raw_img(spectrogram, "./debug/spectrogram.png")
    
    torch.manual_seed(0)
    start = timeit.default_timer()
    audio = spectrogram_converter.spectrogram_to_audio(spectrogram)
    print("Decode time: ", timeit.default_timer() - start)
    save_audio(audio.squeeze(0), dataset_sample_rate, "./debug/reconstructed_audio.flac")

    win_length = spectrogram_params.win_length
    hop_length = spectrogram_params.hop_length
    print(f"win_length: {win_length}, hop_length: {hop_length}")
    save_raw(spectrogram_converter.spectrogram_func.window, "./debug/window.raw")

    if use_mel_scale:
        save_raw(spectrogram_converter.mel_scaler.fb.permute(1, 0), "./debug/mel_scaler_filters.raw")
        coverage = spectrogram_converter.mel_scaler.fb.mean(dim=1); coverage /= coverage.amax()
        save_raw(coverage, "./debug/mel_scaler_filter_coverage.raw")