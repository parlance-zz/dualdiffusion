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

    audio_len = 65536*17
    quantize_level = 0 #16
    noise_level = 0 #0.08
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

    spectrogram_params = SpectrogramParams(sample_rate=dataset_sample_rate,
                                           stereo=dataset_num_channels == 2)
    spectrogram_converter = SpectrogramConverter(spectrogram_params, device=device)
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

    save_raw(spectrogram_converter.mel_scaler.fb.permute(1, 0), "./debug/mel_scaler_filters.raw")
    coverage = spectrogram_converter.mel_scaler.fb.mean(dim=1); coverage /= coverage.amax()
    save_raw(coverage, "./debug/mel_scaler_filter_coverage.raw")