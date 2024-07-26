import utils.config as config

import os
import timeit

import torch

from formats.spectrogram import SpectrogramParams, SpectrogramConverter
from utils.dual_diffusion_utils import (
    load_audio, save_raw_img, save_audio, save_raw, init_cuda, quantize_tensor
)

if __name__ == "__main__":

    init_cuda()

    audio_len = 32000 * 45
    quantize_level = 0 #32
    noise_level = 0 #0.08
    use_mel_scale = True#False
    device = "cuda"

    dataset_path = config.DATASET_PATH
    dataset_cfg = config.load_json(os.path.join(dataset_path, "dataset.json"))

    dataset_format = dataset_cfg["format"]
    dataset_sample_rate = dataset_cfg["sample_rate"]
    dataset_num_channels = dataset_cfg["num_channels"]
    
    #sample_filename = "2/Mario no Super Picross - 109 Mario Puzzle 3.flac"
    #sample_filename = "2/Mega Man X3 - 09 Blast Hornet.flac"
    #sample_filename = "3/Vortex - 10 Magmemo.flac"
    #sample_filename = "1/Ganbare Goemon 4 - Kirakira Douchuu - Boku ga Dancer ni Natta Riyuu - 61 Planet Impact Dam.flac"
    #sample_filename = "1/Bahamut Lagoon - 09 Materaito.flac"
    #sample_filename = "1/Kirby Super Star  [Kirby's Fun Pak] - 36 Mine Cart Riding.flac"
    #sample_filename = "2/Pilotwings - 04 Light Plane.flac"
    #sample_filename = "2/Terranigma - 36 Call at a Port.flac"
    #sample_filename = "1/Great Battle Gaiden 2, The - Matsuri da Wasshoi - 31 Epilogue to the Story (part 1).flac"
    sample_filename = "1/Kirby Super Star  [Kirby's Fun Pak] - 41 Halberd ~ Nightmare Warship.flac"

    spectrogram_params = SpectrogramParams(sample_rate=dataset_sample_rate,
                                           stereo=dataset_num_channels == 2,
                                           use_mel_scale=use_mel_scale)
    spectrogram_converter = SpectrogramConverter(spectrogram_params).to(device)
    crop_width = spectrogram_converter.get_crop_width(audio_len)

    print("Sample filename: ", sample_filename)
    file_ext = os.path.splitext(sample_filename)[1]
    audio = load_audio(os.path.join(dataset_path, sample_filename),
                        start=0, count=crop_width, device=device)
    if config.DEBUG_PATH is not None:
        save_audio(audio, dataset_sample_rate, os.path.join(config.DEBUG_PATH, "original_audio.flac"))
    audio = audio.unsqueeze(0)

    start = timeit.default_timer()
    spectrogram = spectrogram_converter.audio_to_spectrogram(audio)
    print("Encode time: ", timeit.default_timer() - start)
    
    if noise_level > 0:
        spectrogram += torch.randn_like(spectrogram) * noise_level * spectrogram.std()
    if quantize_level > 0:
        spectrogram = quantize_tensor(spectrogram, quantize_level)

    print("Audio shape:", audio.shape, "Spectrogram shape:", spectrogram.shape)
    if config.DEBUG_PATH is not None:
        save_raw_img(spectrogram, os.path.join(config.DEBUG_PATH, "spectrogram.png"))
    
    torch.manual_seed(0)
    start = timeit.default_timer()
    audio = spectrogram_converter.spectrogram_to_audio(spectrogram)
    print("Decode time: ", timeit.default_timer() - start)
    if config.DEBUG_PATH is not None:
        save_audio(audio.squeeze(0), dataset_sample_rate, os.path.join(config.DEBUG_PATH, "reconstructed_audio.flac"))

    win_length = spectrogram_params.win_length
    hop_length = spectrogram_params.hop_length
    print(f"win_length: {win_length}, hop_length: {hop_length}")
    if config.DEBUG_PATH is not None:
        save_raw(spectrogram_converter.spectrogram_func.window, os.path.join(config.DEBUG_PATH, "window.raw"))

    if use_mel_scale:
        coverage = spectrogram_converter.mel_scaler.fb.mean(dim=1); coverage /= coverage.amax()
        if config.DEBUG_PATH is not None:
            save_raw(spectrogram_converter.mel_scaler.fb.permute(1, 0), os.path.join(config.DEBUG_PATH, "mel_scaler_filters.raw"))
            save_raw(coverage, os.path.join(config.DEBUG_PATH, "mel_scaler_filter_coverage.raw"))