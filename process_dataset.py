import os
import ffmpeg
import numpy as np
import torch
import json
import shutil

import torch.nn.functional as F

from dual_diffusion_pipeline import DualDiffusionPipeline

DATASET_DUAL_LOADER_PATH = "./dataset_dual_loader.py"
SOURCE_DIR = './dataset/source'
RAW_DIR = './dataset/raw'
DATA_CFG_PATH = './dataset/config.json'
NEW_MODEL_PATH = './models/new_dualdiffusion'
ENABLE_DEBUG_DUMPS = False
SAMPLE_RATE = 44100

SAMPLE_DIR = './dataset/dual'
SAMPLE_LEN = 2 ** 21
FADEOUT_LEN = 2 ** 17
FADEOUT_STD = 10.

S_RESOLUTION = 4096
F_RESOLUTION = 4096 // 2

def decode_source_to_raw(input_dir, output_dir, sample_rate):

    for dirpath, _, filenames in os.walk(input_dir):

        for filename in filenames:
            
            input_file = os.path.join(dirpath, filename)
            
            relative_dirpath = os.path.relpath(dirpath, input_dir)
            output_file_dir = os.path.join(output_dir, relative_dirpath)
            os.makedirs(output_file_dir, exist_ok=True)
            
            output_file = os.path.join(output_file_dir, filename)
            output_file = os.path.splitext(output_file)[0] + '.raw'
            
            try:
                ffmpeg.input(input_file).output(output_file, ac=1, ar=sample_rate, format='f32le').run()
            except Exception as e:
                print(f"Error processing '{input_file}': {e}")

def preprocess_raw_to_dual(input_dir,
                           output_dir,
                           config_path,
                           sample_len,
                           fadeout_len,
                           fadeout_std,
                           s_resolution,
                           f_resolution,
                           enable_debug_dumps=False
                           ):

    total_processed = 0

    s_avg_std = 0.
    f_avg_std = 0.

    fadeout_window = torch.exp(-torch.square(torch.linspace(0., 1., fadeout_len, device="cuda")) * fadeout_std)

    os.makedirs(output_dir, exist_ok=True)

    for dirpath, _, filenames in os.walk(input_dir):

        for filename in filenames:
            
            input_file = os.path.join(dirpath, filename)
            
            raw_input = np.fromfile(input_file, dtype=np.float32, count=sample_len)
            if len(raw_input) < sample_len: # skip samples that are too short
                print(f"Skipping '{input_file}': input length {len(raw_input)} < target dual length {sample_len}")
                continue
            
            raw_input = torch.from_numpy(raw_input).to("cuda")
            raw_input[-fadeout_len:] *= fadeout_window
            raw_input -= torch.mean(raw_input)           # remove DC offset
            raw_input /= torch.max(torch.abs(raw_input)) # normalize to [-1, 1]
            s_response = DualDiffusionPipeline.get_s_samples(raw_input, s_resolution)
            s_avg_std += torch.std(s_response).item()

            fft_input = torch.fft.fft(raw_input, norm="ortho")[:len(raw_input)//2]
            f_response = DualDiffusionPipeline.get_f_samples(fft_input, f_resolution)
            f_avg_std += torch.std(f_response).item()
            
            if (total_processed == 0) and enable_debug_dumps:
                s_response.cpu().numpy().tofile("./dataset/s_response.raw")
                f_response.cpu().numpy().tofile("./dataset/f_response.raw")

                s_reconstruction = torch.zeros_like(raw_input)
                s_reconstruction = DualDiffusionPipeline.invert_s_samples(s_response, s_reconstruction)
                f_reconstruction = torch.zeros_like(fft_input)
                f_reconstruction = DualDiffusionPipeline.invert_f_samples(f_response, f_reconstruction)

                ifft = torch.fft.irfft(F.pad(fft_input, (0, 1)), norm="ortho")
                ifft.cpu().numpy().tofile("./dataset/ifft.raw")

                with open("./dataset/reconstruction.dual", 'wb') as f:
                    s_reconstruction.cpu().numpy().tofile(f)
                    f_reconstruction.cpu().numpy().tofile(f)

            #relative_dirpath = os.path.relpath(dirpath, input_dir)
            #output_file_dir = os.path.join(output_dir, relative_dirpath)
            #os.makedirs(output_file_dir, exist_ok=True)
            #output_file = os.path.join(output_file_dir, filename)
            #output_file = os.path.splitext(output_file)[0] + '.dual'

            output_file = os.path.join(output_dir, filename)
            output_file = os.path.splitext(output_file)[0] + '.raw'

            with open(output_file, 'wb') as f:
                raw_input.cpu().numpy().tofile(f)
                fft_input.cpu().numpy().tofile(f)

            print(f"Processed '{input_file}'")
            total_processed += 1

    s_avg_std /= total_processed
    f_avg_std /= total_processed

    print(f"\nTotal processed: {total_processed}")
    print(f"Average s std: {s_avg_std}")
    print(f"Average f std: {f_avg_std}")

    print(f"\nSaving config file to '{config_path}'...")
    config = {
        "sample_len": sample_len,
        "s_avg_std": s_avg_std,
        "f_avg_std": f_avg_std,
        "s_resolution": s_resolution,
        "f_resolution": f_resolution,
        }
    with open(config_path, 'w') as f:
        json.dump(config, f)

if __name__ == "__main__":

    if not torch.cuda.is_available():
        print("Error: PyTorch not compiled with CUDA support or CUDA unavailable")
        exit(1)

    #decode_source_to_raw(SOURCE_DIR, RAW_DIR, SAMPLE_RATE)

    preprocess_raw_to_dual(RAW_DIR,
                           SAMPLE_DIR,
                           DATA_CFG_PATH,
                           SAMPLE_LEN,
                           FADEOUT_LEN,
                           FADEOUT_STD,
                           S_RESOLUTION,
                           F_RESOLUTION,
                           ENABLE_DEBUG_DUMPS,
                           )

    DualDiffusionPipeline.create_new(DATA_CFG_PATH, NEW_MODEL_PATH)
    print(f"Created new DualDiffusion model with config at '{NEW_MODEL_PATH}'")