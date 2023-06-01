import os
import ffmpeg
import numpy as np
import torch
import json
import shutil

import torch.nn.functional as F

from dual_diffusion_pipeline import DualDiffusionPipeline

SOURCE_DIR = './dataset/source'
RAW_DIR = './dataset/raw'
DATA_CFG_PATH = './dataset/config.json'
NEW_MODEL_PATH = './models/new_dualdiffusion'
ENABLE_DEBUG_DUMPS = True
SAMPLE_RATE = 44100

SAMPLE_DIR = './dataset/dual'
SAMPLE_LEN = 2 ** 21
FADEOUT_LEN = 2 ** 17
FADEOUT_STD = 10.

S_RESOLUTION = 8192
F_RESOLUTION = 8192 #1024 (abs)

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

def preprocess_raw_to_dual(input_dir: str,
                           output_dir: str,
                           config_path: str,
                           sample_len: int,
                           fadeout_len: int,
                           fadeout_std: float,
                           s_resolution: int,
                           f_resolution: int,
                           enable_debug_dumps: bool=False,
                           ):

    fadeout_window = torch.exp(-torch.square(torch.linspace(0., 1., fadeout_len, device="cuda")) * fadeout_std)

    os.makedirs(output_dir, exist_ok=True)

    total_processed = 0
    s_avg_mean = 0.; s_avg_std = 0.
    f_avg_mean = 0.; f_avg_std = 0.
    avg_freq_response = torch.zeros(sample_len//2, device="cuda")

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
            raw_input /= torch.max(torch.abs(raw_input))
            
            fft_input = torch.fft.fft(raw_input, norm="ortho")
            fft_input[len(fft_input)//2:] = 0.
            raw_input = torch.fft.ifft(fft_input, norm="ortho")
            fft_input = fft_input[:len(fft_input)//2]

            avg_freq_response += torch.abs(fft_input)

            s_response = DualDiffusionPipeline.get_s_samples(raw_input, s_resolution)  
            s_avg_mean += s_response.mean().item()
            s_avg_std += s_response.std().item()

            f_response = DualDiffusionPipeline.get_f_samples(fft_input, f_resolution)
            f_avg_mean += f_response.mean().item()
            f_avg_std += f_response.std().item()

            output_file = os.path.join(output_dir, filename)
            output_file = os.path.splitext(output_file)[0] + '.raw'

            #with open(output_file, 'wb') as f:
            #    raw_input.cpu().numpy().tofile(f)
            #    fft_input.cpu().numpy().tofile(f)

            if (total_processed) == 0 and enable_debug_dumps:
                s_reconstructed  = DualDiffusionPipeline.invert_s_samples(s_response, raw_input)
                f_reconstructed  = DualDiffusionPipeline.invert_f_samples(f_response, fft_input)
                
                s_response.cpu().numpy().tofile("./dataset/s_response.raw")
                s_reconstructed.cpu().numpy().tofile("./dataset/s_reconstructed.raw")
                f_response.cpu().numpy().tofile("./dataset/f_response.raw")                
                f_reconstructed.cpu().numpy().tofile("./dataset/f_reconstructed.raw")

                #exit(0)    

            print(f"Processed '{input_file}'")
            total_processed += 1

    print(f"\nTotal processed: {total_processed}")

    s_avg_mean /= total_processed
    s_avg_std /= total_processed
    f_avg_mean /= total_processed
    f_avg_std /= total_processed
    print(f"\nAverage s mean: {s_avg_mean}  Average s std: {s_avg_std}")
    print(f"Average f mean: {f_avg_mean}  Average f std: {f_avg_std}")

    avg_freq_response /= total_processed
    avg_freq_response.cpu().numpy().tofile("./dataset/avg_freq_response.raw")

    print(f"\nSaving config file to '{config_path}'...")
    config = {
        "sample_len": sample_len,
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

    # copy avg frequency response to model path
    shutil.copyfile("./dataset/avg_freq_response.raw", os.path.join(NEW_MODEL_PATH, "avg_freq_response.raw"))