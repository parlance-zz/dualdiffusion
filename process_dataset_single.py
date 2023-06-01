import os
import ffmpeg
import numpy as np
import torch
import shutil

import torch.nn.functional as F

from single_diffusion_pipeline import SingleDiffusionPipeline

torch.backends.cuda.cufft_plan_cache[0].max_size = 8 # stupid cufft memory leak

SOURCE_DIR = './dataset/source'
RAW_DIR = './dataset/raw'
NEW_MODEL_PATH = './models/new_singlediffusion'
ENABLE_DEBUG_DUMPS = True
RAW_SAMPLE_RATE = 44100
DOWN_SAMPLE_RATE = 8000
#COMPRESS_FACTOR = 0#255.

SAMPLE_DIR = './dataset/single'
SAMPLE_SIZE = 8192 #16384

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
                ffmpeg.input(input_file).output(output_file, ac=1, ar=sample_rate, format='f32le').run(quiet=True)
            except Exception as e:
                print(f"Error processing '{input_file}': {e}")

            print(f"Processed '{input_file}'")

def preprocess_raw_to_single(input_dir: str,
                             output_dir: str,
                             sample_size: int,
                             enable_debug_dumps: bool=False,
                             ):

    os.makedirs(output_dir, exist_ok=True)

    total_processed = 0
    for dirpath, _, filenames in os.walk(input_dir):

        for filename in filenames:
            
            input_file = os.path.join(dirpath, filename)
            
            raw_input = np.fromfile(input_file, dtype=np.float32)
            raw_input = torch.from_numpy(raw_input).to("cuda")
            
            if len(raw_input) % 2 != 0:
                raw_input = raw_input[:-1]
            if len(raw_input) < sample_size: # skip samples that are too short
                print(f"Skipping '{input_file}': (resampled) input length {len(raw_input)} too short")
                continue

            raw_input /= torch.max(torch.abs(raw_input))
            raw_input = SingleDiffusionPipeline.get_analytic_audio(raw_input)
            raw_input = torch.cat((torch.zeros(sample_size, device=raw_input.device, dtype=torch.complex64), raw_input,))
            
            output_file = os.path.join(output_dir, filename)
            output_file = os.path.splitext(output_file)[0] + '.raw'

            raw_input.cpu().numpy().tofile(output_file)

            #if (total_processed) == 0 and enable_debug_dumps:
            #    uncompressed_input = SingleDiffusionPipeline.decompress_audio(raw_input, compress_factor)
            #    uncompressed_input.cpu().numpy().tofile("./dataset/uncompressed.raw")
                #exit(0)    

            print(f"Processed '{input_file}'")
            total_processed += 1

    print(f"\nTotal processed: {total_processed}")

if __name__ == "__main__":

    if not torch.cuda.is_available():
        print("Error: PyTorch not compiled with CUDA support or CUDA unavailable")
        exit(1)

    #decode_source_to_raw(SOURCE_DIR, RAW_DIR, DOWN_SAMPLE_RATE)

    preprocess_raw_to_single(RAW_DIR,
                             SAMPLE_DIR,
                             SAMPLE_SIZE,
                             ENABLE_DEBUG_DUMPS,
                             )

    SingleDiffusionPipeline.create_new(SAMPLE_SIZE,
                                       DOWN_SAMPLE_RATE,
                                       NEW_MODEL_PATH)
    
    print(f"Created new SingleDiffusion model at '{NEW_MODEL_PATH}'")

    # copy avg frequency response to model path
    #shutil.copyfile("./dataset/avg_freq_response.raw", os.path.join(NEW_MODEL_PATH, "avg_freq_response.raw"))