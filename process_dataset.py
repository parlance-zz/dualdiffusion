import os
import numpy as np
import torch

from lg_diffusion_pipeline import LGDiffusionPipeline

#FFMPEG_PATH = None
#SOURCE_DIR = './dataset/source'
#INPUT_FORMATS = ('.mp3', '.flac')

FFMPEG_PATH = './dataset/ffmpeg_gme/bin/ffmpeg.exe'
SOURCE_DIR = './dataset/spc'
INPUT_FORMATS = ('.spc')
OUTPUT_RAW_DIR = './dataset/raw'
MAXIMUM_RAW_LENGTH = "00:01:30"

FORMAT = "complex_1channel"
NOISE_FLOOR = 0. #1e-3
SAMPLE_RATE = 8000
NUM_CHUNKS = 256
SAMPLE_RAW_LENGTH = 65536
OVERLAPPED = False
SPATIAL_WINDOW_LENGTH = 2048
FREQ_EMBEDDING_DIM = 30
MINIMUM_SAMPLE_LENGTH = SAMPLE_RAW_LENGTH * 4
MAXIMUM_SAMPLE_LENGTH = SAMPLE_RAW_LENGTH * 4
INPUT_RAW_DIR = OUTPUT_RAW_DIR
OUTPUT_SAMPLE_DIR = './dataset/samples'

NEW_MODEL_PATH = './models/new_lgdiffusion'

def decode_source_to_raw(input_dir, output_dir, sample_rate, input_formats, max_raw_length, ffmpeg_path=None):

    if ffmpeg_path is None:
        import ffmpeg
    else:
        import subprocess

    num_processed = 0
    num_error = 0

    for dirpath, _, filenames in os.walk(input_dir):
        
        if num_error >= 10:
            print("Too many errors, aborting")
            break

        for filename in filenames: 
            
            file_ext = os.path.splitext(filename)[1]
            if not file_ext in input_formats:
                continue

            input_file = os.path.join(dirpath, filename)
            
            relative_dirpath = os.path.relpath(dirpath, input_dir)
            output_file_dir = os.path.join(output_dir, relative_dirpath)
            os.makedirs(output_file_dir, exist_ok=True)
            
            output_file = os.path.join(output_file_dir, filename)
            output_file = os.path.splitext(output_file)[0] + '.raw'
            
            if ffmpeg_path is None:
                try:
                    ffmpeg.input(input_file).output(output_file, t=max_raw_length, ac=1, ar=sample_rate, format='s16le').run(quiet=True)
                except Exception as e:
                    print(f"Error processing '{input_file}': {e}")
                    num_error += 1
                    continue
                print(f"Processed '{input_file}'")
                num_processed += 1
            else:
                result = subprocess.run([ffmpeg_path,
                                         '-i', input_file,
                                         '-t', max_raw_length,
                                         '-ac', '1',
                                         '-ar', str(sample_rate),
                                         '-f', 's16le',
                                         output_file])
                if result.returncode != 0:
                    print(f"Error processing '{input_file}'")
                    num_error += 1
                else:
                    print(f"Processed '{input_file}'")
                    num_processed += 1

    print("")
    print(f"\nTotal processed: {num_processed}")
    print(f"Total errors: {num_error}")
    print("")

def preprocess_raw_to_sample(input_dir: str, output_sample_dir: str, min_sample_length, max_sample_length, model_params):

    os.makedirs(output_sample_dir, exist_ok=True)
    
    sample_crop_width = LGDiffusionPipeline.get_sample_crop_width(model_params)
    total_processed = 0; num_skipped = 0
    avg_mean = 0; avg_std = 0

    for dirpath, _, filenames in os.walk(input_dir):

        for filename in filenames:
            
            input_file = os.path.join(dirpath, filename)
            
            raw_input = np.fromfile(input_file, dtype=np.int16, count=max_sample_length)
            if len(raw_input) < min_sample_length:
                print(f"Skipping '{input_file}' due to insufficient length")
                num_skipped += 1
                continue

            output_file_raw = os.path.join(output_sample_dir, f"{total_processed+1}.raw")
            raw_input.tofile(output_file_raw)

            raw_input = torch.from_numpy(raw_input).to("cuda").type(torch.float32) / 32768.
            sample = LGDiffusionPipeline.raw_to_freq(raw_input[:sample_crop_width].unsqueeze(0), model_params)
            avg_mean += sample.mean(dim=(1, 2, 3)).item()
            avg_std += sample.std(dim=(1, 2, 3)).item()

            if total_processed == 0:
                print(f"Sample shape: {sample.shape}")
                sample.cpu().numpy().tofile("./output/debug_sample.raw")
                reconstructed_raw_sample = LGDiffusionPipeline.freq_to_raw(sample, model_params)
                reconstructed_raw_sample.cpu().numpy().tofile("./output/debug_reconstructed_raw.raw")

            print(f"Processed '{input_file}'")
            total_processed += 1

    print(f"\nTotal processed: {total_processed}  Total skipped: {num_skipped}")
    print(f"Average mean: {avg_mean/total_processed}")
    print(f"Average std: {avg_std/total_processed}")

    return avg_std/total_processed

if __name__ == "__main__":

    if not torch.cuda.is_available():
        print("Error: PyTorch not compiled with CUDA support or CUDA unavailable")
        exit(1)

    MODEL_PARAMS = {
        "format": FORMAT,
        "sample_raw_length": SAMPLE_RAW_LENGTH,
        "num_chunks": NUM_CHUNKS,
        "overlapped": OVERLAPPED,
        "noise_floor": NOISE_FLOOR,
        "spatial_window_length": SPATIAL_WINDOW_LENGTH,
        "sample_rate": SAMPLE_RATE,
        "freq_embedding_dim": FREQ_EMBEDDING_DIM,
    }

    """
    decode_source_to_raw(SOURCE_DIR,
                         OUTPUT_RAW_DIR,
                         SAMPLE_RATE,
                         INPUT_FORMATS,
                         MAXIMUM_RAW_LENGTH,
                         ffmpeg_path=FFMPEG_PATH)
    """
    """
    avg_std = preprocess_raw_to_sample(INPUT_RAW_DIR,
                                       OUTPUT_SAMPLE_DIR,
                                       MINIMUM_SAMPLE_LENGTH,
                                       MAXIMUM_SAMPLE_LENGTH,
                                       MODEL_PARAMS)
    MODEL_PARAMS["avg_std"] = avg_std #1.4706064990037718375
    """

    pipeline = LGDiffusionPipeline.create_new(MODEL_PARAMS, NEW_MODEL_PATH)
    print(f"Created new LGDiffusion model with config at '{NEW_MODEL_PATH}'")
