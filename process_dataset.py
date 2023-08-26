import os
import multiprocessing
import numpy as np
import torch

from lg_diffusion_pipeline import LGDiffusionPipeline

NUM_PROCESSES = multiprocessing.cpu_count() // 2
FFMPEG_PATH = './dataset/ffmpeg_gme/bin/ffmpeg.exe'
#FFMPEG_PATH = 'ffmpeg'
#FFMPEG_PATH = None
SOURCE_DIR = './dataset/spc'
#SOURCE_DIR = './dataset/source'
INPUT_FORMATS = ('.spc')
#INPUT_FORMATS = ('.mp3', '.flac')
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
NEW_MODEL_PATH = './models/new_lgdiffusion'



if FFMPEG_PATH is None: import ffmpeg
else: import subprocess
    
def decode_source_files_to_raw(input_files):

    num_processed = 0
    num_error = 0

    for dirpath, filename in input_files: 
        
        input_file = os.path.join(dirpath, filename)
        
        relative_dirpath = os.path.relpath(dirpath, SOURCE_DIR)
        output_file_dir = os.path.join(OUTPUT_RAW_DIR, relative_dirpath)
        os.makedirs(output_file_dir, exist_ok=True)
        
        output_file = os.path.join(output_file_dir, filename)
        output_file = os.path.splitext(output_file)[0] + '.raw'
        
        if FFMPEG_PATH is None:
            try:
                ffmpeg.input(input_file).output(output_file, t=MAXIMUM_RAW_LENGTH, ac=1, ar=SAMPLE_RATE, format='s16le').run(quiet=True)
            except Exception as e:
                print(f"Error processing '{input_file}': {e}")
                num_error += 1
                continue
            print(f"Processed '{input_file}'")
            num_processed += 1
        else:
            result = subprocess.run([FFMPEG_PATH,
                                        '-i', input_file,
                                        '-t', MAXIMUM_RAW_LENGTH,
                                        '-ac', '1',
                                        '-ar', str(SAMPLE_RATE),
                                        '-f', 's16le',
                                        output_file])
            if result.returncode != 0:
                print(f"Error processing '{input_file}'")
                num_error += 1
            else:
                print(f"Processed '{input_file}'")
                num_processed += 1
    
    return num_processed, num_error


def decode_source_to_raw():

    input_files = []
    for dirpath, _, filenames in os.walk(SOURCE_DIR):
        for filename in filenames: 
            
            file_ext = os.path.splitext(filename)[1].lower()
            if not file_ext in INPUT_FORMATS:
                continue

            input_files.append((dirpath, filename))

    num_processed = 0; num_error = 0
    pool = multiprocessing.Pool(NUM_PROCESSES)  
    for proc_num_processed, proc_num_error in pool.imap_unordered(decode_source_files_to_raw, input_files):
        num_processed += proc_num_processed
        num_error += proc_num_error

    print("")
    print(f"\nTotal processed: {num_processed}")
    print(f"Total errors: {num_error}")
    print("")

def preprocess_raw_files_to_sample(input_files):

    sample_crop_width = LGDiffusionPipeline.get_sample_crop_width(MODEL_PARAMS)
    total_processed = 0; num_skipped = 0
    avg_mean = 0; avg_std = 0
    
    for input_file in input_files:
        
        raw_input = np.fromfile(input_file, dtype=np.int16, count=MAXIMUM_SAMPLE_LENGTH)
        if len(raw_input) < MINIMUM_SAMPLE_LENGTH:
            print(f"Skipping '{input_file}' due to insufficient length")
            num_skipped += 1
            continue

        output_file_raw = os.path.join(OUTPUT_SAMPLE_DIR, f"{total_processed+1}.raw")
        raw_input.tofile(output_file_raw)

        raw_input = torch.from_numpy(raw_input).to("cuda").type(torch.float32) / 32768.
        sample = LGDiffusionPipeline.raw_to_freq(raw_input[:sample_crop_width].unsqueeze(0), MODEL_PARAMS)
        avg_mean += sample.mean(dim=(1, 2, 3)).item()
        avg_std += sample.std(dim=(1, 2, 3)).item()

        if total_processed == 0:
            print(f"Sample shape: {sample.shape}")
            sample.cpu().numpy().tofile("./output/debug_sample.raw")
            reconstructed_raw_sample = LGDiffusionPipeline.freq_to_raw(sample, MODEL_PARAMS)
            reconstructed_raw_sample.cpu().numpy().tofile("./output/debug_reconstructed_raw.raw")

        print(f"Processed '{input_file}'")
        total_processed += 1

    return total_processed, num_skipped, avg_mean, avg_std

def preprocess_raw_to_sample():

    os.makedirs(OUTPUT_SAMPLE_DIR, exist_ok=True)
    
    input_files = []
    for dirpath, _, filenames in os.walk(INPUT_RAW_DIR):
        for filename in filenames:

            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext != ".raw":
                continue

            input_files.append(os.path.join(dirpath, filename))
    
    total_processed = 0; num_skipped = 0
    avg_mean = 0; avg_std = 0
    pool = multiprocessing.Pool(NUM_PROCESSES)
    for proc_num_processed, proc_num_skipped, proc_avg_mean, proc_avg_std in pool.imap_unordered(preprocess_raw_files_to_sample, input_files):
        total_processed += proc_num_processed
        num_skipped += proc_num_skipped
        avg_mean += proc_avg_mean
        avg_std += proc_avg_std
    
    print(f"\nTotal processed: {total_processed}  Total skipped: {num_skipped}")
    print(f"Average mean: {avg_mean/total_processed}")
    print(f"Average std: {avg_std/total_processed}")

    return avg_std/total_processed

if __name__ == "__main__":

    if not torch.cuda.is_available():
        print("Error: PyTorch not compiled with CUDA support or CUDA unavailable")
        exit(1)

    #decode_source_to_raw()
    #avg_std = preprocess_raw_to_sample()
    #MODEL_PARAMS["avg_std"] = avg_std #1.4706064990037718375

    pipeline = LGDiffusionPipeline.create_new(MODEL_PARAMS, NEW_MODEL_PATH)
    print(f"Created new LGDiffusion model with config at '{NEW_MODEL_PATH}'")
