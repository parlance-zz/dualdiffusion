import os
import subprocess
import uuid
import numpy as np
import torch

from dual_diffusion_pipeline import DualDiffusionPipeline

FFMPEG_PATH = './dataset/ffmpeg_gme/bin/ffmpeg.exe'
SOURCE_DIR = './dataset/spc'
INPUT_FORMATS = ('.spc')
OUTPUT_RAW_DIR = './dataset/raw'
MAXIMUM_RAW_LENGTH = "00:02:30"

INPUT_RAW_DIR = OUTPUT_RAW_DIR
MINIMUM_SAMPLE_LENGTH = 65536 * 8
OUTPUT_SAMPLE_DIR = './dataset/samples'
WRITE_SAMPLES = False

NOISE_FLOOR = 0. #1e-3
SAMPLE_RATE = 8000
NUM_CHUNKS = 256
SAMPLE_RAW_LENGTH = 65536*2
SPATIAL_WINDOW_LENGTH = 4096 #2048
FREQ_EMBEDDING_DIM = 0

NEW_MODEL_PATH = './models/dualdiffusion2d_28'
MODEL_PARAMS = {
    "prediction_type": "v_prediction",
    "beta_schedule": "linear", #"squaredcos_cap_v2",
    "beta_start" : 0.0001,
    "beta_end" : 0.02,
    "sample_raw_length": SAMPLE_RAW_LENGTH,
    "num_chunks": NUM_CHUNKS,
    "noise_floor": NOISE_FLOOR,
    "spatial_window_length": SPATIAL_WINDOW_LENGTH,
    "sample_rate": SAMPLE_RATE,
    "freq_embedding_dim": FREQ_EMBEDDING_DIM,
    "avg_std": [],
    "last_global_step": 0,
}
    
def decode_source_files_to_raw(input_file):

    dirpath = os.path.dirname(input_file)
    filename = os.path.basename(input_file)

    relative_dirpath = os.path.relpath(dirpath, SOURCE_DIR)
    output_file_dir = os.path.join(OUTPUT_RAW_DIR, relative_dirpath)
    os.makedirs(output_file_dir, exist_ok=True)
    
    output_file = os.path.join(output_file_dir, filename)
    output_file = os.path.splitext(output_file)[0] + '.raw'
    
    print(input_file, output_file)
    result = subprocess.run([FFMPEG_PATH,
                            '-i', input_file,
                            '-t', MAXIMUM_RAW_LENGTH,
                            '-ac', '1',
                            '-ar', str(SAMPLE_RATE),
                            '-f', 's16le',
                            output_file])
    if result.returncode != 0:
        print(f"Error processing '{input_file}'")
        return False
    else:
        print(f"Processed '{input_file}'")
        return True

def decode_source_to_raw():

    input_files = []
    for dirpath, _, filenames in os.walk(SOURCE_DIR):
        for filename in filenames: 
            
            file_ext = os.path.splitext(filename)[1].lower()
            if not file_ext in INPUT_FORMATS:
                continue

            input_file = os.path.join(dirpath, filename)
            input_files.append(input_file)

    total_processed = 0
    for input_file in input_files:
        processed = decode_source_files_to_raw(input_file)
        total_processed += int(processed)

    print("")
    print(f"\nTotal processed: {total_processed}")
    print(f"Total errors: {len(input_files)-total_processed}")
    print("")

def preprocess_raw_files_to_sample(input_file):

    sample_crop_width = DualDiffusionPipeline.get_sample_crop_width(MODEL_PARAMS)
    processed = False; mean = 0.; std = 0.
    
    print(f"Processing '{input_file}'") 
    raw_input = np.fromfile(input_file, dtype=np.int16)
    if len(raw_input) < MINIMUM_SAMPLE_LENGTH:
        print(f"Skipping '{input_file}' due to insufficient length")
    else:
        if WRITE_SAMPLES:
            output_filename = f"{str(uuid.uuid4())}.raw"
            output_file_raw = os.path.join(OUTPUT_SAMPLE_DIR, output_filename)
            raw_input.tofile(output_file_raw)

        raw_input = torch.from_numpy(raw_input).to("cuda").type(torch.float32) / 32768.
        sample = DualDiffusionPipeline.raw_to_sample(raw_input[:sample_crop_width].unsqueeze(0), MODEL_PARAMS)
        mean = sample.mean(dim=(1, 2, 3)).item()
        std = sample.std(dim=(1, 2, 3)).item()
        processed = True

    print(f"Processed '{input_file}'")
    return processed, mean, std

def preprocess_raw_to_sample():

    os.makedirs(OUTPUT_SAMPLE_DIR, exist_ok=True)
    
    input_files = []
    for dirpath, _, filenames in os.walk(INPUT_RAW_DIR):
        for filename in filenames:

            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext != ".raw":
                continue

            input_files.append(os.path.join(dirpath, filename))
    
    total_processed = 0; total_mean = 0; total_std = 0
 
    for input_file in input_files:
        processed, mean, std = preprocess_raw_files_to_sample(input_file)
        total_processed += int(processed)
        total_mean += mean
        total_std += std

    avg_mean = total_mean/total_processed
    avg_std = total_std/total_processed

    print(f"\nTotal processed: {total_processed}  Total skipped: {len(input_files)-total_processed}")
    print(f"Average mean: {avg_mean}")
    print(f"Average std: {avg_std}")

    return avg_mean, avg_std

if __name__ == "__main__":

    if not torch.cuda.is_available():
        print("Error: PyTorch not compiled with CUDA support or CUDA unavailable")
        exit(1)
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.cufft_plan_cache[0].max_size = 32 # stupid cufft memory leak

    #decode_source_to_raw()

    #avg_mean, avg_std = preprocess_raw_to_sample()
    #MODEL_PARAMS["avg_mean"] = avg_mean
    #MODEL_PARAMS["avg_std"] = avg_std

    #MODEL_PARAMS["avg_mean"] = 0.0017828649850519483
    #MODEL_PARAMS["avg_std"] = 0.563462784100214

    pipeline = DualDiffusionPipeline.create_new(MODEL_PARAMS, NEW_MODEL_PATH)
    print(f"Created new DualDiffusion model with config at '{NEW_MODEL_PATH}'")
