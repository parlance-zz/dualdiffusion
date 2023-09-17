import os
import subprocess
from dotenv import load_dotenv

import torch
import torchaudio

load_dotenv()

FFMPEG_PATH = os.environ.get("FFMPEG_PATH")
SOURCE_DIR = os.environ.get("DATASOURCE_PATH")
SOURCE_FORMATS = [x.strip().lower() for x in os.environ.get("DATASOURCE_FORMATS").split(",")]
OUTPUT_SAMPLE_DIR = os.environ.get("DATASET_PATH")
OUTPUT_SAMPLE_FORMAT=os.environ.get('DATASET_FORMAT').lower()
OUTPUT_SAMPLE_RATE = int(os.environ.get("DATASET_SAMPLE_RATE"))
MINIMUM_SAMPLE_LENGTH = 50 * OUTPUT_SAMPLE_RATE # 50 seconds
MAXIMUM_SAMPLE_LENGTH = "00:02:00"

def decode_source_file(input_file, sample_num):

    output_file = os.path.join(OUTPUT_SAMPLE_DIR, f"{sample_num}{OUTPUT_SAMPLE_FORMAT}")
    
    args = [FFMPEG_PATH,
            '-loglevel', 'error','-hide_banner',
            '-i', input_file,
            '-t', MAXIMUM_SAMPLE_LENGTH,
            '-ac', '1',
            '-ar', str(OUTPUT_SAMPLE_RATE)]

    if OUTPUT_SAMPLE_FORMAT == '.raw': args += ['-f', 's16le']
    else: args += ['-sample_fmt', 's16']
    if OUTPUT_SAMPLE_FORMAT == '.flac': args += ['-compression_level', '12']
    elif OUTPUT_SAMPLE_FORMAT == '.mp3': args += ['-b:a', '320k']

    args.append(output_file)

    print(input_file, "->", output_file)
    result = subprocess.run(args)
    if result.returncode != 0:
        print(f"Error processing '{input_file}'")
        return False
    else:
        if OUTPUT_SAMPLE_FORMAT == '.raw':
            
            output_file_length = os.path.getsize(output_file) // 2
            if output_file_length < MINIMUM_SAMPLE_LENGTH:
                print(f"Skipping '{input_file}' due to insufficient length ({output_file_length // OUTPUT_SAMPLE_RATE}s)")
                os.remove(output_file)
                return False

        else:
            audio, _ = torchaudio.load(output_file)
            if audio.shape[1] < MINIMUM_SAMPLE_LENGTH:
                print(f"Skipping '{input_file}' due to insufficient length ({audio.shape[1] // OUTPUT_SAMPLE_RATE}s)")
                os.remove(output_file)
                return False
            else:
                del audio
            
        return True

def decode_source_to_samples():

    input_files = []
    for dirpath, _, filenames in os.walk(SOURCE_DIR):
        for filename in filenames: 
            
            file_ext = os.path.splitext(filename)[1].lower()
            if not file_ext in SOURCE_FORMATS:
                continue

            input_file = os.path.join(dirpath, filename)
            input_files.append(input_file)

    total_processed = 0
    for input_file in input_files:
        processed = decode_source_file(input_file, total_processed)
        total_processed += int(processed)

    print("")
    print(f"\nTotal processed: {total_processed}")
    print(f"Total errors / skipped: {len(input_files)-total_processed}")
    print("")

if __name__ == "__main__":

    if not torch.cuda.is_available():
        print("Error: PyTorch not compiled with CUDA support or CUDA unavailable")
        exit(1)
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.cufft_plan_cache[0].max_size = 32 # stupid cufft memory leak

    if os.path.exists(OUTPUT_SAMPLE_DIR):
        print(f"Error: Output folder already exists '{OUTPUT_SAMPLE_DIR}'")
        exit(1)
    else:
        os.makedirs(OUTPUT_SAMPLE_DIR, exist_ok=True)
        decode_source_to_samples()