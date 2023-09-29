import os
import subprocess
from dotenv import load_dotenv

import torchaudio

load_dotenv()

FFMPEG_PATH = os.environ.get("FFMPEG_PATH")
SOURCE_DIR = os.environ.get("DATASOURCE_PATH")
SOURCE_FORMATS = [x.strip().lower() for x in os.environ.get("DATASOURCE_FORMATS").split(",")]
OUTPUT_SAMPLE_DIR = os.environ.get("DATASET_PATH")
OUTPUT_SAMPLE_FORMAT=os.environ.get('DATASET_FORMAT').lower()
OUTPUT_SAMPLE_RATE = int(os.environ.get("DATASET_SAMPLE_RATE"))
OUTPUT_NUM_CHANNELS = int(os.environ.get("DATASET_NUM_CHANNELS"))
MINIMUM_SAMPLE_LENGTH = int(os.environ.get("DATASET_MINIMUM_SAMPLE_LENGTH"))
MAXIMUM_SAMPLE_LENGTH = os.environ.get("DATASET_MAXIMUM_SAMPLE_LENGTH")

RESUME_FROM = -1

def decode_source_file(input_file, sample_num):

    output_file = os.path.join(OUTPUT_SAMPLE_DIR, f"{sample_num}{OUTPUT_SAMPLE_FORMAT}")
    
    args = [FFMPEG_PATH,
            '-loglevel', 'error','-hide_banner',
            '-i', input_file,
            '-t', MAXIMUM_SAMPLE_LENGTH,
            '-ac', str(OUTPUT_NUM_CHANNELS),
            '-ar', str(OUTPUT_SAMPLE_RATE),
            '-filter:a', 'loudnorm']
    
    if OUTPUT_SAMPLE_FORMAT == '.raw': args += ['-f', 's16le']
    if OUTPUT_SAMPLE_FORMAT == '.flac': args += ['-sample_fmt', 's16']
    elif OUTPUT_SAMPLE_FORMAT == '.mp3': args += ['-codec:a', 'libmp3lame', '-qscale:a', '2']

    args.append(output_file)

    print(input_file, "->", output_file)
    result = subprocess.run(args)
    if result.returncode != 0:
        print(f"Error processing '{input_file}'")
        return False
    else:
        if OUTPUT_SAMPLE_FORMAT == '.raw':       
            output_file_length = os.path.getsize(output_file) // 2 // OUTPUT_NUM_CHANNELS
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
    sample_num = 0
    for input_file in input_files:
        if sample_num > RESUME_FROM:
            processed = decode_source_file(input_file, sample_num)
            total_processed += int(processed)
        sample_num += 1

    print("")
    print(f"\nTotal processed: {total_processed}")
    print(f"Total errors / skipped: {len(input_files)-total_processed}")
    print("")

if __name__ == "__main__":

    if os.path.exists(OUTPUT_SAMPLE_DIR):
        print(f"Warning: Output folder already exists '{OUTPUT_SAMPLE_DIR}'")
        if input("Resume processing? (y/n): ").lower() not in ["y","yes"]: exit()
        
        existing_files = os.listdir(OUTPUT_SAMPLE_DIR)
        if len(existing_files) > 0:
            existing_files.sort()
            RESUME_FROM = int(os.path.splitext(existing_files[-1])[0])
            print(f"Resuming from sample num {RESUME_FROM}...")

    else:
        os.makedirs(OUTPUT_SAMPLE_DIR, exist_ok=True)
        decode_source_to_samples()