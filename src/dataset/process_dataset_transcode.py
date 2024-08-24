# MIT License
#
# Copyright (c) 2023 Christopher Friesen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import utils.config as config

import os
import subprocess

import torchaudio


def decode_source_file(input_file, sample_num):

    input_dir, input_filename = os.path.split(input_file)
    input_filename, _ = os.path.splitext(input_filename)
    input_dir = os.path.basename(input_dir)
    output_filename = f"{input_dir} - {input_filename}{OUTPUT_SAMPLE_FORMAT}"
    output_file = os.path.join(OUTPUT_SAMPLE_DIR, output_filename)
    
    if os.path.exists(output_file):
        return False

    args = [FFMPEG_PATH,
            '-loglevel', 'error','-hide_banner',
            '-i', input_file,
            '-t', MAXIMUM_SAMPLE_LENGTH,
            '-ac', str(OUTPUT_NUM_CHANNELS),
            '-ar', str(OUTPUT_SAMPLE_RATE),
            ]#'-filter:a', 'loudnorm']
    
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
                print(f"Skipping '{input_file}' due to insufficient length")
                os.remove(output_file)
                return False

        else:
            #audio, _ = torchaudio.load(output_file)
            #if audio.abs().amax() >= 0.99:
            #    print("WARNING - CLIPPING in file:", output_file)
            #    _ = input("Press enter to continue...")

            if torchaudio.info(output_file).num_frames < MINIMUM_SAMPLE_LENGTH:
                print(f"Skipping '{input_file}' due to insufficient length")
                os.remove(output_file)
                return False
            
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
        processed = decode_source_file(input_file, sample_num)
        total_processed += int(processed)
        sample_num += 1

    print("")
    print(f"\nTotal processed: {total_processed}")
    print(f"Total errors / skipped: {len(input_files)-total_processed}")
    print("")

if __name__ == "__main__":

    FFMPEG_PATH = config.FFMPEG_PATH
    SOURCE_DIR = config.DATASOURCE_PATH
    OUTPUT_SAMPLE_DIR = config.DATASET_PATH

    dataset_cfg = config.load_json(os.path.join(config.CONFIG_PATH, "dataset", "dataset.json"))
    SOURCE_FORMATS = dataset_cfg["datasource_formats"]
    OUTPUT_SAMPLE_FORMAT = dataset_cfg["dataset_format"]
    OUTPUT_SAMPLE_RATE = dataset_cfg["dataset_sample_rate"]
    OUTPUT_NUM_CHANNELS = dataset_cfg["dataset_num_channels"]
    MINIMUM_SAMPLE_LENGTH = dataset_cfg["dataset_min_sample_length"]
    MAXIMUM_SAMPLE_LENGTH = dataset_cfg["dataset_max_sample_length"]

    RESUME_FROM = -1

    if os.path.exists(OUTPUT_SAMPLE_DIR):        
        existing_files = os.listdir(OUTPUT_SAMPLE_DIR)
        if len(existing_files) > 0:

            print(f"Warning: Output folder already exists '{OUTPUT_SAMPLE_DIR}'")
            if input("Resume processing? (y/n): ").lower() not in ["y","yes"]: exit()
    else:
        os.makedirs(OUTPUT_SAMPLE_DIR, exist_ok=True)

    decode_source_to_samples()