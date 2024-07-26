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
import json
import subprocess
import random
import math
import os


def extract_sample_metadata(sample_file):

    print(sample_file)
    if not os.path.exists(sample_file):
        return None

    sample_rel_path = os.path.relpath(sample_file, OUTPUT_SAMPLE_DIR)

    metadata = { 'file_name': sample_rel_path.replace('\\', '/'),
                    'game': '',
                    'song': '',
                    'author': [],
                }

    args = [FFMPEG_PATH,
            '-y', '-loglevel', 'error','-hide_banner',
            '-i', sample_file,
            '-f', 'ffmetadata',
            '_meta.txt'
            ]            
    result = subprocess.run(args)

    if result.returncode != 0:
        print(f"Error extracting metadata for '{sample_file}'")
        return None
    try:
        with open('_meta.txt', 'r') as f:

            for line in f.read().split('\n'):
                field = line.split('=')[0].strip().lower()
                if field in ['game', 'author', 'song']:
                    
                    value = line.split('=')[1].strip()

                    if field == "author":
                        metadata['author'].extend([x.strip() for x in value.split(',')])
                    else:
                        metadata[field] = value

    except Exception as e:
        print(f"Error reading metadata for '{sample_file}' - {e}")
        return None
    finally:
        try:
            os.remove('_meta.txt')
        except:
            pass

    return metadata

def extract_dataset_metadata():

    print(f"Extracting metadata from dataset in '{OUTPUT_SAMPLE_DIR}'...")

    sample_files = []
    for dirpath, _, filenames in os.walk(OUTPUT_SAMPLE_DIR):
        for filename in filenames:         
            if os.path.splitext(filename)[1].lower() != OUTPUT_SAMPLE_FORMAT:
                continue

            sample_file = os.path.join(dirpath, filename)
            sample_files.append(sample_file)

    total_processed = 0
    metadata = []
    for sample_file in sample_files:
        data = extract_sample_metadata(sample_file)

        if data is not None:
            metadata.append(data)
            total_processed += 1

    print(f"\nTotal processed: {total_processed}")
    print(f"Total errors / skipped: {len(sample_files)-total_processed}")

    return metadata

def process_dataset_metadata(metadata):

    game_ids = {}
    author_ids = {}
    game_sample_count = {}

    train_metadata = []
    validation_metadata = []

    for data in metadata:

        if not data['game'] in game_ids:
            game_ids[data['game']] = len(game_ids)
            game_sample_count[data['game']] = 1
            validation_metadata.append(data)
        else:
            game_sample_count[data['game']] += 1
            train_metadata.append(data)

        for author in data['author']:
            if not author in author_ids:
                author_ids[author] = len(author_ids)
    

    for data in random.sample(validation_metadata, len(validation_metadata)):
        if game_sample_count[data['game']] < MIN_NUM_CLASS_SAMPLES_FOR_VALIDATION:
            train_metadata.append(data)
            validation_metadata.remove(data)

    for data in random.sample(validation_metadata, len(validation_metadata)):
        if len(validation_metadata) > MAX_NUM_VALIDATION_SAMPLES:
            train_metadata.append(data)
            validation_metadata.remove(data)

    print(f"Total Songs: {len(metadata)}")
    print(f"Total Games: {len(game_ids)}")
    print(f"Total Authors: {len(author_ids)}")
    print(f"Total Training Samples: {len(train_metadata)}")
    print(f"Total Validation Samples: {len(validation_metadata)}")
    
    def write_split_metadata(metadata, split_name):
        try:
            metadata_file = os.path.join(OUTPUT_SAMPLE_DIR, f'{split_name}.jsonl')
            with open(metadata_file, 'w') as f:
                for data in metadata:
                    data['game_id'] = game_ids[data['game']]
                    data['author_id'] = [author_ids[x] for x in data['author']]
                    f.write(json.dumps(data) + '\n')
            
            print(f"Metadata written to {metadata_file}")

        except Exception as e:
            print(f"Error writing metadata file to {metadata_file} - {e}")

    write_split_metadata(train_metadata, 'train')
    write_split_metadata(validation_metadata, 'validation')

    dataset_info = {
        'features': {
            'song' : { 'type': 'string' },
            'game' : { 'type': 'string' },
            'author' : {
                'type': 'list',
                'value_type': { 'type': 'string' }
            },
            'game_id': {
                'type': 'int',
                'num_classes': len(game_ids),
            },
            'author_id': {
                'type': 'list',
                'value_type': {
                    'type': 'int',
                    'num_classes': len(author_ids),
                }
            },
        },
        'games': game_ids,
        'authors': author_ids,
        }

    try:
        dataset_info_file = os.path.join(OUTPUT_SAMPLE_DIR, 'dataset_infos', 'dataset_info.json')
        os.makedirs(os.path.join(OUTPUT_SAMPLE_DIR, 'dataset_infos'), exist_ok=True)

        with open(dataset_info_file, 'w') as f:
            f.write(json.dumps(dataset_info, indent=4))

        print(f"Dataset info written to {dataset_info_file}")

    except Exception as e:
        print(f"Error writing dataset info file to {dataset_info_file} - {e}")

def organize_dataset(): # organize samples into directories of 10,000 samples each due to huggingface limitations

    files = []
    for dirpath, _, filenames in os.walk(OUTPUT_SAMPLE_DIR):
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() == OUTPUT_SAMPLE_FORMAT:
                files.append(os.path.join(dirpath, filename))

    files.sort(key=lambda x: os.path.basename(x))
    
    num_samples = len(files)
    num_directories = math.ceil(num_samples / 10000)

    for i in range(1, num_directories+1):
        directory = os.path.join(OUTPUT_SAMPLE_DIR, str(i))
        os.makedirs(directory, exist_ok=True)

    for i, file in enumerate(files):
        directory = os.path.join(OUTPUT_SAMPLE_DIR, str((i // 10000) + 1))
        new_file = os.path.join(directory, os.path.basename(file))

        if file != new_file:
            print(f"{file} -> {new_file}")
            os.rename(file, new_file)
    
    for i in range(1, num_directories+1):
        directory = os.path.join(OUTPUT_SAMPLE_DIR, str(i))
        if not os.listdir(directory):
            os.rmdir(directory)

    print(f"Organized {num_samples} samples into {num_directories} directories")

if __name__ == "__main__":

    dataset_cfg = config.load_json(os.path.join(config.CONFIG_PATH, "dataset.json"))

    FFMPEG_PATH = config.FFMPEG_PATH
    OUTPUT_SAMPLE_DIR = config.DATASET_PATH
    OUTPUT_SAMPLE_FORMAT = dataset_cfg["dataset_format"]

    MIN_NUM_CLASS_SAMPLES_FOR_VALIDATION = 8 # minimum number of examples of sample class to be included in validation set
    MAX_NUM_VALIDATION_SAMPLES = 100         # maximum number of samples to be included in validation set

    if not os.path.exists(OUTPUT_SAMPLE_DIR):        
        print(f"Error: Output directory '{OUTPUT_SAMPLE_DIR}' does not exist")
        exit(1)

    organize_dataset()
    process_dataset_metadata(extract_dataset_metadata())

