from utils import config

import torchaudio

import os
import subprocess
import shutil

folder_path = "D:/dualdiffusion/dataset/samples_newest"
backup_path = "D:/dualdiffusion/dataset/backup/truncated"
max_duration = 300  # max length in seconds

def truncate_flac_files(directory, max_length, backup_path=None):

    if not os.path.isdir(directory):
        print(f"The directory {directory} does not exist.")
        return

    for root, _, files in os.walk(directory):
        for file in files:
            file_name_no_ext, file_ext = os.path.splitext(file)
            if file_ext.lower() == ".flac":
                input_path = os.path.join(root, file)
                output_path = os.path.join(root, f"{file_name_no_ext}_trnk{file_ext}")

                if os.path.isfile(output_path) or "_trnk" in file_name_no_ext:
                    continue
                try:
                    audio_info = torchaudio.info(input_path)
                    audio_length = audio_info.num_frames / audio_info.sample_rate
                    if audio_length <= max_length:
                        continue
                except Exception as e:
                    print(f"Error retrieving duration for {input_path}: {e}")
                    continue

                # ffmpeg command to truncate the file while preserving format and metadata
                command = [
                    config.FFMPEG_PATH,
                    "-i", input_path,        # Input file
                    "-t", str(max_length),   # Truncate to max_length
                    "-c:a", file_ext[1:].lower(), # Re-encode to FLAC
                    "-map_metadata", "0",    # Preserve all metadata
                    output_path              # Output file
                ]

                try:
                    print(f"Processing {input_path}...")
                    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    print(f"Truncated file saved as {output_path}")

                    if backup_path is not None:
                        file_backup_path = os.path.join(backup_path, os.path.relpath(input_path, directory))
                        os.makedirs(os.path.dirname(file_backup_path), exist_ok=True)
                        shutil.move(input_path, file_backup_path)
                    else:
                        os.remove(input_path)

                except Exception as e:
                    print(f"Error processing {input_path}: {e}")

if __name__ == "__main__":
    if backup_path is None:
        if input(f"WARNING: No backup path specified. Original files will be deleted after truncation. Continue? (y/n): ").lower() not in ("y","yes"): exit()
    else:
        print(f"Truncated files will be backed up to {backup_path}")
    truncate_flac_files(folder_path, max_duration, backup_path=backup_path)
