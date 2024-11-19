import os
import shutil

def move_files_with_prefix(root_src, root_dest, album_name, prefix):
    src_path = os.path.join(root_src, album_name)
    dest_path = os.path.join(root_dest, album_name)

    # Ensure destination folder exists
    
    created_dir = False

    # Traverse through the source directory and its subdirectories
    for subdir, _, files in os.walk(src_path):
        for file in files:
            if file.startswith(prefix):
                # Construct full file path in source and destination
                src_file_path = os.path.join(subdir, file)
                dest_file_path = os.path.join(dest_path, file.replace(prefix, ''))
                
                # Move file to the destination
                if not created_dir:
                    os.makedirs(dest_path, exist_ok=True)
                shutil.move(src_file_path, dest_file_path)
                print(f"Moved {src_file_path} to {dest_file_path}")

# Example usage
root_src = 'd:/dualdiffusion/dataset/psf/to_transcode'
root_dest = 'd:/dualdiffusion/dataset/psf/transcoded'
prefix = 'transcoded_'  # Replace with the specific prefix you're looking for

# get name of all folders directly under root_src
for album_name in os.listdir(root_src):
    if os.path.isdir(os.path.join(root_src, album_name)):
        move_files_with_prefix(root_src, root_dest, album_name, prefix)