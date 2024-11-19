import os
import py7zr

def extract_7z_files(source_folder, target_folder):
    # Ensure the target folder exists
    os.makedirs(target_folder, exist_ok=True)

    # Iterate over each file in the source folder
    for filename in os.listdir(source_folder):
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext == '.7z':
            # Full path to the 7z file
            file_path = os.path.join(source_folder, filename)

            # Create a subfolder based on the filename without the .7z extension
            subfolder_name = os.path.splitext(filename)[0]
            # remove everything after the first ( or [ in subfolder_name
            subfolder_name = subfolder_name.split('(')[0]
            subfolder_name = subfolder_name.split('[')[0]
            subfolder_name = subfolder_name.replace("&amp;", "&").strip()
            # if the subfolder_name has any dots at the end, remove them
            while subfolder_name[-1] == '.':
                subfolder_name = subfolder_name[:-1]

            subfolder_path = os.path.join(target_folder, subfolder_name)
            if os.path.isdir(subfolder_path):
                print(f"Skipping {filename} as {subfolder_path} already exists")
                continue

            os.makedirs(subfolder_path, exist_ok=True)

            # Extract the 7z file into the subfolder
            print(f'Extracting {filename} to {subfolder_path}')
            with py7zr.SevenZipFile(file_path, mode='r') as archive:
                archive.extractall(path=subfolder_path)

# Define your source and target folders
source_folder = 'Z:/dualdiffusion/dataset/psf/zip'
target_folder = 'D:/dualdiffusion/dataset/psf/to_transcode'

# Run the extraction
extract_7z_files(source_folder, target_folder)
