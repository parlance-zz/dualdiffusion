import os
import py7zr

system = 'psf2'
source_folder = f'Y:/{system}/zip'
target_folder = f'Y:/{system}/to_transcode'

error_count = 0

def extract_7z_files(source_folder, target_folder):
    
    os.makedirs(target_folder, exist_ok=True)
    error_count = 0
    extracted_count = 0

    for filename in os.listdir(source_folder):
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in ['.7z', '.rar', '.zip']:

            file_path = os.path.join(source_folder, filename)
            subfolder_name = os.path.splitext(filename)[0]
            
            # sanitize the subfolder name
            subfolder_name = subfolder_name.split('(')[0]
            subfolder_name = subfolder_name.split('[')[0]
            subfolder_name = subfolder_name.replace("&amp;", "&").strip()
            while subfolder_name[-1] == '.':
                subfolder_name = subfolder_name[:-1]

            subfolder_path = os.path.join(target_folder, subfolder_name)
            if os.path.isdir(subfolder_path) == True:
                if len(os.listdir(subfolder_path)) > 0: # only skip if folder is not empty                    
                    #print(f"Skipping {filename} as {subfolder_path} already exists")
                    continue

            os.makedirs(subfolder_path, exist_ok=True)

            print(f'Extracting {filename} to {subfolder_path}')
            try:
                with py7zr.SevenZipFile(file_path, mode='r') as archive:
                    archive.extractall(path=subfolder_path)
                extracted_count += 1
            except Exception as e:
                print(f'Error extracting {filename}, skipping: {e}')
                error_count += 1

        else:
            print(f"Error: {filename} as it is not a known archive file type")
            error_count += 1

    print(f"Extraction complete (extracted: {extracted_count}) (errors: {error_count})")

if __name__ == "__main__":
    extract_7z_files(source_folder, target_folder)
