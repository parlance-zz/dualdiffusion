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

import os

import py7zr


systems = ["2sf", "3do", "3sf", "dsf", "gcn", "hes", "psf", "psf2", "smd", "spc", "ssf", "usf", "wii", "wiiu"]
root_source_folder = f'/mnt/vault/dataset_import'
root_dest_folder = f'/mnt/vault/dataset_import'
zip_file_extensions = [".zip", ".7z", ".rar", ".tar", ".tar.gz", ".tar.bz2"]
delete_archives_that_fail_to_extract = False

def get_directory_size(path: str) -> int:
    total_size = 0
    with os.scandir(path) as it:
        for entry in it:
            try:
                if entry.is_file(follow_symlinks=False):
                    total_size += entry.stat().st_size
                elif entry.is_dir(follow_symlinks=False):
                    total_size += get_directory_size(entry.path)  # recurse for subdirectories
            except PermissionError:
                pass  # skip files/directories without permission
    return total_size

if __name__ == "__main__":

    if delete_archives_that_fail_to_extract == True:
        print("WARNING: Archives that fail to extract will be deleted")
        if input("Continue? (y/n): ").lower() != 'y':
            exit()

    errors = []
    extracted_count = 0
    skipped_count = 0
    
    for system in systems:

        source_folder = os.path.join(root_source_folder, system, "zip")
        target_folder = os.path.join(root_dest_folder, system, "to_transcode")

        print(f"\nExtracting files from {source_folder} to {target_folder} ...\n")
        os.makedirs(target_folder, exist_ok=True)

        for filename in os.listdir(source_folder):
            if os.path.splitext(filename)[1].lower() in zip_file_extensions:

                file_path = os.path.join(source_folder, filename)
                subfolder_name = os.path.splitext(filename)[0]
                
                # sanitize the subfolder name
                subfolder_name = subfolder_name.split('(')[0]
                subfolder_name = subfolder_name.split('[')[0]
                subfolder_name = subfolder_name.replace("&amp;", "&").strip()
                while subfolder_name[-1] == '.':
                    subfolder_name = subfolder_name[:-1]
                subfolder_path = os.path.join(target_folder, subfolder_name)

                # if extract target path exists, check if the sizes match. if mismatch extract anyway
                if os.path.isdir(subfolder_path) == True:
                    try:
                        with py7zr.SevenZipFile(file_path, mode='r') as archive:
                            total_uncompressed_size = 0
                            for file in archive.list():
                                total_uncompressed_size += file.uncompressed
                        
                        existing_subfolder_size = get_directory_size(subfolder_path)
                        if (total_uncompressed_size // 1024) == (existing_subfolder_size // 1024):
                            skipped_count += 1
                            if skipped_count % 1000 == 0:
                                print(f"Skipped {skipped_count} so far")
                            continue

                    except Exception as e:
                        error_txt = f'Error verifying size for {filename}: {e}'
                        print(error_txt)
                        errors.append(error_txt)
                        continue       
                    
                os.makedirs(subfolder_path, exist_ok=True)
                print(f'Extracting {filename} to {subfolder_path}')
                try:
                    with py7zr.SevenZipFile(file_path, mode='r') as archive:
                        archive.extractall(path=subfolder_path)
                    extracted_count += 1
                    if extracted_count % 1000 == 0:
                        print(f"Extracted {extracted_count} so far")
                except Exception as e:
                    error_txt = f'Error extracting {filename}: {e}'
                    if delete_archives_that_fail_to_extract == True:
                        os.remove(file_path)
                        error_txt += " (archive was deleted)"
                    print(error_txt)
                    errors.append(error_txt)

            else:
                error_txt = f"Error: {filename} is not in zip file extension list"
                print(error_txt)
                errors.append(error_txt)

    print(f"\nExtraction complete.")
    print(f"  num extracted: {extracted_count}")
    print(f"  num skipped: {skipped_count}")
    print(f"  num errors: {len(errors)}\n")

    if len(errors) > 0:
        print("Errors:")
        for error in errors: print(error)
