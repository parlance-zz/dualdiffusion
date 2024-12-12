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
root_source_folder = f'/mnt/vault'
root_dest_folder = f'/mnt/vault'
zip_file_extensions = [".zip", ".7z", ".rar", ".tar", ".tar.gz", ".tar.bz2"]
delete_archives_that_fail_to_extract = True


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
                if os.path.isdir(subfolder_path) == True:
                    if len(os.listdir(subfolder_path)) > 0: # only skip if folder is not empty                    
                        #print(f"Skipping {filename} as {subfolder_path} already exists")
                        skipped_count += 1
                        continue

                os.makedirs(subfolder_path, exist_ok=True)

                print(f'Extracting {filename} to {subfolder_path}')
                try:
                    with py7zr.SevenZipFile(file_path, mode='r') as archive:
                        archive.extractall(path=subfolder_path)
                    extracted_count += 1
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
