from utils.dual_diffusion_utils import get_no_clobber_filepath

import os
import shutil

# this script moves files with a specified prefix and extension recursively from root_src to root_dest
# also removes prefix from the moved files
root_src = 'y:/spc/to_transcode'
root_dest = 'y:/spc/transcoded'
prefix = 't_'      # filename prefix for files to move
file_ext = '.flac' # file extension for files to move
no_clobber = True  # renames target filename to not overwrite existing files
remove_before_2_underscores = True # remove everything before the 2nd underscore in the target filename instead of the prefix
verbose = True     # print all moved files

def move_files_with_prefix(root_src, root_dest, album_name, prefix,
                           file_ext, no_clobber, remove_before_2_underscores, verbose):
    src_path = os.path.join(root_src, album_name)
    dest_path = os.path.join(root_dest, album_name)
    created_album_dir = False

    for subdir, _, files in os.walk(src_path):
        for file in files:
            if prefix is not None and (not file.startswith(prefix)): continue
            if file_ext is not None and (os.path.splitext(file)[1].lower() != file_ext): continue
            
            src_file_path = os.path.join(subdir, file)
            if remove_before_2_underscores == True and file.count('_') >= 2:
                file = '_'.join(file.split('_', 2)[2:])
            elif prefix is not None:
                file = file[len(prefix):]
            dest_file_path = os.path.join(dest_path, file)
            
            if not created_album_dir:
                os.makedirs(dest_path, exist_ok=True)
            if no_clobber == True:
                dest_file_path = get_no_clobber_filepath(dest_file_path)
            shutil.move(src_file_path, dest_file_path)
            if verbose == True:
                print(f"Moved {src_file_path} to {dest_file_path}")


if __name__ == '__main__':

    print(f"This will move all files from '{root_src}' with prefix '{prefix}' and extension '{file_ext}' to '{root_dest}'")
    if input("Are you sure you want to continue? (y/n): ").lower() != 'y': exit()
    
    for album_name in os.listdir(root_src):
        if os.path.isdir(os.path.join(root_src, album_name)):
            move_files_with_prefix(root_src, root_dest, album_name,
                prefix, file_ext, no_clobber, remove_before_2_underscores, verbose)