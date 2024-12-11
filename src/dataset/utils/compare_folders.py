from difflib import get_close_matches
import os
import shutil
import webbrowser


# this script is useful for merging 2 large scrapes of the same content
# it will compare folders under path a to folders under path b, and if no match is found you can copy it to path c
path_a = "Y:\\spc\\old_zophars_to_transcode"
path_b = "Y:\\spc\\to_transcode"
path_c = "Y:\\spc\\new_to_transcode_pre_spc_fix_backup"

def get_folders(path):
    """Retrieve the list of folders in the given path."""
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

def compare_folders(path_a, path_b, path_c):
    """Compare folders in path A to those in path B."""
    folders_a = get_folders(path_a)
    folders_b = get_folders(path_b)

    for folder in folders_a:
        if folder not in folders_b:
            closest_matches = get_close_matches(folder, folders_b, n=1)
            closest_match = closest_matches[0] if closest_matches else "No close match found"
            print(f"Folder '{folder}' is not in {path_b}. Closest match: {closest_match}")

            while True:
                _input = input(f"Do you want to copy '{folder}' to '{path_c}'? (y/n/c): ").lower()
                if _input == "y":
                    src_path = os.path.join(path_a, folder)
                    dest_path = os.path.join(path_c, folder)
                    shutil.copytree(src_path, dest_path)
                    print(f"Copied '{src_path}' to {dest_path}")
                    break
                elif _input == "n":
                    break
                elif _input == "c":
                    webbrowser.open(os.path.join(path_a, folder))

def main(path_a, path_b, path_c):

    if not os.path.exists(path_a) or not os.path.exists(path_b):
        print("One or both paths do not exist. Please provide valid paths.")
        return

    os.makedirs(path_c, exist_ok=True)
    compare_folders(path_a, path_b, path_c)

if __name__ == "__main__":
    main(path_a, path_b, path_c)
