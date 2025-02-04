from difflib import get_close_matches, SequenceMatcher
import os
import shutil
import webbrowser


root_path = "/mnt/vault/dataset"
cutoff = 0.73

def get_folders(path):
    """Retrieve the list of folders in the given path."""
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

def get_folder_size(path):
    total_size = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file(follow_symlinks=False):
                total_size += entry.stat().st_size
            elif entry.is_dir(follow_symlinks=False):
                total_size += get_folder_size(entry.path)
    return total_size

def get_similarity_score(a, b):
    return SequenceMatcher(None, a, b).ratio()

def decide(full_path_a, full_path_b):
    while True:
        _input = input(f"Keep (a/b)?: ").lower()
        if _input == "a":
            #shutil.rmtree(full_path_b)
            print(f"Deleted {full_path_b}")
            return
        elif _input == "b":
            #shutil.rmtree(full_path_a)
            print(f"Deleted {full_path_a}")
            return
        elif _input == "c":
            webbrowser.open(full_path_a)
            webbrowser.open(full_path_b)
        elif _input == "":
            return
            
def compare_folders(path_a, path_b, root_path, cutoff):
    
    folders_a = get_folders(os.path.join(root_path, path_a))
    folders_b = get_folders(os.path.join(root_path, path_b))

    for folder in folders_a:
        if folder not in folders_b:
            closest_matches = get_close_matches(folder, folders_b, n=1, cutoff=cutoff)
            if not closest_matches: continue
            score = get_similarity_score(folder, closest_matches[0])

            print(f"\nPotential duplicates (score: {score:.4f}):")
            full_path_a = os.path.join(root_path, path_a, folder)
            full_path_b = os.path.join(root_path, path_b, closest_matches[0])
            print(f"(a) {path_a}  {(get_folder_size(full_path_a)/1024/1024):.2f}mb  '{folder}'")
            print(f"(b) {path_b}  {(get_folder_size(full_path_b)/1024/1024):.2f}mb  '{closest_matches[0]}'")
            decide(full_path_a, full_path_a)
        else:
            print(f"\nDuplicate:\n {folder}")
            full_path_a = os.path.join(root_path, path_a, folder)
            full_path_b = os.path.join(root_path, path_b, folder)
            print(f"(a) {path_a}  {(get_folder_size(full_path_a)/1024/1024):.2f}mb")
            print(f"(b) {path_b}  {(get_folder_size(full_path_b)/1024/1024):.2f}mb")
            decide(full_path_a, full_path_a)


if __name__ == "__main__":

    all_folders = get_folders(root_path)
    
    for i in range(len(all_folders)):
        for j in range(i+1, len(all_folders)):
            compare_folders(all_folders[i], all_folders[j], root_path, cutoff=cutoff)