import os
import time

folders_to_monitor = [
    "./dataset/2sf/zip",
    "./dataset/dsf/zip",
    "./dataset/gsf/zip",
    "./dataset/usf/zip",
    "./dataset/ssf/zip",
    "./dataset/psf/zip",
]

def delete_old_files(folder):

    files = sorted(
        (os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))),
        key=os.path.getmtime,
        reverse=True
    )

    for file_to_delete in files[1:]:
        os.remove(file_to_delete)
        print(f"Deleted: {file_to_delete}")
    #print(folder, files[0])

def monitor_folders(folders):
    while True:
        for folder in folders:
            if os.path.exists(folder):
                try:
                    delete_old_files(folder)
                except Exception as e:
                    print(f"Failed to delete files in '{folder}': {e}")
        time.sleep(60) 

if __name__ == "__main__":
    monitor_folders(folders_to_monitor)