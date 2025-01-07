from collections import defaultdict
import os
import shutil

def count_file_types(root_path, examples_path, good_file_types, bad_file_types):

    good_file_types = good_file_types or []
    bad_file_types = bad_file_types or []

    examples = defaultdict(list)
    file_types = defaultdict(int)

    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            file_extension = os.path.splitext(filename)[1].lower()  # Get file extension in lowercase
            if file_extension in bad_file_types: continue
            if file_extension in good_file_types: continue
            
            file_types[file_extension] += 1  # Count each file type
            if file_types[file_extension] < 10 and file_extension.startswith(".mini") == False:
                examples[file_extension] += [os.path.join(dirpath, filename)]

    # Display the result
    total_files = 0
    print("File Type Counts:")
    for file_type, count in sorted(file_types.items()):
        print(f"{file_type or '[no extension]'}: {count}")
        total_files += count
    print("")
    print(f"Total files: {total_files}")

    #sorted_file_types = sorted(file_types.items(), key=lambda item: item[1])
    #sorted_file_types = {k: v for k, v in sorted_file_types}
    #for file_type, count in sorted(sorted_file_types.items()):
    #    print(f"{file_type or '[no extension]'}: {count}")
    #print("")

    if examples_path is not None:
        os.makedirs(examples_path, exist_ok=True)
        for file_type, example_list in examples.items():
            for example_path in example_list:
                dest_path = os.path.join(examples_path, f"{file_type[1:]}_{os.path.basename(example_path)}")
                if not os.path.isfile(dest_path):
                    shutil.copy2(example_path, dest_path)


root_path = "/mnt/vault/dataset_import/smd/to_transcode"
examples_path = None#"/mnt/vault/dataset_import/smd/file_type_examples"

good_file_types = [".mp3", ".flac", ".wav", ".ape", ".minipsf", ".minissf", ".psf", ".ssf", ".dsf", ".nsf", ".spc"]
bad_file_types = [".7z", ".zip", ".rar", ".gz", ".bak", ".pak"
                  ".jpg", ".jpeg", ".png", ".bmp", ".gif"
                  ".c", "*.cpp", ".py", ".bat", ".vbs", ".ips"
                  ".log", ".ini", ".nfo", ".info", ".md", ".txt", ".txtp", ".txth",
                  ".bms", ".snd", ".wic", ".cpk", ".sam",
                  ".m3u", ".m3u8", ".fpl",
                  ""]
# maybe .snd, .mus, .str, .p16, .txtp, .vgmstream

count_file_types(root_path, examples_path, None, None)
