from dataset.dataset_processor import find_files

from collections import defaultdict
import os

def count_file_types(root_path, good_file_types, delete=False):

    good_file_types = good_file_types or []
    file_types = defaultdict(int)

    all_files = find_files(root_path)
    for file in all_files:
        if not os.path.isfile(file):
            continue

        file_extension = os.path.splitext(file)[1].lower()
        if file_extension in good_file_types: continue
        
        if delete == True:
            print(file)
            os.remove(file)

        file_types[file_extension] += 1
            
    sorted_file_types = sorted(file_types.items(), key=lambda item: item[1])
    total_bad_files = 0
    for file_type, count in sorted_file_types:
        print(f"{file_type or '[no extension]'}: {count}")
        total_bad_files += count
    print("")
    print(f"Found {total_bad_files}/{len(all_files)} bad files")


good_file_types = [
    # misc
    ".dsf",
    ".ape",
    ".minissf",
    ".ssf",
    ".flac",
    ".wav",
    ".vpk",
    ".xwb",
    ".xag",
    ".vig",
    ".vgs",
    ".vas",
    ".txtp",
    ".mus",
    ".jstm",
    ".tak",
    ".svag",
    ".str",
    ".stream",
    ".sng",
    ".snd",
    ".rws",
    ".rstm",
    ".rsd",
    ".pcm",
    ".ogg",
    ".npsf",
    ".musx",
    ".musc",
    ".mp3",
    ".int",
    ".hxd",
    ".gms",
    ".bik",
    ".aus",
    ".asf",
    ".ads",

    # psf2
    ".adpcm",
    ".minipsf",
    ".minipsf2",
    ".psf",
    ".psf2",
    ".psflib",
    ".psf2lib",
    ".ss2",
    ".adx",
    ".genh",
    ".mib",
    ".mic",
    ".vag",
    ".vgmstream",

    # psf3
    ".msf",
    ".at3",
    ".wem",
    ".fsb",
    ".xvag",
    ".hca",
    ".sps",
    ".aa3",
    ".lwav",
    ".aax",
    ".nub",
    ".awc",
    ".snu",
    ".sgb",
    ".ac3",
    ".nub2",
    
    # psf4
    ".lopus",
    ".bfstm",
    ".ktss",
    ".bwav",
    ".kno",
    ".m4a",
    ".acm",
    ".kns",
    ".bfwav",
    ".sab",
    ".dsp",
    ".rak",
    ".logg",
    ".ckd",
    ".mab",
    ".opus",
    ".mp4",
    ".webm",
    ".wma",
    
    #xbox
    ".wavm",
    ".xwav",
    ".rwx",

    #psp
    ".oma",
    ".sgd",

    #vita
    ".at9",

]

#root_path = "/mnt/vault/dataset_import/psf4/to_transcode"
#root_path = "/mnt/vault/dataset_import/pc/to_transcode"
#root_path = "/mnt/vault/dataset_import/psf5/to_transcode"
#root_path = "/mnt/vault/dataset_import/x360/to_transcode"

root_path = "/mnt/vault/dataset_import/vita/to_transcode"
delete = True

if delete == True:
    print(f"WARNING: This will delete all files that do not match known good file types")
    prompt = "Overwrite existing model? (y/n): "
    if input("Continue? (y/n):").lower() not in ["y","yes"]: exit()

count_file_types(root_path, good_file_types, delete=delete)
