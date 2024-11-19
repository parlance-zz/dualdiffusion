import os
from collections import defaultdict

def count_file_types(root_path):
    file_types = defaultdict(int)

    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            file_extension = os.path.splitext(filename)[1].lower()  # Get file extension in lowercase
            file_types[file_extension] += 1  # Count each file type

    # Display the result
    print("File Type Counts:")
    for file_type, count in sorted(file_types.items()):
        print(f"{file_type or '[no extension]'}: {count}")

# Example usage:
root_path = "D:/dualdiffusion/dataset/psf/to_transcode"  # Replace with the desired path
count_file_types(root_path)
