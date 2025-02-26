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

from typing import Optional
import os
import difflib
import fnmatch


def get_files_in_dir(directory: str, ignore_patterns: Optional[list[str]] = None,
                     whitelist_patterns: Optional[list[str]] = None) -> list[str]:
    
    """Get all file paths in a directory recursively, filtering out ignored patterns."""
    ignore_patterns = ignore_patterns or []
    whitelist_patterns = whitelist_patterns or []
        
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            relative_path = os.path.relpath(os.path.join(root, file), directory)
            if not any(fnmatch.fnmatch(relative_path, pattern) for pattern in ignore_patterns):
                if whitelist_patterns:
                    if any(fnmatch.fnmatch(relative_path, pattern) for pattern in whitelist_patterns):
                        match = True
                    else:
                        match = False
                else:
                    match = True

                if match == True:
                    file_paths.append(relative_path)

    return sorted(file_paths)

def compare_files(file1: str, file2: str) -> list[str]:
    """Compare two files and return the diff."""
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        file1_lines = f1.readlines()
        file2_lines = f2.readlines()
        diff = list(difflib.unified_diff(file1_lines, file2_lines, lineterm='\n'))

    return diff

def compare_dirs(output_path: str, dir1: str, dir2: str,
                ignore_patterns: Optional[list[str]] = None,
                whitelist_patterns: Optional[list[str]] = None) -> str:
    
    """Compare two directories and generate a diff-like output."""
    files1 = get_files_in_dir(dir1, ignore_patterns, whitelist_patterns)
    files2 = get_files_in_dir(dir2, ignore_patterns, whitelist_patterns)

    added_files = [f for f in files2 if f not in files1]
    removed_files = [f for f in files1 if f not in files2]
    common_files = [f for f in files1 if f in files2]

    diff_output = []

    for file in added_files:
        diff_output.append(f"Added: {file}")
    for file in removed_files:
        diff_output.append(f"Removed: {file}")

    for file in common_files:
        file1_path = os.path.join(dir1, file)
        file2_path = os.path.join(dir2, file)
        diff = compare_files(file1_path, file2_path)
        if diff:
            diff_output.append(f"\nDiff for {file}:\n" + ''.join(diff))

    if diff_output:
        diff_output = "\n".join(diff_output)
    else:
        diff_output = "No differences found."
    
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as output_file:
            output_file.write(diff_output)

    return diff_output

if __name__ == "__main__":
    diff_output = compare_dirs(
        "/home/parlance/dualdiffusion/debug/compare_dirs/output.diff",
        "/home/parlance/dualdiffusion/src",
        "/home/parlance/dualdiffusion/models/edm2_ddec_mclt4/ddec_checkpoint-19557/src",
        ignore_patterns=["*.pyc"])
    
    print(diff_output)
    