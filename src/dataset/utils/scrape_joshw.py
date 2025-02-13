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

from html.parser import HTMLParser
import requests
import urllib.parse
import urllib.request
import html
import os
import time
import shutil


systems = [
    "2sf", "3do", "3sf", "dsf", "gcn", "hes", "psf", "psf2",
    "smd", "spc", "ssf", "usf", "wii", "wiiu", "psf5", "x360",
    "psf3", "xbox", "psp", "vita", "switch", "psf4", "pc"
]
pages = ["0-9", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
allowed_file_extensions = [".zip", ".7z", ".rar", ".tar", ".tar.gz", ".tar.bz2"]
root_downloads_dir = "/mnt/vault/dataset_import"
minimum_disk_space_mb = 25000 # None
request_throttle_delay_seconds = 0.25


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "a":
            for attr, value in attrs:
                if attr.lower() == "href":
                    self.hrefs.append(value)

    def get_links(self, content: str, allowed_extensions: list[str]) -> list[str]:
        self.hrefs.clear()
        self.feed(content)
        return [link for link in self.hrefs
            if any(link.lower().endswith(ext) for ext in allowed_extensions)]

errors = []
num_downloads_found = 0
num_downloaded = 0
num_skipped = 0
link_parser = LinkParser()

for system in systems:
    
    base_url = f"https://{system}.joshw.info"
    target_zip_dir = os.path.join(root_downloads_dir, system, "zip")
    os.makedirs(target_zip_dir, exist_ok=True)
    print(f"\nDownloading files for {system}...\n")

    for page in pages:

        page_url = html.unescape(f"{base_url}/{page}")
        response = requests.get(page_url)

        if response.status_code == 200:
            time.sleep(request_throttle_delay_seconds)

            download_links = link_parser.get_links(response.text, allowed_file_extensions)
            num_downloads_found += len(download_links)

            for link in download_links:
                
                full_link = html.unescape(f"{page_url}/{link}")
                zip_filename = urllib.parse.unquote(os.path.basename(full_link))

                zip_save_path = os.path.join(target_zip_dir, zip_filename)
                if os.path.isfile(zip_save_path):
                    #print(f"Skipping {zip_filename} as it already exists")
                    num_skipped += 1
                    continue

                try:
                    if minimum_disk_space_mb is not None:
                        free_disk_space_mb = shutil.disk_usage(target_zip_dir).free / 1024 / 1024
                        if free_disk_space_mb < minimum_disk_space_mb:
                            print(f"Minimum disk space threshold reached ({free_disk_space_mb:.1f} MB), aborting...")
                            exit(1)

                    print(f"Downloading {full_link}")
                    urllib.request.urlretrieve(full_link, zip_save_path)
                    time.sleep(request_throttle_delay_seconds)
                    num_downloaded += 1

                except Exception as e:
                    error_txt = f"Error downloading {full_link} to {zip_save_path}: {e}"
                    print(error_txt)
                    errors.append(error_txt)
                
        else:
            error_txt = f"Error loading '{page_url}'. Status code: {response.status_code}"
            print(error_txt)
            errors.append(error_txt)

print(f"\nDownload process completed.")
print(f"  num downloads found: {num_downloads_found}")
print(f"  num downloaded: {num_downloaded}")
print(f"  num skipped: {num_skipped}")
print(f"  num errors: {len(errors)}\n")
if len(errors) > 0:
    print("Errors:")
    for error in errors: print(error)