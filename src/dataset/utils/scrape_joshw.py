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

import requests
import re
import urllib.parse
import urllib.request
import html
import os
import time
import shutil

system = "3sf"
pages = ["0-9", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
base_url = f"https://{system}.joshw.info"
download_link_pattern = re.compile(r'href\s*=\s*["\']?([^"\'>\s]+\.(7z|zip|rar))["\']?\s*>', re.IGNORECASE)
request_throttle_delay_seconds = 0.1
target_zip_dir = f"/mnt/vault/{system}/zip"
minimum_disk_space_mb = 25000

os.makedirs(target_zip_dir, exist_ok=True)

for page in pages:

    page_url = html.unescape(f"{base_url}/{page}")
    response = requests.get(page_url)

    if response.status_code == 200:
        time.sleep(request_throttle_delay_seconds) # throttling

        download_links = [match[0] for match in re.findall(download_link_pattern, response.text)]
        for link in download_links:
            
            full_link = html.unescape(f"{page_url}/{link}")
            zip_filename = urllib.parse.unquote(os.path.basename(full_link))

            zip_save_path = os.path.join(target_zip_dir, zip_filename)
            if os.path.isfile(zip_save_path):
                continue

            try:
                free_disk_space_mb = shutil.disk_usage(target_zip_dir).free / 1024 / 1024
                if free_disk_space_mb < minimum_disk_space_mb:
                    print(f"Minimum disk space threshold reached ({free_disk_space_mb:.1f} MB), aborting...")
                    exit(1)

                print(f"Downloading: {full_link}")
                urllib.request.urlretrieve(full_link, zip_save_path)
                time.sleep(request_throttle_delay_seconds) # throttling
            except Exception as e:
                print(f"Failed to download {zip_filename}: {e}")
                continue
            

    else:
        print(f"Failed to retrieve page '{page}'. Status code: {response.status_code}")

print("Download process completed.")

