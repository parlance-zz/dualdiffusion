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
import os
import time

pages = ["0-9", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
base_url = "https://usf.joshw.info"
download_link_pattern = re.compile(r'href="([^"]+\.7z)">')
request_throttle_delay_seconds = 0.1
target_zip_dir = "d:/dualdiffusion/dataset/usf/zip"

os.makedirs(target_zip_dir, exist_ok=True)

for page in pages:

    page_url = f"{base_url}/{page}"
    response = requests.get(page_url)

    if response.status_code == 200:
        time.sleep(request_throttle_delay_seconds) # throttling

        download_links = re.findall(download_link_pattern, response.text)
        for link in download_links:
            
            full_link = f"{page_url}/{link}"
            zip_filename = urllib.parse.unquote(os.path.basename(full_link))

            zip_save_path = os.path.join(target_zip_dir, zip_filename)
            if os.path.isfile(zip_save_path):
                continue

            try:
                print(f"Downloading: {full_link}")
                urllib.request.urlretrieve(full_link, zip_save_path)
                time.sleep(request_throttle_delay_seconds) # throttling
            except Exception as e:
                print(f"Failed to download {zip_filename}: {e}")
                continue

    else:
        print(f"Failed to retrieve page '{page}'. Status code: {response.status_code}")

print("Download process completed.")

