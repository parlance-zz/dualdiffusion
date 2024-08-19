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
import urllib.request
import os
import time
import zipfile

base_game_url = "https://www.zophar.net"
base_page_url = "https://www.zophar.net/music/playstation-psf?page="
game_page_pattern = re.compile(r'href=["\'](/music/playstation-psf/[^"\']*)["\']')
zip_link_pattern = re.compile(r'href=["\'](https://[^"\']*MP3[^"\']*\.zip)["\']')
request_throttle_delay_seconds = 0.1
target_zip_dir = "./dataset/psf/zip"
target_spc_dir = "./dataset/psf"
start_page = 1
end_page = 7

# re-extract zip files
"""
for dirpath, _, filenames in os.walk(target_zip_dir):
    for filename in filenames:
        zip_filename = os.path.join(dirpath, filename)
        try:
            game_name = os.path.splitext(filename)[0]
            game_dir = os.path.join(target_spc_dir, game_name)
            if not os.path.exists(game_dir):
                os.makedirs(game_dir)
            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                zip_ref.extractall(game_dir)
            print(f"Extracted '{zip_filename}' to '{game_dir}'")
        except Exception as e:
            print(f"Failed to extract contents of '{zip_filename}': {e}")     
exit()
"""

os.makedirs(target_zip_dir, exist_ok=True)

for page_number in range(start_page, end_page + 1):
    
    page_url = f"{base_page_url}{page_number}"
    response = requests.get(page_url)

    if response.status_code == 200:
        
        game_pages = re.findall(game_page_pattern, response.text)
        for game_page in game_pages:
            
            time.sleep(request_throttle_delay_seconds) # throttling
            
            game_page_url = f"{base_game_url}{game_page}"
            response = requests.get(game_page_url)

            if response.status_code == 200:

                zip_links = re.findall(zip_link_pattern, response.text)
                for zip_url in zip_links:
                    
                    zip_filename = urllib.parse.unquote(os.path.basename(zip_url))
                    zip_filename = zip_filename.replace(".zophar", "").replace(" (EMU)", "").replace("(EMU)", "").replace(" (MP3)", "").replace("(MP3)", "")
                    zip_save_path = os.path.join(target_zip_dir, zip_filename)

                    try:
                        print(f"Downloading: {zip_filename}")
                        urllib.request.urlretrieve(zip_url, zip_save_path)
                    except Exception as e:
                        print(f"Failed to download {zip_filename}: {e}")
                        continue

                    try:
                        game_name = os.path.splitext(zip_filename)[0]
                        game_dir = os.path.join(target_spc_dir, game_name)
                        os.makedirs(game_dir, exist_ok=True)
                        with zipfile.ZipFile(zip_save_path, 'r') as zip_ref:
                            zip_ref.extractall(game_dir)
                        print(f"Extracted '{zip_save_path}' to '{game_dir}'")
                    except Exception as e:
                        print(f"Failed to extract contents of '{zip_save_path}': {e}")
                        
            else:
                print(f"Failed to retrieve page {game_page_url}. Status code: {response.status_code}")

    else:
        print(f"Failed to retrieve page {page_number}. Status code: {response.status_code}")

print("Download process completed.")



