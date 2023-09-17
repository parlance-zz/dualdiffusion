import requests
import re
import urllib.request
import os
import time
import zipfile

base_game_url = "https://www.zophar.net"
base_page_url = "https://www.zophar.net/music/nintendo-snes-spc?page="
game_page_pattern = re.compile(r'href=["\'](/music/nintendo-snes-spc/[^"\']*)["\']')
zip_link_pattern = re.compile(r'href=["\'](https://[^"\']*EMU[^"\']*\.zip)["\']')
request_throttle_delay_seconds = 1
target_zip_dir = "./dataset/spc/zip"
target_spc_dir = "./dataset/spc"
start_page = 1
end_page = 9

if not os.path.exists(target_zip_dir):
    os.makedirs(target_zip_dir)

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
                    zip_filename = zip_filename.replace(".zophar", "").replace(" (EMU)", "").replace("(EMU)", "")
                    zip_save_path = os.path.join(target_zip_dir, zip_filename)

                    try:
                        urllib.request.urlretrieve(zip_url, zip_save_path)
                        print(f"Downloaded: {zip_filename}")
                    except Exception as e:
                        print(f"Failed to download {zip_filename}: {e}")
                        continue

                    try:
                        game_name = os.path.splitext(zip_filename)[0]
                        game_dir = os.path.join(target_spc_dir, game_name)
                        if not os.path.exists(game_dir):
                            os.makedirs(game_dir)
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



