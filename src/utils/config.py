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

import os
from dotenv import load_dotenv
from typing import Union
from json import dumps as json_dumps, load as json_load

def load_json(json_path: str) -> dict:
    with open(json_path, "r") as f:
        return json_load(f)
    
def save_json(data: Union[dict, list], json_path: str, indent: int = 2) -> None:
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        f.write(json_dumps(data, indent=indent))

load_dotenv(override=True)

CONFIG_PATH = os.getenv("CONFIG_PATH")
MODELS_PATH = os.getenv("MODELS_PATH")
DEBUG_PATH = os.getenv("DEBUG_PATH")
SRC_PATH = os.getenv("PYTHON_PATH")
CACHE_PATH = os.getenv("CACHE_PATH")

FFMPEG_PATH = os.getenv("FFMPEG_PATH")
DATASOURCE_PATH = os.getenv("DATASOURCE_PATH")
DATASET_PATH = os.getenv("DATASET_PATH")
LATENTS_DATASET_PATH = os.getenv("LATENTS_DATASET_PATH")