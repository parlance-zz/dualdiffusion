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
from types import GenericAlias
from typing import Union, Type, Optional, Any, get_args, get_origin
from dataclasses import fields, is_dataclass
from json import dumps as json_dumps, load as json_load, loads as json_loads
from logging import getLogger

from pyjson5 import load as json5_load


def load_json(json_path: str) -> dict:
    if os.path.splitext(json_path)[1].lower() == ".jsonl":
        with open(json_path, "r") as f:
            return [json_loads(line) for line in f]
    
    with open(json_path, "r") as f:
        return json_load(f)
    
def save_json(data: Union[dict, list], json_path: str,
        indent: int = 2, copy_on_write: bool = False) -> None:
    
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    def write_fn(f, _data):
        if os.path.splitext(json_path)[1].lower() == ".jsonl":
            for i, item in enumerate(_data):
                if i == len(_data) - 1: f.write(json_dumps(item))
                else: f.write(json_dumps(item) + "\n")
        else:
            f.write(json_dumps(_data, indent=indent))
            
    if copy_on_write == True:
        tmp_path = f"{json_path}.tmp"
        try:
            with open(tmp_path, "w") as f:
                write_fn(f, data)

            os.rename(tmp_path, json_path)
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)

        except Exception as e:
            try:
                if os.path.isfile(tmp_path):
                    os.remove(tmp_path)
            except: pass
            raise e
    else:
        with open(json_path, "w") as f:
            write_fn(f, data)

# if type is Optional[T], return T, otherwise, return type
def _unwrap_optional(_type: Type) -> Type:
    
    if get_origin(_type) is Union:
        args = get_args(_type)

        # type is optional if it is a Union between any non-none type(s) and a None type
        non_none_types = [t for t in args if t is not type(None)]
        if len(non_none_types) == 1:
            return non_none_types[0]  
    return _type

def load_config(config_class: Type, path: str, data: Optional[dict] = None, quiet: bool = False) -> Any:

    # config classes are required to be dataclasses
    if is_dataclass(config_class) == False:
        raise ValueError(f"Error: Unable to load config '{path}', "
                         f"'{config_class.__name__}' is not a dataclass")
    
    # load from file using json5 if data is None
    if data is None:
        try:
            with open(path, "r") as f:
                data: dict = json5_load(f)
        except Exception as e:
            raise ValueError(f"Error: Unable to load config '{path}', '{e}'")

    elif isinstance(data, dict) == False: # data must be a dict if specified
        raise ValueError(f"Error: Unable to load config '{path}', "
                         f"expected dict for type(data), got '{type(data).__name__}'")
    
    # if the specified config class has a _config_path attribute,
    # populate it when loading the config from a file path
    if hasattr(config_class, "_config_path"):
        data["_config_path"] = path
    
    # used for missing field/attribute warnings
    logger = getLogger() if quiet == False else None

    config_fields: dict[str, type] = {field.name: _unwrap_optional(field.type) for field in fields(config_class)}
    config_dict = {}

    # build dict to instantiate specified dataclass from data
    for field, value in data.items():
        if field not in config_fields:
            if logger is not None: # show warning and ignore if config data contains fields missing in the config class
                logger.warning(f"Warning: field '{field}' not found in config dataclass '{config_class.__name__}', ignoring...")
            continue
        
        if type(value) is type(None):
            config_dict[field] = None
            continue

        # recursively instantiate nested data classes based on their type annotation
        if is_dataclass(config_fields[field]):
            if isinstance(value, dict) == True:
                config_dict[field] = load_config(config_fields[field], path, value, quiet)
            else:
                raise ValueError(f"Error: Unable to load config, expected dict for type({field}), "
                                f"got '{type(value).__name__}'")

        # same for lists of dataclasses
        elif get_origin(config_fields[field]) is list and isinstance(config_fields[field], GenericAlias):

            elem_type = get_args(config_fields[field])[0]
            if is_dataclass(elem_type) and isinstance(value, list):
                config_dict[field] = [load_config(elem_type, path, v, quiet) for v in value]
            else:
                config_dict[field] = value

        # same for dicts of dataclasses
        elif get_origin(config_fields[field]) is dict and isinstance(config_fields[field], GenericAlias):
            
            elem_type = get_args(config_fields[field])[1]
            if is_dataclass(elem_type) and isinstance(value, dict):
                config_dict[field] = {k: load_config(elem_type, path, v, quiet) for k,v in value.items()}
            else:
                config_dict[field] = value

        else: # normal non-dataclass field
            config_dict[field] = value

    # finally instantiate the specified config dataclass
    config = config_class(**config_dict)

    # and show warning if any fields were missing and intialized with default values
    for field in config_fields:
        if field not in data and logger is not None:
            logger.warning(f"Warning: field '{field}' not found in config file '{path}',"
                            f" initializing with default value '{getattr(config, field)}'")
    
    return config

# recursively decomposes nested dataclasses into a json-friendly dict
def _jsonify(obj: Any, path: Optional[str] = None) -> dict:
    if is_dataclass(obj):
        return _jsonify(obj.__dict__, path)
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v, path) for v in obj]
    if isinstance(obj, dict):
        return {k: _jsonify(v, path) for k,v in obj.items()}
    
    raise ValueError(f"Error: Unsupported type '{type(obj).__name__}' in save_config '{path}'")

def save_config(config: Any, path: str) -> None:
    save_json(_jsonify(config, path), path, copy_on_write=True)

load_dotenv(override=True)

CONFIG_PATH = os.getenv("CONFIG_PATH")
MODELS_PATH = os.getenv("MODELS_PATH")
DEBUG_PATH = os.getenv("DEBUG_PATH")
SRC_PATH = os.getenv("PYTHONPATH")
CACHE_PATH = os.getenv("CACHE_PATH")

FFMPEG_PATH = os.getenv("FFMPEG_PATH")
DATASET_PATH = os.getenv("DATASET_PATH")

CLAP_MODEL_PATH = os.getenv("CLAP_MODEL_PATH")

NO_GUI = int(os.getenv("NO_GUI") or 0) == 1