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

import utils.config as config

from dataclasses import dataclass
from typing import Generator, Optional, Union, Literal, Any
from abc import ABC, abstractmethod
from traceback import format_exception
from queue import Full as QueueFullException
from datetime import datetime
from multiprocessing.synchronize import Lock, Event
from copy import deepcopy
import os
import json
import logging
import shutil
import time
import signal
import threading

from tqdm.auto import tqdm
import torch
import mutagen
import torch.multiprocessing as mp
import safetensors.torch as safetensors

from utils.dual_diffusion_utils import (
    dict_str, normalize, save_safetensors, get_audio_metadata,
    init_logging, init_cuda, get_available_torch_devices
)


def _process_worker(stage: "DatasetProcessStage", rank: int,
        cuda_device: Optional[str], critical_lock: Lock, finish_event: Event) -> None:

    # disable sigint, it will be handled by the main process
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # init pytorch 
    if cuda_device is not None:
        assert stage.get_stage_type() == "cuda"
        init_cuda()
    
    # if using cpu fallback for a cuda stage then allow default torch threads, otherwise set to 1
    if cuda_device != "cpu":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    # init logging
    stage_name = stage.__class__.__name__
    logger = logging.getLogger(stage_name)
    log_handler = WorkerLogHandler(stage.log_queue, stage.warning_queue, stage.error_queue, stage_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(log_handler)

    # init process
    try:
        stage._init_process(rank, cuda_device, logger, critical_lock, finish_event)
        stage.start_process()
    except Exception as e:
        logger.error("".join(format_exception(type(e), e, e.__traceback__)))

        # flag worker as finished and wait forever
        stage.finish_event.set()
        threading.Event().wait()
    
    while True:
        input_dict: dict = stage.input_queue.get()
        if input_dict is None:
            try:
                stage.finish_process()
            except Exception as e:
                logger.error("".join(format_exception(type(e), e, e.__traceback__)))

            # flag worker as finished and wait forever
            stage.finish_event.set()
            threading.Event().wait()

        try:
            cloned_input_dict = deepcopy(input_dict)
            del input_dict  # necessary to satisfy pytorch shared memory constraints
            output = stage.process(cloned_input_dict)

            if output is not None:
                if isinstance(output, list):
                    for item in output:
                        stage.output_queue.put(item)
                elif isinstance(output, dict):
                    stage.output_queue.put(output)
                else:
                    raise ValueError(f"stage.process returned unrecognized type '{type(output)}'")
            else:
                del cloned_input_dict
                
        except Exception as e:
            logger.error("".join(format_exception(type(e), e, e.__traceback__)))
            try: del input_dict
            except: pass

def _log_worker(process_name: str, verbose: bool, log_queue: mp.Queue):

    # disable sigint, it will be handled by the main process
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    logger = init_logging(process_name, group_name="dataset_processor", verbose=verbose)
    while True:
        record = log_queue.get()
        if record is None: os._exit(0)

        logger.handle(record)
        for handler in logger.handlers:
            handler.flush()

def _monitor_worker(input_queues: list["WorkQueue"], stage_names: list[str], finish_event: Event) -> None:

    # disable sigint, it will be handled by the main process
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    progress_bars = []
    for name in stage_names:
        progress_bar = tqdm(total=1, smoothing=0.9)
        progress_bar.set_description(name, refresh=False)
        progress_bars.append(progress_bar)

    def update_progress(progress_bar: tqdm, input_queue: WorkQueue) -> None:
        processed, total = input_queue.get_processed_total()
        progress_bar.total = total
        progress_bar.n = processed
        progress_bar.refresh()

    while True:
        for progress_bar, input_queue in zip(progress_bars, input_queues):
            update_progress(progress_bar, input_queue)
        
        if finish_event.is_set(): break
        time.sleep(0.5)

    for progress_bar, input_queue in zip(progress_bars, input_queues):
        update_progress(progress_bar, input_queue)
        progress_bar.close()

    os._exit(0)

def _terminate_worker(process: mp.Process, timeout: float = 0.25, close: bool = False) -> None:

    process.join(timeout=timeout)

    if process.is_alive() == True:
        process.terminate()
        process.join(timeout=timeout)

        if process.is_alive() == True:
            process.kill()
            process.join()

    if close == True:
        process.close()

class WorkQueue: # normal Queue class behavior extended with progress tracking

    def __init__(self, *args, **kwargs) -> None:
        self.total_count = mp.Value("i", 0)  # 'i' means integer
        self.processed_count = mp.Value("i", 0)
        self.queue = mp.Queue(*args, **kwargs)

    def put(self, obj, *args, **kwargs) -> None:
        if obj is not None:
            with self.total_count.get_lock():
                self.total_count.value += 1
        self.queue.put(obj, *args, **kwargs)

    def get(self, *args, **kwargs) -> Any:
        obj = self.queue.get(*args, **kwargs)
        if obj is not None:
            with self.processed_count.get_lock():
                self.processed_count.value += 1
        return obj

    def get_processed_total(self) -> tuple[int, int]:
        with self.total_count.get_lock():
            with self.processed_count.get_lock():
                total = self.total_count.value
                processed = self.processed_count.value
        return processed, total

# dummy log handler that sends all log records to an mp queue and saves warnings and errors in separate queues
class WorkerLogHandler(logging.Handler):
    def __init__(self, log_queue: mp.Queue, warning_queue: mp.Queue, error_queue: mp.Queue,
                 process_name: Optional[str] = None) -> None:
        
        super().__init__(level=logging.DEBUG)
        self.process_name = process_name or ""

        self.log_queue = log_queue
        self.warning_queue = warning_queue
        self.error_queue = error_queue

    def emit(self, record: Optional[logging.LogRecord]) -> None:
        if record is not None:
            if record.msg not in ("" , "\n"):
                if not hasattr(record, "label"):
                    setattr(record, "label", self.process_name)
                    record.msg = f"{record.label}: {record.msg}"

            if record.levelno >= logging.ERROR:
                self.error_queue.put(record)
            elif record.levelno >= logging.WARNING:
                self.warning_queue.put(record)

        self.log_queue.put(record)

# main workhorse class for dataset processing. processes should subclass DatasetProcessStage and
# implement process (required), and get_stage_type, start_process, finish_process, info_banner, summary_banner, limit_output_queue_size (optional)
class DatasetProcessStage(ABC):

    def __init__(self) -> None:
        pass

    def _start_worker_queue(self, input_queue: WorkQueue, log_queue: mp.Queue,
                num_proc: int, processor_config: "DatasetProcessorConfig") -> WorkQueue:

        self.processor_config = processor_config
        self.num_proc = num_proc

        self.log_queue = log_queue
        self.warning_queue = mp.Queue()
        self.error_queue = mp.Queue()

        if self.limit_output_queue_size() == True:
            max_output_queue_size = max(num_proc * processor_config.buffer_memory_level, 1)
        else:
            max_output_queue_size = 0
        self.input_queue = input_queue
        self.output_queue = WorkQueue(max_output_queue_size)

        critical_locks = [mp.Lock() for _ in range(num_proc)]
        finish_events = [mp.Event() for _ in range(num_proc)]
        workers: list[mp.Process] = []

        for rank in range(self.num_proc):
            cuda_device = self.processor_config.cuda_devices[rank] if self.get_stage_type() == "cuda" else None
            worker = mp.Process(target=_process_worker, daemon=True,
                args=(self, rank, cuda_device, critical_locks[rank], finish_events[rank]))
            workers.append(worker)
            worker.start()

        self.workers = workers
        self.critical_locks = critical_locks
        self.finish_events = finish_events

        return self.output_queue

    # adds one None per worker process to the input queue to signal end of input
    # this should only be called after _ALL_ items have been added to the input queue
    # timeouts are necessary because if the main process blocks it will no longer respond to ctrl+c
    def _finish_worker_queue(self) -> int: # returns total worker errors

        sentinels_sent = 0
        while sentinels_sent < len(self.workers):
            try:
                self.input_queue.put(None, timeout=0.01)
                sentinels_sent += 1
            except QueueFullException:
                time.sleep(0.1)

        while True:
            num_running = 0
            for finish_event in self.finish_events:
                if finish_event.wait(timeout=0.01) == False:
                    num_running += 1

            if num_running == 0: break
            time.sleep(0.1)

        return self.error_queue.qsize(), self.warning_queue.qsize()

    def _init_process(self, rank: int, device: Optional[str],
            logger: logging.Logger, critical_lock: Lock, finish_event: Event) -> None:
        self.rank = rank
        self.device = device
        self.logger = logger
        self.critical_lock = critical_lock
        self.finish_event = finish_event

    # acquiring the associated critical lock before terminating each worker will prevent data corruption
    def _terminate(self) -> None:
        for critical_lock, worker in zip(self.critical_locks, self.workers):
            with critical_lock:
                worker.terminate()
        
        for worker in self.workers:
            if worker.is_alive() == True:
                worker.join()

    # should only be called after _ALL_ workers have terminated due to shared memory
    def _close(self) -> None:
        for worker in self.workers:
            worker.close()

    # subclass can override this to print some info at the start of processing
    # this is executed on the first process stage in the main process
    # before any worker processes are started
    def info_banner(self, logger: logging.Logger) -> None:
        pass

    def summary_banner(self, logger: logging.Logger) -> None:
        pass
    
    # if the stage output objects are large this can be overridden to return True
    # to cap the maximum number of items that can reside in the output queue for this stage
    def limit_output_queue_size(self) -> bool:
        return False

    # io stages get 1 process, cuda stages get 1 process per device, the remaining processes are
    # evenly divided between any cpu stages until max_num_proc is reached
    def get_stage_type(self) -> Literal["io", "cpu", "cuda"]:
        return "io"
    
    @torch.inference_mode()
    def start_process(self) -> None:
        pass
    
    @torch.inference_mode()
    def finish_process(self) -> None:
        pass

    @abstractmethod
    @torch.inference_mode()
    def process(self, input_dict: dict) -> Optional[Union[dict, list[dict]]]:
        pass

@dataclass
class DatasetProcessorConfig:

    sample_rate: int                = 32000      # sample rate for dataset audio
    num_channels: int               = 2          # mono or stereo for dataset audio
    max_num_proc: Optional[int]     = None       # set max number of (total) processes for cpu stages. default is 1/2 cpu cores
    buffer_memory_level: int        = 2          # higher values increase max queue sizes and memory usage
    cuda_devices: list[str]         = ("cuda",)  # list of devices to use in cuda stages ("cuda:0", "cuda:1", etc)
    force_overwrite: bool           = False      # disables skipping files that have been previously processed
    copy_on_write: bool             = False      # enable to guarantee data integrity in event of abnormal/sudden termination
    test_mode: bool                 = False      # disables moving, copying, or writing any changes to files
    write_error_summary_logs: bool  = True       # when the process is completed or aborted write a separate log file for each stage's errors/warnings
    verbose: bool                   = False      # prints debug logs to stdout

    # import process
    import_paths: list[str]        = ()                # list of paths to move/copy from
    import_filter_regex: str       = "(?i)^.*\\.flac$" # regex for filtering and transforming filenames (default: *.flac)
    import_filter_group: int       = 0                 # regex group for destination filename.
    import_dst_path: Optional[str] = None              # import destination path. default is $DATASET_PATH
    import_warn_file_size_mismatch: bool  = True       # write warnings to debug log if the existing destination file has a different size
    import_overwrite_if_larger: bool      = True       # if the file to be imported exists but is larger, import it and overwrite the existing file
    import_move_no_copy: bool             = True       # enable to move files instead of copying them
    import_min_tree_depth: Optional[int]  = 1          # files with paths above min tree depth will use generated folder names
    import_max_tree_depth: Optional[int]  = 1          # folders below max tree depth in the source file path aren't included in destination path

    # normalize process
    normalize_target_lufs: float               = -20.  # desired loudness level for dataset audio in the normalization process
    normalize_trim_silence: bool               = True  # removes any silence at the beginning or end of the audio file
    normalize_trim_max_length: Optional[float] = 600   # truncates the length of the audio to this max length (in seconds)

    pre_encoded_latents_vae: Optional[str] = None
    pre_encoded_latents_num_time_offset_augmentations: int = 8
    pre_encoded_latents_pitch_offset_augmentations: list[int] = ()
    pre_encoded_latents_stereo_mirroring_augmentation: bool = True

    clap_embedding_model: Optional[str] = None
    clap_embedding_labels: Optional[dict[str, list[str]]] = None
    clap_embedding_tags: Optional[list[str]] = None

class DatasetSplit:

    def __init__(self, path: str) -> None:
        self.path = path
        self.name = os.path.splitext(os.path.basename(path))[0]

        self.init_split()

    def init_split(self) -> None:

        self.empty_sample = {
            "file_name": None,
            "sample_rate": None,
            "num_channels": None,
            "sample_length": None,
            "system": None,
            "game": None,
            "song": None,
            "author": None,
            "prompt": None,
            "rating": None,
            "latents_file_name": None,
            "latents_length": None,
            "latents_num_variations": None,
            "latents_has_audio_embeddings": None,
            "latents_has_text_embeddings": None,
        }
        if os.path.isfile(self.path):
            logging.getLogger().info(f"Loading split from {self.path}")
            with open(self.path, "r") as f:
                self.samples = [self.empty_sample | json.loads(line) for line in f]
        else:
            logging.getLogger().warning(f"Split not found at {self.path}, creating new split")
            self.samples: list[dict] = []

    def remove_samples(self, indices: list[int]) -> None:
        self.samples = [sample for index, sample in enumerate(self.samples) if index not in indices]
    
    def add_samples(self, samples: list[dict]) -> None:
        for sample in samples:
            self.samples.append(self.empty_sample | sample)

    def save(self, path: Optional[str] = None) -> None:
        with open(path or self.path, "w") as f:
            for sample in self.samples:
                f.write(json.dumps(sample) + "\n")

class SampleList:

    def __init__(self) -> None:
        self.samples: dict[DatasetSplit, list[int]] = {}
        self.annotations : dict[tuple[DatasetSplit, int], str] = {}

    def add_sample(self, split: DatasetSplit, index: int, annotation: Optional[str] = None) -> None:
        if split not in self.samples:
            self.samples[split] = []
        self.samples[split].append(index)

        if annotation is not None:
            self.annotations[(split, index)] = annotation

    def __len__(self) -> int:
        return sum(len(indices) for indices in self.samples.values())
    
    def __iter__(self) -> Generator[DatasetSplit, int, dict]:
        for split, indices in self.samples.items():
            yield from ((split, index, split.samples[index]) for index in indices)
    
    def remove_samples_from_dataset(self) -> None:
        for split, indices in self.samples.items():
            split.remove_samples(indices)

    def show_samples(self) -> None:
        logger = logging.getLogger()
        for split, index, sample in self:
            sample_str = f'{split.name}_{index}: "{sample["file_name"]}"'
            annotation = self.annotations.get((split, index), None)
            if annotation is not None: sample_str += f" ({annotation})"
            logger.debug(sample_str)

class DatasetProcessor:
    
    def __init__(self) -> None:

        if config.CONFIG_PATH is None:
            raise ValueError("CONFIG_PATH not defined")
        if not os.path.isdir(config.CONFIG_PATH):
            raise FileNotFoundError(f"CONFIG_PATH '{config.CONFIG_PATH}' not found")
        
        self.config = DatasetProcessorConfig(
            **config.load_json(os.path.join(config.CONFIG_PATH, "dataset", "dataset.json")))
        
        self.config.max_num_proc = self.config.max_num_proc or mp.cpu_count() // 2
        if self.config.cuda_devices is None:
            self.config.cuda_devices = []
        if isinstance(self.config.cuda_devices, str):
            self.config.cuda_devices = [self.config.cuda_devices]
        
    def load_splits(self) -> None:
        splits = [
            DatasetSplit(os.path.join(config.DATASET_PATH, f))
            for f in os.listdir(config.DATASET_PATH)
            if f.lower().endswith(".jsonl")
        ]
        self.splits = {split.name: split for split in splits}

    def init_dataset(self) -> None:
        
        # init dataset info
        self.dataset_info = {
            "features": {
                "file_name": {"type": "string"},
                "sample_rate": {"type": "int"},
                "num_channels": {"type": "int"},
                "sample_length": {"type": "int"},
                "system": {"type": "string"},
                "game": {"type": "string"},
                "song": {"type": "string"},
                "author": {"type": "string"},
                "prompt": {"type": "string"},
                "rating": {"type": "int"},
                "latents_file_name": {"type": "string"},
                "latents_length": {"type": "int"},
                "latents_num_variations": {"type": "int"},
                "latents_has_audio_embeddings": {"type": "bool"},
                "latents_has_text_embeddings": {"type": "bool"},
            },
            "num_split_samples": {},
            "total_num_samples": 0,
            "processor_config": self.config.__dict__,
        }

        if "train" not in self.splits:
            self.splits["train"] = DatasetSplit(os.path.join(config.DATASET_PATH, "train.jsonl"))
        if "validation" not in self.splits:
            self.splits["validation"] = DatasetSplit(os.path.join(config.DATASET_PATH, "validation.jsonl"))
        if "negative" not in self.splits:
            self.splits["negative"] = DatasetSplit(os.path.join(config.DATASET_PATH, "negative.jsonl"))

    def show_dataset_summary(self) -> None:

        self.logger.info(f"\nTotal samples: {self.num_samples()}")
        self.logger.info("Splits:")
        for split in self.splits.values():
            self.logger.info(f"  {split.name}: {len(split.samples)} samples")
    
    def all_samples(self) -> Generator[DatasetSplit, int, dict]:
        for _, split in self.splits.items():
            yield from ((split, index, sample) for index, sample in enumerate(split.samples))

    def num_samples(self) -> int:
        return sum(len(split.samples) for split in self.splits.values())

    def validate_files(self) -> None:

        # search for any sample file_name in splits that no longer exists (or null file_name)
        # if any found, prompt to remove the samples from splits
        missing_samples = SampleList()
        for split, index, sample in self.all_samples():
            if sample["file_name"] is None or (not os.path.isfile(os.path.join(config.DATASET_PATH, sample["file_name"]))):
                missing_samples.add_sample(split, index)
        
        num_missing = len(missing_samples)
        if num_missing > 0:
            self.logger.warning(f"Found {num_missing} samples in dataset with missing files or no file_name")
            missing_samples.show_samples()
            if input(f"Remove {num_missing} samples with missing files? (y/n): ").lower() == "y":
                missing_samples.remove_samples_from_dataset()
                self.logger.info(f"{num_missing} samples with missing files removed")
            self.logger.info("")

        # search for any latents_file_name in splits that no longer exists
        # if any found, prompt to clear latents metadata for those samples
        missing_latents_samples = SampleList()
        for split, index, sample in self.all_samples():
            if sample["latents_file_name"] is not None:
                if not os.path.isfile(os.path.join(config.DATASET_PATH, sample["latents_file_name"])):
                    missing_latents_samples.add_sample(split, index)

        num_missing_latents = len(missing_latents_samples)
        if num_missing_latents > 0:
            self.logger.warning(f"Found {num_missing_latents} samples with nonexistent latents_file_name")
            missing_latents_samples.show_samples()
            if input(f"Clear latents metadata for {num_missing_latents} samples? (y/n): ").lower() == "y":
                for _, _, sample in missing_latents_samples:
                    sample["latents_file_name"] = None
                    sample["latents_file_size"] = None
                    sample["latents_length"] = None
                    sample["latents_num_variations"] = None
                    sample["latents_quantized"] = None
                    sample["latents_has_audio_embeddings"] = None
                    sample["latents_has_text_embeddings"] = None
                self.logger.info(f"Cleared latents metadata for {num_missing_latents} samples")
            self.logger.info("")

        # search for any samples with a file_name that is not in the source formats list
        # if any found, prompt to remove them from splits
        invalid_format_samples = SampleList()
        valid_sample_file_formats = self.config.source_formats + self.config.dataset_formats
        for split, index, sample in self.all_samples():
            if os.path.splitext(sample["file_name"])[1].lower() not in valid_sample_file_formats:
                invalid_format_samples.add_sample(split, index)

        num_invalid_format = len(invalid_format_samples)
        if num_invalid_format > 0:
            self.logger.warning(f"Found {num_invalid_format} samples with file formats not in the source format list ({self.config.source_formats})")
            invalid_format_samples.show_samples()
            if input(f"Remove {num_invalid_format} samples with invalid file formats? (this will not delete the files) (y/n): ").lower() == "y":
                invalid_format_samples.remove_samples_from_dataset()
                self.logger.info(f"{num_invalid_format} samples with invalid file formats removed")
            self.logger.info("")

        # search for any samples with the same file_name in any splits
        # if any found, prompt to remove duplicates
        sample_files = set()
        duplicate_samples = SampleList()
        for split, index, sample in self.all_samples():
            norm_path = os.path.normpath(sample["file_name"])
            if norm_path in sample_files:
                duplicate_samples.add_sample(split, index)
            else:
                sample_files.add(norm_path)

        num_duplicates = len(duplicate_samples)
        if num_duplicates > 0:
            self.logger.warning(f"Found {num_duplicates} samples with duplicated file_names")
            duplicate_samples.show_samples()
            if input(f"Remove {num_duplicates} samples with duplicate file_names? (y/n): ").lower() == "y":
                duplicate_samples.remove_samples_from_dataset()
                self.logger.info(f"{num_duplicates} samples with duplicate file_names removed")
            self.logger.info("")

    def clean(self) -> None:

        # search for any files (excluding dataset metadata and latents files) that are not in the source formats list
        # if any found, prompt to permanently delete them
        invalid_format_files = []
        valid_dataset_file_formats = [".flac", ".safetensors", ".jsonl", ".json", ".md"]
        for root, _, files in os.walk(config.DATASET_PATH):
            for file in files:
                if os.path.splitext(file)[1].lower() not in valid_dataset_file_formats:
                    invalid_format_files.append(os.path.join(root, file))

        num_invalid_format_files = len(invalid_format_files)
        if num_invalid_format_files > 0:
            self.logger.warning(f"Found {num_invalid_format_files} files with formats not in the source format list ({valid_dataset_file_formats})")
            for file in invalid_format_files:
                self.logger.debug(f'"{file}"')
            if input(f"Delete {num_invalid_format_files} files with invalid formats? (WARNING: this is permanent and cannot be undone) (type 'delete' to confirm): ").lower() == "delete":
                for file in invalid_format_files:
                    os.remove(file)
                self.logger.info(f"Deleted {num_invalid_format_files} files with invalid formats")
            print("")
    
        # search for any empty folders, if any found prompt to delete
        empty_folders = []
        for root, dirs, files in os.walk(config.DATASET_PATH):
            if len(dirs) == 0 and len(files) == 0:
                empty_folders.append(root)
        
        if len(empty_folders) > 0:
            self.logger.warning(f"Found {len(empty_folders)} empty folders in dataset ({config.DATASET_PATH})")
            for folder in empty_folders:
                self.logger.debug(f'"{folder}"')
            if input(f"Delete {len(empty_folders)} empty folders? (WARNING: this is permanent and cannot be undone) (type 'delete' to confirm): ").lower() == "delete":
                for folder in empty_folders:
                    os.rmdir(folder)
                self.logger.info(f"Deleted {len(empty_folders)} empty folders")
            print("")

    # process a pipeline of DatasetProcessStage objects in parallel with a specified input path or WorkQueue
    def process(self, process_name: str, process_stages: list[DatasetProcessStage],
                input: Optional[Union[list[str], WorkQueue]] = None) -> None:
    
        # first stage input is either a specified path, None (dataset path), or pre-filled WorkQueue
        if input is None:
            if config.DATASET_PATH is None:
                raise ValueError("DATASET_PATH not defined")
            input = [config.DATASET_PATH]

        elif isinstance(input, str):
            input = [input]

        if isinstance(input, list):
            scan_paths = input
            input_queue = WorkQueue()

            for scan_path in scan_paths:
                if not os.path.isdir(scan_path):
                    raise FileNotFoundError(f"Input path '{scan_path}' not found")
        elif isinstance(input, WorkQueue):
            scan_paths = None
            input_queue = input
        else:
            raise ValueError(f"Unrecognized input type in DatasetProcessor.process: {type(input)}")

        # only one stage is allowed to have cuda device(s)
        cuda_stage_names = [stage.__class__.__name__ for stage in process_stages if stage.get_stage_type() == "cuda"]
        if len(cuda_stage_names) > 1:
            raise ValueError(f"More than one stage has cuda devices: {cuda_stage_names}")

        # validate selected cuda devices if there is a cuda stage
        if len(cuda_stage_names) > 0:
            available_cuda_devices = get_available_torch_devices()
            for device in self.config.cuda_devices:
                if device not in available_cuda_devices:
                    raise ValueError(f"Selected cuda device '{device}' not available")

        # start logging process
        log_queue = mp.Queue()
        log_worker = mp.Process(target=_log_worker, daemon=True,
            args=(process_name, self.config.verbose, log_queue))
        log_worker.start()

        logger = logging.getLogger("DatasetProcessor")
        logger.setLevel(logging.DEBUG if self.config.verbose == True else logging.INFO)
        warning_queue, error_queue = mp.Queue(), mp.Queue()
        log_handler = WorkerLogHandler(log_queue, warning_queue, error_queue, "DatasetProcessor")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(log_handler)
        logger.info("")

        # 3 second chance to abort if force overwrite is enabled
        if self.config.force_overwrite == True and self.config.test_mode == False:
            logger.warning("WARNING: Force overwrite is enabled - existing files will be overwritten (Press ctrl+c to abort)")
            time.sleep(3)

        if self.config.test_mode == True:
            logger.warning("WARNING: Test mode is enabled - no files will be written")

        # if no cuda devices are configured but there is a cuda stage in the process, use all available cuda devices
        if len(cuda_stage_names) > 0 and len(self.config.cuda_devices) == 0:
            if len(available_cuda_devices) > 0:
                logger.warning(f"No cuda devices configured, using all available cuda devices ({available_cuda_devices})")
                self.config.cuda_devices = available_cuda_devices
            else: # fallback to cpu for cuda stage if no devices are available
                logger.warning("No cuda devices available, using cpu processing only")
                self.config.cuda_devices = ["cpu"]
        
        # allocate processes to each stage based on stage type
        # todo: if this isn't good enough we should use PriorityQueue to create a cpu proc pool
        stage_names = [stage.__class__.__name__ for stage in process_stages]
        stage_types = [stage.get_stage_type() for stage in process_stages]

        remaining_num_procs = self.config.max_num_proc
        num_cpu_stages = 0
        stage_num_procs: list[int] = []

        for stage, stage_type in zip(process_stages, stage_types):
            if stage_type == "io":
                stage_num_procs.append(1)
            elif stage_type == "cuda":
                stage_num_procs.append(len(self.config.cuda_devices))
            elif stage_type == "cpu":
                stage_num_procs.append(0)
                num_cpu_stages += 1
            else:
                raise ValueError(f"Unrecognized stage type '{stage_type}'")
            
            remaining_num_procs -= stage_num_procs[-1]

        cpu_stage_procs = max(remaining_num_procs // max(num_cpu_stages, 1), 1)
        stage_num_procs = [num_proc if num_proc > 0 else cpu_stage_procs for num_proc in stage_num_procs]

         # disable sigint to avoid data corruption
        original_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        # setup processing queue chain and start workers for each stage
        logger.info(f"Process layout for {len(process_stages)} stage(s) using {sum(stage_num_procs)} total processes")
        input_queues: list[WorkQueue] = []

        for stage, num_proc, stage_name, stage_type in zip(process_stages, stage_num_procs,
                                                           stage_names, stage_types):
            logger.info(
                f"Stage: {stage_name:<{max(len(n) for n in stage_names)+1}} "
                f"Type: {stage_type:<{max(len(t) for t in stage_types)+1}} "
                f"Processes: {num_proc:<{3}}"
                f"CUDA Devices: {self.config.cuda_devices if stage_type == 'cuda' else 'n/a'} "
            )

            input_queues.append(input_queue)
            input_queue = stage._start_worker_queue(
                input_queue, log_queue, num_proc, self.config)

        logger.info(""); process_stages[0].info_banner(logger); logger.info("")
        time.sleep(0.2)

        # start a process for monitoring progress of all stages
        monitor_finish_event = mp.Event()
        monitor_worker = mp.Process(target=_monitor_worker, daemon=True,
            args=(input_queues, stage_names, monitor_finish_event))
        monitor_worker.start()

        process_result = "completed successfully"
        process_start_time = datetime.now()
        signal.signal(signal.SIGINT, original_sigint_handler)  # re-enable sigint as it will be caught
        try:
            # if first stage input is a list of paths, scan the paths and fill the first input_queue
            if scan_paths is not None:
                for scan_path in scan_paths:
                    for root, _, files in os.walk(scan_path):
                        for file in files:
                            input_queues[0].put({"scan_path": scan_path, "file_path": os.path.join(root, file)})
            
            # wait for each stage to finish
            total_errors = 0; total_warnings = 0
            for stage in process_stages:
                errors, warnings = stage._finish_worker_queue()
                total_errors += errors; total_warnings += warnings

            if total_errors > 0:
                process_result = f"completed with {total_errors} errors, {total_warnings} warnings"

        except (KeyboardInterrupt, Exception) as e:
            # disable sigint to avoid data corruption while worker processes shut down
            signal.signal(signal.SIGINT, signal.SIG_IGN)

            process_result = "aborted" if isinstance(e, KeyboardInterrupt) else "failed"
            logger.error("".join(format_exception(type(e), e, e.__traceback__)))

        # safely terminate worker processes by acquiring each per-process critical section lock
        # doing this safely requires reverse order because of pytorch shared memory
        logger.info("Safely terminating worker processes...")
        for stage in reversed(process_stages):
            stage._terminate()

        # with worker processes terminated, finally re-enable sigint once more
        signal.signal(signal.SIGINT, original_sigint_handler)

        # close monitor process
        monitor_finish_event.set()
        _terminate_worker(monitor_worker, close=True)
        
        # get stage error / warning counts
        stage_error_counts = []; stage_warning_counts = []
        for stage in process_stages:
            stage_error_counts.append(stage.error_queue.qsize())
            stage_warning_counts.append(stage.warning_queue.qsize())

        # write error / warning summaries
        if self.config.write_error_summary_logs == True:
            for stage in process_stages:
                try:
                    if stage.error_queue.qsize() > 0:
                        logger.info(""); logger.info(f"{stage.__class__.__name__} errors:")
                        while stage.error_queue.qsize() > 0:
                            log_handler.emit(stage.error_queue.get())
                except: pass
                
                try:
                    if stage.warning_queue.qsize() > 0:
                        logger.info(""); logger.info(f"{stage.__class__.__name__} warnings:")
                        while stage.warning_queue.qsize() > 0:
                            log_handler.emit(stage.warning_queue.get())
                except: pass
        
        # summarize results
        process_finish_time = datetime.now(); logger.info("")
        try: process_stages[-1].summary_banner(logger)
        except Exception as e:
            logger.error("".join(format_exception(type(e), e, e.__traceback__)))

        for i, stage in enumerate(process_stages):
            stage_name = stage.__class__.__name__
            processed, total = stage.input_queue.get_processed_total()
            errors, warnings = stage_error_counts[i], stage_warning_counts[i]
            logger.info(f"{stage_name}: {processed}/{total} processed - {errors} errors, {warnings} warnings")

        logger.info(f"Process '{process_name}' {process_result} - time elapsed: {process_finish_time - process_start_time}\n")

        # close logging process
        log_handler.emit(None) # None value signals the log_worker process to gracefully end
        _terminate_worker(log_worker, close=True)

        # finally, release all memory/resources for all worker processes
        for stage in process_stages:
            stage._close()

    def transcode(self) -> None:

        # search for any samples in splits that have any null audio metadata fields
        # if any found, prompt to extract audio metadata
        need_metadata_samples = SampleList()
        metadata_check_keys = ["file_size", "sample_rate", "num_channels", "sample_length"]
        for split, index, sample in self.all_samples():
            if any(sample[key] is None for key in metadata_check_keys):
                need_metadata_samples.add_sample(split, index)

        def get_audio_metadata(sample: dict) -> None:
            if sample["file_name"] is None:
                sample["file_size"] = None
                sample["sample_rate"] = None
                sample["num_channels"] = None
                sample["sample_length"] = None
                sample["bit_rate"] = None
                return
            audio_info = mutagen.File(os.path.join(config.DATASET_PATH, sample["file_name"])).info
            sample["file_size"] = os.path.getsize(os.path.join(config.DATASET_PATH, sample["file_name"]))
            sample["sample_rate"] = audio_info.sample_rate
            sample["num_channels"] = audio_info.channels
            sample["sample_length"] = int(audio_info.length * sample["sample_rate"])
            sample["bit_rate"] = int(sample["file_size"] / 128 / audio_info.length)

        num_need_metadata = len(need_metadata_samples)
        if num_need_metadata > 0:
            self.logger.warning(f"Found {num_need_metadata} samples with missing audio metadata")
            need_metadata_samples.show_samples()
            if input(f"Extract missing audio metadata for {num_need_metadata} samples? (y/n): ").lower() == "y":
                failed_metadata_extraction_samples = SampleList()
                for split, index, sample in tqdm(need_metadata_samples, total=num_need_metadata, mininterval=1):
                    try:
                        get_audio_metadata(sample)
                    except Exception as e:
                        failed_metadata_extraction_samples.add_sample(split, index, str(e))
                        continue

                num_failed_metadata = len(failed_metadata_extraction_samples)
                if num_failed_metadata > 0:
                    self.logger.warning(f"Failed to extract audio metadata for {len(failed_metadata_extraction_samples)} samples")
                    failed_metadata_extraction_samples.show_samples()
                self.logger.info(f"Extracted missing audio metadata for {num_need_metadata - num_failed_metadata} samples")

            self.logger.info("")

        # search for any samples with with sample_rate / sample_length / kbps < config values
        # if any found, prompt to delete them. if not, prompt to remove them
        invalid_audio_samples = SampleList()
        min_sample_length = self.config.min_sample_length / self.config.sample_rate if self.config.min_sample_length is not None else None
        max_sample_length = self.config.max_sample_length / self.config.sample_rate if self.config.max_sample_length is not None else None
        for split, index, sample in self.all_samples():
            if sample["sample_rate"] is not None:
                if sample["sample_rate"] != self.config.sample_rate:
                    invalid_audio_samples.add_sample(split, index, f"sample_rate {sample['sample_rate']} != {self.config.sample_rate}")
            if sample["sample_length"] is not None and sample["sample_rate"] is not None:
                sample_length = sample["sample_length"] / sample["sample_rate"]
                if min_sample_length is not None and sample_length < min_sample_length:
                    invalid_audio_samples.add_sample(split, index, f"sample_length {sample_length:.2f}s < {min_sample_length:.2f}s")
                if max_sample_length is not None and sample_length > max_sample_length:
                    invalid_audio_samples.add_sample(split, index, f"sample_length {sample_length:.2f}s > {max_sample_length:.2f}s")
            if sample["num_channels"] is not None:
                if sample["num_channels"] != self.config.num_channels:
                    invalid_audio_samples.add_sample(split, index, f"num_channels {sample['num_channels']} != {self.config.num_channels}")            

        num_invalid_audio = len(invalid_audio_samples)
        if num_invalid_audio > 0:
            self.logger.warning(f"Found {num_invalid_audio} samples with sample_rate, length, or num_channels that do not conform to configured dataset values")
            invalid_audio_samples.show_samples()
            self.logger.info("")

        # transcoding in dataset_processor removed, it is done better in an external tool (foobar2000)
    
    def validate_metadata(self) -> None:
        
        # search for any samples in splits that have a latents_file_name and any null latents metadata fields
        # if any found, prompt to extract latents metadata
        need_metadata_samples = SampleList()
        metadata_check_keys = [key for key in self.dataset_info["features"].keys() if key.startswith("latents_") and key != "latents_file_name"]
        for split, index, sample in self.all_samples():
            if sample["latents_file_name"] is not None:
                if any(sample[key] is None for key in metadata_check_keys) or (sample["prompt"] is not None and sample["latents_has_text_embeddings"] == False):
                    need_metadata_samples.add_sample(split, index)

        num_need_metadata = len(need_metadata_samples)
        if num_need_metadata > 0:
            self.logger.warning(f"Found {num_need_metadata} samples with missing latents metadata")
            need_metadata_samples.show_samples()
            if input(f"Extract missing latents metadata for {num_need_metadata} samples? (y/n): ").lower() == "y":
                failed_metadata_extraction_samples = SampleList()
                for split, index, sample in tqdm(need_metadata_samples, total=num_need_metadata, mininterval=1):
                    try:
                        latents_file_path = os.path.join(config.DATASET_PATH, sample["latents_file_name"])
                        sample["latents_file_size"] = os.path.getsize(latents_file_path)

                        with safetensors.safe_open(latents_file_path, framework="pt") as f:
                            latents_shape = f.get_slice("latents").get_shape()
                            sample["latents_length"] = latents_shape[-1]
                            sample["latents_num_variations"] = latents_shape[0]
                            
                            try:
                                _ = f.get_slice("offset_and_range")
                                sample["latents_quantized"] = True
                            except Exception as _:
                                sample["latents_quantized"] = False

                            try:
                                _ = f.get_slice("clap_audio_embeddings")
                                sample["latents_has_audio_embeddings"] = True
                            except Exception as _:
                                sample["latents_has_audio_embeddings"] = False
                            
                            try:
                                _ = f.get_slice("clap_text_embeddings")
                                sample["latents_has_text_embeddings"] = True
                            except Exception as _:
                                sample["latents_has_text_embeddings"] = False

                    except Exception as e:
                        failed_metadata_extraction_samples.add_sample(split, index, str(e))
                        continue

                num_failed_metadata = len(failed_metadata_extraction_samples)
                if num_failed_metadata > 0:
                    self.logger.warning(f"Failed to extract latents metadata for {len(failed_metadata_extraction_samples)} samples")
                    failed_metadata_extraction_samples.show_samples()
                self.logger.info(f"Extracted missing latents metadata for {num_need_metadata - num_failed_metadata} samples")

            self.logger.info("")


        # search for any samples in splits that have any null id metadata fields
        # if any found, prompt to extract metadata
        need_metadata_samples = SampleList()
        metadata_check_keys = ["system", "game", "song", "author", "system_id", "game_id", "author_id"]
        for split, index, sample in self.all_samples():
            if any(sample[key] is None for key in metadata_check_keys):
                need_metadata_samples.add_sample(split, index)

        num_need_metadata = len(need_metadata_samples)
        if num_need_metadata > 0:
            self.logger.warning(f"Found {num_need_metadata} samples with missing id metadata")
            need_metadata_samples.show_samples()
            if input(f"Extract missing id metadata for {num_need_metadata} samples? (y/n): ").lower() == "y":

                num_pre_update_system_ids = len(self.dataset_info["system_id"])
                num_pre_update_game_ids = len(self.dataset_info["game_id"])
                num_pre_update_author_ids = len(self.dataset_info["author_id"])

                metadata_check_keys = ["system", "game", "song", "author"]
                failed_metadata_extraction_samples = SampleList()
                for split, index, sample in tqdm(need_metadata_samples, total=num_need_metadata, mininterval=1):
                    try:
                        if any(sample[key] is None for key in metadata_check_keys):
                            file_path_list = os.path.normpath(sample["file_name"]).split(os.sep)
                            if len(file_path_list) == 3:
                                if sample["system"] is None: sample["system"] = file_path_list[0]
                                if sample["game"] is None: sample["game"] = f"{file_path_list[0]}/{file_path_list[1]}"
                                if sample["song"] is None: sample["song"] = os.path.splitext(os.path.basename(sample["file_name"]))[0]
                                if sample["author"] is None:
                                    sample["author"] = []
                                    sample_metadata = get_audio_metadata(os.path.join(config.DATASET_PATH, sample["file_name"]))
                                    if "author" in sample_metadata:
                                        for author in sample_metadata["author"]:
                                            sample["author"].extend([author.strip() for author in author.split(",")])
                            else:
                                failed_metadata_extraction_samples.add_sample(split, index, "path does not match system/game/song format")
                                continue

                        if sample["system_id"] is None: sample["system_id"] = self.get_id("system_id", sample["system"])
                        if sample["game_id"] is None: sample["game_id"] = self.get_id("game_id", sample["game"])
                        if sample["author_id"] is None: sample["author_id"] = [self.get_id("author_id", author) for author in sample["author"]]

                    except Exception as e:
                        failed_metadata_extraction_samples.add_sample(split, index, str(e))
                        continue
                
                num_failed_metadata = len(failed_metadata_extraction_samples)
                if num_failed_metadata > 0:
                    self.logger.warning(f"Failed to extract id metadata for {len(failed_metadata_extraction_samples)} samples")
                    failed_metadata_extraction_samples.show_samples()

                self.logger.info(f"Extracted missing id metadata for {num_need_metadata - num_failed_metadata} samples")
                if num_pre_update_system_ids != len(self.dataset_info["system_id"]):
                    self.logger.info(f"Added {len(self.dataset_info['system_id']) - num_pre_update_system_ids} new system id(s)")
                if num_pre_update_game_ids != len(self.dataset_info["game_id"]):
                    self.logger.info(f"Added {len(self.dataset_info['game_id']) - num_pre_update_game_ids} new game id(s)")
                if num_pre_update_author_ids != len(self.dataset_info["author_id"]):
                    self.logger.info(f"Added {len(self.dataset_info['author_id']) - num_pre_update_author_ids} new author id(s)")

            self.logger.info("")

        # search for any unused system, game, or author ids
        # if any found, prompt to rebuild dataset_info / ids from current metadata
        unused_system_ids, unused_game_ids, unused_author_ids = self.get_unused_ids()
        if len(unused_system_ids) > 0 or len(unused_game_ids) > 0 or len(unused_author_ids) > 0:
            self.logger.warning("Found unused system, game, or author ids in dataset info")
            if len(unused_system_ids) > 0:
                self.logger.warning(f"Unused system ids: {len(unused_system_ids)}")
                self.logger.debug(f"{dict_str(unused_system_ids)}")
            if len(unused_game_ids) > 0:
                self.logger.warning(f"Unused game ids: {len(unused_game_ids)}")
                self.logger.debug(f"{dict_str(unused_game_ids)}")
            if len(unused_author_ids) > 0:
                self.logger.warning(f"Unused author ids: {len(unused_author_ids)}")
                self.logger.debug(f"{dict_str(unused_author_ids)}")

            if input("Rebuild all dataset ids from current metadata? (WARNING: any models trained with current ids will have incorrect class labels) (type 'rebuild' to confirm): ").lower() == "rebuild":
                self.dataset_info["system_id"] = {}
                self.dataset_info["features"]["system_id"]["num_classes"] = 0
                self.dataset_info["game_id"] = {}
                self.dataset_info["features"]["game_id"]["num_classes"] = 0
                self.dataset_info["author_id"] = {}
                self.dataset_info["features"]["author_id"]["value_type"]["num_classes"] = 0

                for _, _, sample in self.all_samples():
                    sample["system_id"] = self.get_id("system_id", sample["system"])
                    sample["game_id"] = self.get_id("game_id", sample["game"])
                    sample["author_id"] = [self.get_id("author_id", author) for author in sample["author"]]

                self.logger.info("Rebuilt all dataset ids from current metadata")
            self.logger.info("")

        # search for any samples in splits that have any null prompt field
        # if any found, prompt to to create a default prompt for each sample
        need_prompt_samples = SampleList()
        for split, index, sample in self.all_samples():
            if sample["prompt"] is None:
                if (sample["song"] is not None and sample["song"] != "") or (sample["game"] is not None and sample["game"] != ""):
                    need_prompt_samples.add_sample(split, index)

        num_need_prompt = len(need_prompt_samples)
        if num_need_prompt > 0:
            self.logger.warning(f"Found {num_need_prompt} samples with song metadata and missing prompt")
            need_prompt_samples.show_samples()
            if input(f"Create default prompt for {num_need_prompt} samples? (y/n): ").lower() == "y":
                for _, _, sample in need_prompt_samples:
                    song = sample["song"] or ""
                    game = sample["game"] or ""
                    if game != "": game = game.split("/")[1]
                    if "miscellaneous" in game.lower(): game = ""
                    if game.lower().split(" ")[0] in song.lower().split(" ")[0]: game = ""

                    prompt = f"{game} - {song}" if game != "" else song
                    if sample["author"] is not None and len(sample["author"]) > 0:
                        prompt += f" by {', '.join(sample['author'])}"
                    sample["prompt"] = prompt
                    self.logger.debug(prompt)

                self.logger.info(f"Created default prompt for {num_need_prompt} samples")
            self.logger.info("")

    def encode_latents(self) -> None:

        """
        # if self.config.pre_encoded_latents_vae is null skip encode_latents step
        if self.config.pre_encoded_latents_vae is None:
            self.logger.warning("Skipping encode_latents because config.pre_encoded_latents_vae is not defined")

        # search for any samples with a null latents_vae_model, or
        # a latents_vae_model that doesn't match the config vae model name
        # or a null latents_file_name
        # if any found, prompt to encode them with configured vae and update metadata
        # will need to launch subprocess to use accelerate
        # accelerate config for pre_encoding is in config.CONFIG_PATH/dataset/dataset_accelerate.yaml
        """
        

        samples_with_audio_embeddings = SampleList()
        samples_with_text_embeddings = SampleList()
        samples_with_embeddings = SampleList()
        for split, index, sample in self.all_samples():
            if sample["latents_has_audio_embeddings"] == True:
                samples_with_audio_embeddings.add_sample(split, index)
            if sample["latents_has_text_embeddings"] == True:
                samples_with_text_embeddings.add_sample(split, index)
            if sample["latents_has_audio_embeddings"] == True or sample["latents_has_text_embeddings"] == True:
                samples_with_embeddings.add_sample(split, index)

        num_samples_with_embeddings = len(samples_with_embeddings)
        if num_samples_with_embeddings > 0:
            num_samples_with_audio_embeddings = len(samples_with_audio_embeddings)
            num_samples_with_text_embeddings = len(samples_with_text_embeddings)
            self.logger.info(f"Found {num_samples_with_audio_embeddings} samples with audio embeddings and {num_samples_with_text_embeddings} samples with text embeddings")
            samples_with_embeddings.show_samples()

            if input(f"Aggregate embeddings? (you only need to do this when the dataset embeddings have changed) (y/n): ").lower() == "y":
                self.logger.info("Aggregating dataset audio and text embeddings...")
                dataset_embeddings_dict = {
                    "_unconditional_audio": torch.zeros(512, dtype=torch.float64),
                    "_unconditional_text": torch.zeros(512, dtype=torch.float64),
                }

                for split, index, sample in tqdm(samples_with_embeddings, total=num_samples_with_embeddings, mininterval=1):
                    latents_path = os.path.join(config.DATASET_PATH, sample["latents_file_name"])
                    with safetensors.safe_open(latents_path, framework="pt") as f:
                        
                        if sample["latents_has_audio_embeddings"] == True:
                            dataset_embeddings_dict["_unconditional_audio"].add_(
                                f.get_slice("clap_audio_embeddings")[:].to(torch.float64).mean(dim=0), alpha=1./num_samples_with_audio_embeddings)
                            if sample["game"] is not None:
                                game_audio_embeddings = dataset_embeddings_dict.get(f"{sample['game']}_audio",
                                    torch.zeros_like(dataset_embeddings_dict["_unconditional_audio"]))
                                game_audio_embeddings.add_(f.get_slice("clap_audio_embeddings")[:].to(torch.float64).mean(dim=0))
                                dataset_embeddings_dict[f"{sample['game']}_audio"] = game_audio_embeddings

                        if sample["latents_has_text_embeddings"] == True:
                            dataset_embeddings_dict["_unconditional_text"].add_(
                                f.get_slice("clap_text_embeddings")[:].to(torch.float64).mean(dim=0), alpha=1./num_samples_with_text_embeddings)
                            if sample["game"] is not None:
                                game_text_embeddings = dataset_embeddings_dict.get(f"{sample['game']}_text",
                                    torch.zeros_like(dataset_embeddings_dict["_unconditional_text"]))
                                game_text_embeddings.add_(f.get_slice("clap_text_embeddings")[:].to(torch.float64).mean(dim=0))
                                dataset_embeddings_dict[f"{sample['game']}_text"] = game_text_embeddings
                
                dataset_embeddings_dict = {k: normalize(v).float() for k, v in dataset_embeddings_dict.items()}
                output_path = os.path.join(config.DATASET_PATH, "dataset_infos", "dataset_embeddings.safetensors")
                self.logger.info(f"Saving aggregated dataset embeddings to '{output_path}'...")
                save_safetensors(dataset_embeddings_dict, output_path)
                self.logger.info("")
 
    def save(self, dataset_path: Optional[str] = None) -> None:
    
        dataset_path = dataset_path or config.DATASET_PATH

        # add total number of samples in train / validation splits to dataset_info
        self.dataset_info["total_num_samples"] = self.num_samples()
        self.dataset_info["num_split_samples"] = {}
        for split in self.splits.values():
            self.dataset_info["num_split_samples"][split.name] = len(split.samples)

        # prompt to save and backup existing metadata files to config.DEBUG_PATH

        if os.path.isfile(self.dataset_info_path):
            if self.backup_path is None:
                backup_warning = " (WARNING: Dataset metadata backup is NOT enabled)"
            else:
                backup_warning = f" (Backing up to '{self.backup_path}')"
        else:
            backup_warning = " No existing dataset metadata to backup"
            self.backup_path = None

        if input(f"Save changes to dataset metadata? (path: '{dataset_path}') (y/n){backup_warning}: ").lower() == "y":
            if self.backup_path is not None:
                self.logger.info(f"Backing up dataset metadata to '{self.backup_path}'")
                backup_dataset_info_path = os.path.join(self.backup_path, "dataset_infos", "dataset_info.json")
                os.makedirs(os.path.dirname(backup_dataset_info_path), exist_ok=True)
                
                shutil.copy(self.dataset_info_path, backup_dataset_info_path)
                for split in self.splits.values():
                    shutil.copy(split.path, os.path.join(self.backup_path, f"{split.name}.jsonl"))

            self.logger.info(f"Saving dataset metadata to '{dataset_path}'")
            config.save_json(self.dataset_info, os.path.join(dataset_path, "dataset_infos", "dataset_info.json"))
            for split in self.splits.values():
                split.save(os.path.join(dataset_path, f"{split.name}.jsonl"))
            self.logger.info(f"Saved dataset metadata to '{dataset_path}' successfully")
        else:
            self.logger.info(f"Finished without saving changes")