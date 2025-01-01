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
from typing import Generator, Optional, Union, Any
from abc import ABC, abstractmethod
from traceback import format_exception
from queue import Full as QueueFullException
from datetime import datetime
import os
import json
import logging
import shutil
import time

from tqdm.auto import tqdm
import torch
import mutagen
import torch.multiprocessing as mp
import safetensors.torch as safetensors

from utils.dual_diffusion_utils import (
    dict_str, normalize, save_safetensors, get_audio_metadata, init_logging, init_cuda
)


def _process_worker(stage: "DatasetProcessStage", rank: int, cuda_device: Optional[str]) -> None:

    stage_name = stage.__class__.__name__

    # init pytorch 
    if cuda_device is not None:
        assert stage.is_cuda_stage()
        init_cuda()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    # init logging
    logger = logging.getLogger(stage_name)
    logger.setLevel(logging.DEBUG if stage.processor_config.verbose == True else logging.INFO)
    logging.basicConfig(handlers=[WorkerLogHandler(stage.log_queue, stage_name)])

    # init process
    try:
        stage.start_process(rank, cuda_device, logger)
    except Exception as e:
        logger.error("".join(format_exception(type(e), e, e.__traceback__)))
        return
    
    while True:
        input_dict: dict = stage.input_queue.get()
        if input_dict is None:
            stage.finish_process()
            return

        try:
            output = stage.process(input_dict)
            if output is not None:
                if isinstance(output, list):
                    for item in output:
                        stage.output_queue.put(item)
                elif isinstance(output, dict):
                    stage.output_queue.put(output)
                else:
                    raise ValueError(f"stage.process returned unrecognized type '{type(output)}'")

        except Exception as e:
            logger.error("".join(format_exception(type(e), e, e.__traceback__)))

def _log_worker(process_name: str, verbose: bool, log_queue: mp.Queue):

    logger = init_logging(process_name, group_name="dataset_processor", verbose=verbose)
    while True:
        record = log_queue.get()
        if record is None: return

        logger.handle(record)
        for handler in logger.handlers:
            handler.flush()

def _monitor_worker(input_queues: list["WorkQueue"], stage_names: list[str], finish_event) -> None:

    progress_bars = []
    for name in stage_names:
        progress_bar = tqdm(total=1)
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
        time.sleep(0.1)

    for progress_bar, input_queue in zip(progress_bars, input_queues):
        update_progress(progress_bar, input_queue)
        progress_bar.close()

def _terminate_worker(process: mp.Process, timeout: float = 0.1):
    process.join(timeout=timeout)
    if process.exitcode is None:
        process.terminate()
        process.join(timeout=timeout)
        if process.exitcode is None:
            process.kill()
            process.join()

class WorkQueue: # normal Queue class extended with progress tracking

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

class WorkerLogHandler(logging.Handler): # dummy log handler that sends log records to a target queue
    def __init__(self, log_queue: mp.Queue, process_name: Optional[str] = None) -> None:
        super().__init__()
        self.log_queue = log_queue
        self.process_name = process_name or ""

    def emit(self, record: Optional[logging.LogRecord]) -> None:
        if record is not None:
            record.msg = f"{self.process_name}: {record.msg}"
        self.log_queue.put(record)

class DatasetProcessStage(ABC):

    def __init__(self) -> None:
        pass
    
    def _start_worker_queue(self, input_queue: WorkQueue, log_queue: mp.Queue,
                processor_config: "DatasetProcessorConfig") -> WorkQueue:

        self.processor_config = processor_config
        self.log_queue = log_queue
        self.input_queue = input_queue
        self.output_queue = WorkQueue()

        if self.is_cuda_stage() == True:
            num_proc = len(self.processor_config.cuda_devices)
        else:
            num_proc = 4#self.processor_config.max_num_proc #todo:

        workers: list[mp.Process] = []
        for rank in range(num_proc):
            worker = mp.Process(target=_process_worker, daemon=True,
                args=(self, rank, self.processor_config.cuda_devices[rank] if self.is_cuda_stage() else None))
            workers.append(worker)
            worker.start()

        self.workers = workers
        return self.output_queue

    # adds one None per worker process to the input queue to signal end of input
    # this should only be called after _ALL_ items have been added to the input queue
    # timeouts are necessary because if the main process blocks it will no longer respond to ctrl+c
    def _finish_worker_queue(self) -> None:

        sentinels_sent = 0
        while sentinels_sent < len(self.workers):
            try:
                self.input_queue.put(None, timeout=0.01)
                sentinels_sent += 1
            except QueueFullException:
                time.sleep(0.1)

        workers_terminated = 0
        while workers_terminated < len(self.workers):
            for worker in self.workers:
                if worker.exitcode is None:
                    worker.join(timeout=0.01)
                if worker.exitcode is not None:
                    workers_terminated += 1
            
            if workers_terminated < len(self.workers):
                time.sleep(0.1)

    # should only be called after _ALL_ workers have terminated due to shared memory
    def _close(self) -> None:
        for worker in self.workers:
            worker.close()

    # timeouts are necessary because if the main process blocks it will no longer respond to ctrl+c
    def _terminate(self, timeout: float = 0.1) -> None:
        for worker in self.workers:
            worker.terminate()

        if len(self.workers) > 0:
            self.workers[0].join(timeout=timeout)

        for worker in self.workers:
            worker.join(timeout=0.01)
            if worker.exitcode is None:
                worker.kill()
        
        for worker in self.workers:
            worker.join()

    def is_cuda_stage(self) -> bool:
        return False

    def start_process(self, rank: int, device: Optional[str], logger: logging.Logger) -> None:
        self.rank = rank
        self.device = device
        self.logger = logger

    def finish_process(self) -> None:
        pass

    @abstractmethod
    def process(self, input_dict: dict) -> Optional[Union[dict, list[dict]]]:
        pass

@dataclass
class DatasetProcessorConfig:

    sample_rate: int            = 32000
    num_channels: int           = 2
    target_lufs: float          = -20.
    max_num_proc: Optional[int] = None
    cuda_devices: list[str]     = ("cuda",)
    verbose: bool               = False

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
            raise ValueError("ERROR: CONFIG_PATH not defined")
        if not os.path.isdir(config.CONFIG_PATH):
            raise ValueError(f"ERROR: CONFIG_PATH '{config.CONFIG_PATH}' not found")
        if config.DATASET_PATH is None:
            raise ValueError("ERROR: DATASET_PATH not defined")
        if not os.path.isdir(config.DATASET_PATH):
            raise ValueError(f"ERROR: DATASET_PATH '{config.DATASET_PATH}' not found")
        
        self.config = DatasetProcessorConfig(
            **config.load_json(os.path.join(config.CONFIG_PATH, "dataset", "dataset.json")))    
        self.config.max_num_proc = self.config.max_num_proc or mp.cpu_count() // 2
        
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

    def process(self, process_name: str, process_stages: list[DatasetProcessStage], input: Optional[Union[str, WorkQueue]] = None) -> None:

        # start logging process
        log_queue = mp.Queue()
        log_worker = mp.Process(target=_log_worker, daemon=True,
            args=(process_name, self.config.verbose, log_queue))
        log_worker.start()

        logger = logging.getLogger("DatasetProcessor")
        logger.setLevel(logging.DEBUG if self.config.verbose == True else logging.INFO)
        log_handler = WorkerLogHandler(log_queue, "DatasetProcessor")
        logging.basicConfig(handlers=[log_handler])
        
        # first stage input is either a specified path, None (dataset path), or pre-filled WorkQueue
        if input is None or isinstance(input, str):
            scan_path = input or config.DATASET_PATH
            input_queue = WorkQueue()
        elif isinstance(input, WorkQueue):
            scan_path = None
            input_queue = input
        else:
            raise ValueError(f"Unrecognized input type in DatasetProcessor.process: {type(input)}")

        # setup processing queue chain and start workers for each stage
        input_queues: list[WorkQueue] = []
        for stage in process_stages:
            input_queues.append(input_queue)
            input_queue = stage._start_worker_queue(input_queue, log_queue, self.config)

        # start a process for monitoring progress of all stages
        stage_names = [stage.__class__.__name__ for stage in process_stages]
        monitor_finish_event = mp.Event()
        monitor_worker = mp.Process(target=_monitor_worker, daemon=True,
            args=(input_queues, stage_names, monitor_finish_event))
        monitor_worker.start()

        process_completed_successfully = True
        process_start_time = datetime.now()
        try:
            # if first stage input is a path, scan the path and fill the first input_queue
            if scan_path is not None:
                for root, _, files in os.walk(scan_path):
                    for file in files:
                        input_queues[0].put({"scan_path": scan_path, "file_path": os.path.join(root, file)})
            
            # wait for each stage to finish
            for stage in process_stages:
                stage._finish_worker_queue()

        except (KeyboardInterrupt, Exception) as e:
            process_completed_successfully = False
            logger.error("".join(format_exception(type(e), e, e.__traceback__)))

            for stage in process_stages:
                stage._terminate()

        # gracefully close monitor process
        monitor_finish_event.set()
        _terminate_worker(monitor_worker)
        monitor_worker.close()
        
        # close logging process
        process_finish_time = datetime.now(); print("")
        if process_completed_successfully == True:
            logger.info(f"Process '{process_name}' completed successfully - time elapsed: {process_finish_time - process_start_time}")
        else:
            logger.error(f"Process '{process_name}' failed - time elapsed: {process_finish_time - process_start_time}")
        
        log_handler.emit(None) # None value signals the log_worker process to gracefully end
        _terminate_worker(log_worker)
        log_worker.close()

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
        
    def filter(self) -> None:

        # todo: detect any duplicate / highly similar samples in splits,
        # detect any abnormal / anomalous samples in splits
        # if any found, prompt to remove them
        pass
    
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

        # todo: pre-encoding latents and embeddings is done in separate scripts, ideally launching them from here would be nice
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