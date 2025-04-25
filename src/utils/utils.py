import os
import logging
import datetime
import torch
import lightning as L
import multiprocessing
from functools import partial
from tqdm import tqdm
from dataclasses import dataclass


@dataclass
class SimplePathConfig:
    """A simplified path configuration for use with Yucca split configuration."""

    train_data_dir: str

    @property
    def task_dir(self) -> str:
        """For compatibility"""
        return self.train_data_dir

    def __init__(self, train_data_dir=None):
        """Initialize with either train_data_dir or task_dir (train_data_dir has priority)."""
        self.train_data_dir = train_data_dir


def setup_seed(continue_from_most_recent=False):
    """Set up a random seed for reproducibility."""
    if not continue_from_most_recent:
        dt = datetime.datetime.now()
        seed = int(dt.strftime("%m%d%H%M%S"))
    else:
        seed = None  # Will be loaded from checkpoint if available

    L.seed_everything(seed=seed, workers=True)
    return torch.initial_seed()


def find_checkpoint(version_dir, continue_from_most_recent):
    """Find the latest checkpoint if continuing training."""
    checkpoint_path = None
    if continue_from_most_recent:
        potential_checkpoint = os.path.join(version_dir, "checkpoints", "last.ckpt")
        if os.path.isfile(potential_checkpoint):
            checkpoint_path = potential_checkpoint
            logging.info(
                "Using last checkpoint and continuing training: %s", checkpoint_path
            )
    return checkpoint_path


def load_pretrained_weights(weights_path, compile_flag):
    """Load pretrained weights with handling for compiled models and PyTorch Lightning checkpoints."""
    checkpoint = torch.load(weights_path, map_location=torch.device("cpu"))

    # Extract the state_dict from PyTorch Lightning checkpoint if needed
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        print("Loading from PyTorch Lightning checkpoint")
        state_dict = checkpoint["state_dict"]
    else:
        print("Loading from standard model checkpoint")
        state_dict = checkpoint

    # Handle compiled checkpoints when loading to uncompiled model
    if isinstance(state_dict, dict) and len(state_dict) > 0:
        first_key = next(iter(state_dict))
        if "_orig_mod" in first_key and not compile_flag:
            print("Converting compiled model weights to uncompiled format")
            uncompiled_state_dict = {}
            for key in state_dict.keys():
                new_key = key.replace("_orig_mod.", "")
                uncompiled_state_dict[new_key] = state_dict[key]
            state_dict = uncompiled_state_dict

    return state_dict


def parallel_process(process_func, tasks, num_workers=None, desc="Processing"):
    """
    Process tasks in parallel using multiprocessing.

    Args:
        process_func: Function that processes a single task
        tasks: List of tasks to process
        num_workers: Number of parallel workers (default: CPU count - 1)
        desc: Description for the progress bar

    Returns:
        List of results from processing each task
    """
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)

    print(f"Processing {len(tasks)} items using {num_workers} workers")

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(
            tqdm(pool.imap(process_func, tasks), total=len(tasks), desc=desc)
        )

    # Print results summary
    successful = sum(
        1
        for result in results
        if isinstance(result, str) and not result.startswith("Error")
    )
    print(
        f"Processing complete: {successful}/{len(tasks)} items processed successfully"
    )

    # Print any errors
    errors = [
        result
        for result in results
        if isinstance(result, str) and result.startswith("Error")
    ]
    if errors:
        print(f"Encountered {len(errors)} errors:")
        for error in errors[
            :10
        ]:  # Show only first 10 errors to avoid cluttering output
            print(f"  {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    return results
