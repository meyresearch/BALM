import os
import random
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import torch
import yaml
from torch import nn
from transformers import set_seed

from .configs import Configs


def setup_experiment_folder(outputs_dir: str) -> Tuple[str, str]:
    """
    Utility function to create and setup the experiment output directory.
    Return both output and checkpoint directories.

    Args:
        outputs_dir (str): The parent directory to store
            all outputs across experiments.

    Returns:
        Tuple[str, str]:
            outputs_dir: Directory of the outputs (checkpoint_dir and logs)
            checkpoint_dir: Directory of the training checkpoints
    """
    now = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    outputs_dir = os.path.join(outputs_dir, now)
    checkpoint_dir = os.path.join(outputs_dir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)

    return outputs_dir


def setup_device(device: Optional[str] = None) -> torch.device:
    """
    Utility function to setup the device for training.
    Set to the selected CUDA device(s) if exist.
    Set to "mps" if using Apple silicon chip.
    Set to "cpu" for others.

    Args:
        device (Optional[str], optional): Integer of the GPU device id,
            or str of comma-separated device ids. Defaults to None.

    Returns:
        torch.device: The chosen device(s) for the training
    """
    if torch.cuda.is_available():
        device = f"cuda:{device}"
    else:
        try:
            if torch.backends.mps.is_available():
                device = "mps"
        except:
            device = "cpu"
    return torch.device(device)


def setup_random_seed(seed: int, is_deterministic: bool = True) -> None:
    """
    Utility function to setup random seed. Apply this function early on the training script.

    Args:
        seed (int): Integer indicating the desired seed.
        is_deterministic (bool, optional): Set deterministic flag of CUDNN. Defaults to True.
    """
    # set the seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if is_deterministic is True:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    set_seed(seed)


def count_parameters(model: nn.Module) -> int:
    """
    Utility function to calculate the number of parameters in a model.

    Args:
        model (nn.Module): Model in question.

    Returns:
        int: Number of parameters of the model in question.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_yaml(filepath: str) -> dict:
    """
    Utility function to load yaml file, mainly for config files.

    Args:
        filepath (str): Path to the config file.

    Raises:
        exc: Stop process if there is a problem when loading the file.

    Returns:
        dict: Training configs.
    """
    with open(filepath, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise exc


def save_training_configs(configs: Configs, output_dir: str):
    """
    Save training config
    Args:
        configs (Configs): Configs used during training for reproducibility
        output_dir (str): Path to the output directory
    """
    filepath = os.path.join(output_dir, "configs.yaml")
    with open(filepath, "w") as file:
        _ = yaml.dump(configs.dict(), file)


def delete_files_in_directory(directory: str):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path} due to {e}")
