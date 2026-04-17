import os
import torch
import logging
import re
import random
import numpy as np
import json
from typing import Any, Dict, Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

def save_model(
    model: torch.nn.Module,
    best_accuracy: float,
    save_path: str,
    model_name: str,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
) -> None:
    """
    Save the model, optimizer, and scheduler states with the model name.
    Keep only the top 3 checkpoints based on best accuracy.
    
    Args:
        model: The model to be saved.
        best_accuracy: The best accuracy achieved by the model during training.
        save_path: Path to the directory where the checkpoint will be saved.
        model_name: The name of the model to be included in the checkpoint filename.
        optimizer: The optimizer used during training (optional, if you want to save optimizer state).
        scheduler: The learning rate scheduler (optional, if you want to save scheduler state).
    """

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()

    checkpoint = {
        'best_accuracy': best_accuracy,
        'model_state_dict': model_state_dict,
    }

    if optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if scheduler:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    checkpoint_path = os.path.join(save_path, f'{model_name}_checkpoint_{best_accuracy:.4f}.pth')
    torch.save(checkpoint, checkpoint_path)

    manage_checkpoints(save_path, model_name, max_checkpoints=3)


def manage_checkpoints(
    save_path: str,
    model_name: str,
    max_checkpoints: int = 3,
) -> None:
    """
    Manage the checkpoints by keeping only the top N checkpoints with the highest accuracy.
    
    Args:
        save_path: Path where the checkpoints are saved.
        model_name: The model name to search for in the filenames.
        max_checkpoints: Maximum number of checkpoints to keep.
    """

    checkpoint_pattern = re.compile(rf"{model_name}_checkpoint_(\d+\.\d+).pth")
    
    checkpoint_files = [f for f in os.listdir(save_path) if checkpoint_pattern.match(f)]
    
    checkpoints = []
    for filename in checkpoint_files:
        match = checkpoint_pattern.match(filename)
        if match:
            accuracy = float(match.group(1))  
            file_path = os.path.join(save_path, filename)
            checkpoints.append((accuracy, file_path))
    
    checkpoints = sorted(checkpoints, key=lambda x: x[0], reverse=True)
    
    if len(checkpoints) > max_checkpoints:
        for accuracy, file_path in checkpoints[max_checkpoints:]:
            os.remove(file_path)


def load_weights(
    model: torch.nn.Module,
    weights_path: str,
) -> torch.nn.Module:
    """
    Load model weights from a checkpoint file.

    Args:
        model: The model to load weights into.
        weights_path (str): Path to the model checkpoint file.

    Returns:
        model: The model with loaded weights.
    """
    checkpoint = torch.load(weights_path)
    
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    return model

def get_logger(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("app_logger")
    if not logger.hasHandlers():
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(stream_handler)
        
    logger.setLevel(level)
    return logger



def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for Python, NumPy, and PyTorch (both CPU and GPU).
    
    Args:
        seed (int): The seed value to use for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_metrics(
    metrics: Dict[str, Any],
    metrics_path: str,
) -> None:
    """
    Save the given metrics to the specified path, overwriting any existing data.

    Args:
        metrics (dict): Dictionary containing the metrics to save.
        metrics_path (str): Path to the file where metrics should be saved.
    """
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

