from typing import Optional, Dict
from src.fr_model.model import FRModel
from src.blocks.losses import FRLoss
from src.common import fr_train_loop
from src.fr_model.data_prep.data_loader import get_loader
from .evaluate import evaluate_fr
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


def train_fr(fr_config: Dict) -> None:
    """Trains a face recognition model based on provided configuration."""
    model = FRModel(
        embedding_size=fr_config['model']['embedding_size'],
        num_classes=fr_config['model']['num_classes'],
        backbone=fr_config['model']['backbone'],
        head=fr_config['model']['head']
    )
    loss_fn = FRLoss()
    optimizer = get_optimizer(model, fr_config['optimizer'])
    scheduler = get_scheduler(optimizer, fr_config['scheduler']) if fr_config.get('scheduler') else None

    train_loader = get_loader(
        dataset_name=fr_config['trainer']['train_dataset']['dataset_name'],
        dataset_dir=fr_config['trainer']['train_dataset']['path'],
        batch_size=fr_config['trainer']['batch_size'],
        augment=fr_config['trainer']['train_dataset']['augment'],
        num_workers=fr_config['trainer']['num_workers'],
        synth_dir=fr_config['trainer']['train_dataset'].get('synth_dir', None)
    )

    fr_train_loop(
        model=model,
        criterion=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        config=fr_config['trainer'],
        evaluation_fn=evaluate_fr
    )


def get_optimizer(model: FRModel, optimizer_config: Dict) -> optim.Optimizer:
    """Initializes the optimizer based on configuration."""
    optimizer_class = getattr(optim, optimizer_config['type'], None)
    if optimizer_class:
        return optimizer_class(model.parameters(), **optimizer_config['params'])
    raise ValueError(f"Unsupported optimizer type: {optimizer_config['type']}")


def get_scheduler(optimizer: optim.Optimizer, scheduler_config: Dict) -> Optional[lr_scheduler._LRScheduler]:
    """Initializes the learning rate scheduler based on configuration."""
    scheduler_class = getattr(lr_scheduler, scheduler_config['type'], None)
    if scheduler_class:
        return scheduler_class(optimizer, **scheduler_config['params'])
    raise ValueError(f"Unsupported scheduler type: {scheduler_config['type']}")
