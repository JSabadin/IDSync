from typing import Optional, Dict
from src.atribute_model.model import AtributeModel
from src.blocks.losses import AtributeLoss
from src.common import atribute_train_loop
from src.atribute_model.data_prep.data_loader import get_data_loaders
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


def train_atribute(config: Dict) -> None:
    """Trains an attribute prediction model based on provided configuration."""
    model = AtributeModel( # ADDING IMPROVET MODEL
        embedding_size=config['model']['embedding_size'],
        num_attributes=config['model'].get('num_attributes'),
        num_ids=config['model']['num_ids'],
        backbone=config['model']['backbone']
    )

    loss_fn = AtributeLoss(config['model']['attribute_loss_weight'], config['model']['id_loss_weight'])

    optimizer = get_optimizer(model, config['optimizer'])
    scheduler = get_scheduler(optimizer, config['scheduler']) if config.get('scheduler') else None

    data_loaders = get_data_loaders(
        dataset_type=config['trainer']['dataset']['type'],  # e.g., "celebA" or "casiawebface"
        image_dir=config['trainer']['dataset']['image_dir'],
        id_file=config['trainer']['dataset'].get('id_file'),
        attr_file=config['trainer']['dataset'].get('attr_file'),
        mapping_file=config['trainer']['dataset'].get('mapping_file'),
        batch_size=config['trainer']['batch_size'],
        num_workers=config['trainer']['num_workers']
    )

    


    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    test_loader = data_loaders['test']

    atribute_train_loop(
        model=model,
        criterion=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config['trainer'],
        attributes = config['model'].get('num_attributes')
    )


def get_optimizer(model: AtributeModel, optimizer_config: Dict) -> optim.Optimizer:
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
