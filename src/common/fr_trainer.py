import torch
from torch.amp import autocast, GradScaler
from src.common.utils import save_model, load_weights
from tqdm import tqdm
from .utils import get_logger
from contextlib import nullcontext

from typing import Any, Callable, Dict, Optional
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

logger = get_logger()

def fr_train_loop(
    model: torch.nn.Module,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: Optimizer,
    train_loader: DataLoader,
    config: Optional[Dict[str, Any]] = None,
    evaluation_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    scheduler: Optional[_LRScheduler] = None,
) -> None:
    """Training loop for face recognition models with gradient accumulation and mixed precision support."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)  # Wrap the model to use multiple GPUs
    model.to(device)
    use_amp = config.get('mixed_precision', True) 
    scaler = GradScaler() if use_amp else None 
    best_accuracy = 0.0
    accumulate_grad_batches = config.get('accumulate_grad_batches', 1)
    assert accumulate_grad_batches > 0, "accumulate_grad_batches should be greater than 0"

    if config and config.get('weights_path'):
        logger.info(f"Loading weights from {config['weights_path']}...")
        model = load_weights(model, config['weights_path'])

    if config and config.get("only_validate", False):
        if evaluation_fn: evaluation_fn(model, epoch=0, val_datasets=config['eval_datasets'])
        logger.info("Validation Complete.")
        return

    logger.info("Starting Training...")
    for epoch in range(config['epochs']):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)):
            images, labels = images.to(device), labels.to(device)

            with autocast(device.type) if use_amp else nullcontext():
                logits = model(images, labels) # labels for Arface head during training
                loss = criterion(logits, labels)
                loss = loss / accumulate_grad_batches

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % accumulate_grad_batches == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * accumulate_grad_batches
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        if (batch_idx + 1) % accumulate_grad_batches != 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        if scheduler:
            scheduler.step()
        
        if (epoch + 1) % config.get('val_interval', 5) == 0 and evaluation_fn:
            logger.info(f"Running external evaluation at epoch {epoch+1}...")
            result = evaluation_fn(model, epoch=epoch + 1, val_datasets=config['eval_datasets'])
            if best_accuracy < result["best_accuracy"]:
                best_accuracy = result["best_accuracy"]
                logger.info(f"New best accuracy: {best_accuracy:.2f}%, saving model...")
                save_model(model, best_accuracy, config['save_path'], config['model_name'])

        logger.info(f'Epoch [{epoch+1}/{config["epochs"]}], Loss: {running_loss/len(train_loader):.4f}, '
                    f'Accuracy: {100 * correct / total:.2f}%')

    logger.info("Training Complete.")
