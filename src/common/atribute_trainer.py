import torch
from contextlib import nullcontext
from torch.amp import autocast, GradScaler
from .utils import get_logger
from src.common.utils import save_model, load_weights, save_metrics
from tqdm import tqdm 
from typing import Any, Dict, Tuple, Optional, Union, Callable
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
logger = get_logger()

def atribute_train_loop(
    model: torch.nn.Module,
    criterion: Callable[..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    optimizer: Optimizer,
    scheduler: Union[_LRScheduler, ReduceLROnPlateau],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: Dict[str, Any],
    attributes: Optional[int] = None,
) -> None:
    """
    Main training loop for the attribute prediction model.

    Args:
        model (torch.nn.Module): The model to be trained.
        criterion (Callable): Loss function.
        optimizer (Optimizer): Optimizer for training.
        scheduler (Union[_LRScheduler, ReduceLROnPlateau]): Learning rate scheduler.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        test_loader (DataLoader): DataLoader for test data.
        config (Dict[str, Any]): Configuration dictionary containing training parameters.
        attributes (Optional[int]): Number of attributes. If None, only ID training is performed.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    task = "all" if attributes else "ids"

    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    if config and config.get('weights_path'):
        logger.info(f"Loading weights from {config['weights_path']}...")
        model = load_weights(model, config['weights_path'])

    if config and config.get("only_validate", False):
        logger.info("Validation-only mode enabled.")
        _, metrics = evaluate(model, criterion, val_loader, device, config, task, mode="Validation")
        save_metrics(metrics, config['metrics_path'])
        return

    if config and config.get("only_test", False):
        logger.info("Test-only mode enabled.")
        _, metrics = evaluate(model, criterion, test_loader, device, config, task, mode="Test")
        save_metrics(metrics, config['metrics_path'])
        return

    num_epochs = config['epochs']
    val_interval = config.get('val_interval', 1)
    accumulate_grad_batches = config.get('accumulate_grad_batches', 1)
    use_amp = config.get('mixed_precision', True)

    scaler = GradScaler(enabled=use_amp)
    best_id_accuracy = 0.0
    best_attr_accuracy = 0.0
    metrics = {"train_loss": [], "val_loss": [], "val_id_accuracy": [], "val_attr_accuracy": [], "train_id_accuracy": [], "train_attr_accuracy": []}
    logger.info("Starting Training...")

    for epoch in range(num_epochs):
        model.train()
        train_loss, total_attr_loss, total_id_loss = 0.0, 0.0, 0.0
        total_attr_correct, total_id_correct = 0, 0
        total_attr_elements, total_id_elements = 0, 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)):
            if task == "ids":
                images, id_labels = batch
                attr_labels = None  # Placeholder for consistency
            else:
                images, id_labels, attr_labels = batch

            images, id_labels = images.to(device), id_labels.to(device)
            if attr_labels is not None:
                attr_labels = attr_labels.to(device)

            with autocast(device.type) if use_amp else nullcontext():
                if task == "ids":
                    id_preds = model(images, task=task)
                    attr_preds = None  # No attribute predictions in ID-only task
                    loss, _, id_loss = criterion(None, None, id_preds, id_labels)
                else:
                    attr_preds, id_preds = model(images, task=task)
                    loss, attr_loss, id_loss = criterion(attr_preds, attr_labels, id_preds, id_labels)

                loss = loss / accumulate_grad_batches

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulate_grad_batches == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * accumulate_grad_batches
            if task != "ids":
                total_attr_loss += attr_loss.item()
                attr_preds_binary = (torch.sigmoid(attr_preds) > 0.5).float()
                total_attr_correct += (attr_preds_binary == attr_labels).sum().item()
                total_attr_elements += torch.numel(attr_labels)

            id_preds_class = id_preds.argmax(dim=1)
            total_id_correct += (id_preds_class == id_labels).sum().item()
            total_id_elements += id_labels.size(0)

        train_id_accuracy = total_id_correct / total_id_elements
        metrics["train_loss"].append(train_loss / len(train_loader))
        metrics["train_id_accuracy"].append(train_id_accuracy)

        if task != "ids":
            train_attr_accuracy = total_attr_correct / total_attr_elements
            metrics["train_attr_accuracy"].append(train_attr_accuracy)
            logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Train Attr Accuracy: {train_attr_accuracy:.4f}")

        logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss / len(train_loader):.4f}, Train ID Accuracy: {train_id_accuracy:.4f}")

        if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

        if (epoch + 1) % val_interval == 0:
            val_loss, val_metrics = evaluate(model, criterion, val_loader, device, config, task, mode="Validation")
            metrics["val_loss"].append(val_loss)
            metrics["val_id_accuracy"].append(val_metrics["id_accuracy"])

            if task != "ids":
                metrics["val_attr_accuracy"].append(val_metrics["attr_accuracy"])

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["id_accuracy"])

            if val_metrics["id_accuracy"] > best_id_accuracy:
                best_id_accuracy = max(best_id_accuracy, val_metrics["id_accuracy"])
                if task != "ids":
                    best_attr_accuracy = max(best_attr_accuracy, val_metrics["attr_accuracy"])
                save_model(model, best_id_accuracy, config['save_path'], config['model_name'])

    save_metrics(metrics, config['metrics_path'])

def evaluate(
    model: torch.nn.Module,
    criterion: Callable[..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
    task: str,
    mode: str = "Validation",
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluates the model on the given dataset.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        criterion (Callable): Loss function.
        loader (DataLoader): DataLoader for the dataset.
        device (torch.device): Device to perform computations on.
        config (Dict[str, Any]): Configuration dictionary containing evaluation parameters.
        task (str): Task type ("ids" or "all").
        mode (str): Evaluation mode ("Validation" or "Test").
    """
    model.eval()
    loss, attr_loss, id_loss = 0.0, 0.0, 0.0
    total_attr_correct, total_id_correct = 0, 0
    total_attr_elements, total_id_elements = 0, 0
    use_amp = config.get('mixed_precision', True)

    with torch.no_grad():
        for batch in loader:
            if task == "ids":
                images, id_labels = batch
                attr_labels = None
            else:
                images, id_labels, attr_labels = batch

            images, id_labels = images.to(device), id_labels.to(device)
            if attr_labels is not None:
                attr_labels = attr_labels.to(device)

            with autocast(device.type) if use_amp else nullcontext():
                if task == "ids":
                    id_preds = model(images, task=task)
                    attr_preds = None
                    batch_loss, _, batch_id_loss = criterion(None, None, id_preds, id_labels)
                else:
                    attr_preds, id_preds = model(images, task=task)
                    batch_loss, batch_attr_loss, batch_id_loss = criterion(attr_preds, attr_labels, id_preds, id_labels)

            loss += batch_loss.item()
            id_loss += batch_id_loss.item()

            if task != "ids":
                attr_loss += batch_attr_loss.item()
                attr_preds_binary = (torch.sigmoid(attr_preds) > 0.5).float()
                total_attr_correct += (attr_preds_binary == attr_labels).sum().item()
                total_attr_elements += torch.numel(attr_labels)

            id_preds_class = id_preds.argmax(dim=1)
            total_id_correct += (id_preds_class == id_labels).sum().item()
            total_id_elements += id_labels.size(0)

    attr_accuracy = total_attr_correct / total_attr_elements if task != "ids" else 0.0
    id_accuracy = total_id_correct / total_id_elements

    logger.info(f"{mode} - Loss: {loss / len(loader):.4f}, ID Accuracy: {id_accuracy:.4f}, Attr Accuracy: {attr_accuracy:.4f}")

    metrics = {"loss": loss / len(loader), "attr_accuracy": attr_accuracy, "id_accuracy": id_accuracy}
    return loss / len(loader), metrics
