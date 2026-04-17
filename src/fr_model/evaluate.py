
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, accuracy_score
from sklearn.model_selection import KFold
from .data_prep.data_loader import get_loader

from src.common.utils import get_logger

logger = get_logger()

def evaluate_fr(model, epoch, val_datasets, n_splits=10):
    """
    Perform 10-fold cross-validation on face verification datasets.

    Args:
        model         : a PyTorch model that maps images to embeddings
        epoch         : current epoch number (for logging / RNG seed)
        val_datasets  : dict mapping dataset names to {"path":…, "num_pairs":…}
        n_splits      : number of CV folds (default 10)

    Returns:
        dict with key "best_accuracy" giving the mean accuracy across datasets
    """
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    all_dataset_accs = []

    for dataset_name, info in val_datasets.items():
        logger.info(f"Starting 10-fold CV on {dataset_name.upper()} at epoch {epoch}")
        # 1) collect all similarities & labels
        loader = get_loader(
            dataset_name=dataset_name,
            dataset_dir=info["path"],
            num_pairs=info["num_pairs"],
            batch_size=64,
            num_workers=8,
            augment=False
        )

        sims, labs = [], []
        with torch.no_grad():
            for (img1, img2), labels in tqdm(loader, desc=f"Embedding {dataset_name}"):
                img1, img2 = img1.to(device), img2.to(device)
                e1 = model(img1)
                e2 = model(img2)
                sim = 1.0 / (euclidean_distance(e1, e2) + 1e-8)
                sims.extend(sim.cpu().numpy())
                labs.extend(labels.cpu().numpy())

        sims = np.array(sims)
        labs = np.array(labs)

        # 2) set up K-fold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=epoch)
        fold_accs = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(sims), start=1):
            train_s, train_l = sims[train_idx], labs[train_idx]
            test_s,  test_l  = sims[test_idx],  labs[test_idx]

            # 3a) find best threshold on training folds
            fpr_t, tpr_t, thr_t = roc_curve(train_l, train_s)
            best_acc, best_thr = 0.0, 0.0
            for thr in thr_t:
                preds = (train_s > thr).astype(int)
                acc = accuracy_score(train_l, preds)
                if acc > best_acc:
                    best_acc, best_thr = acc, thr

            # 3b) evaluate on held-out fold
            test_preds = (test_s > best_thr).astype(int)
            acc_test = accuracy_score(test_l, test_preds)
            fold_accs.append(acc_test)
            logger.info(f"[{dataset_name.upper()}] Fold {fold_idx}: acc={acc_test*100:.2f}%, thr={best_thr:.4f}")

        # 4) average across folds
        mean_acc = np.mean(fold_accs)
        logger.info(f"[{dataset_name.upper()}] 10-fold mean accuracy: {mean_acc*100:.2f}%\n")
        all_dataset_accs.append(mean_acc)

    overall_mean = np.mean(all_dataset_accs)
    return {"best_accuracy": overall_mean}


def euclidean_distance(embedding1, embedding2):
    """
    Compute the Euclidean distance between two sets of embeddings.
    
    Args:
        embedding1 (torch.Tensor): Tensor of shape (batch_size, embedding_dim)
        embedding2 (torch.Tensor): Tensor of shape (batch_size, embedding_dim)

    Returns:
        torch.Tensor: Euclidean distance for each pair in the batch
    """
    return torch.norm(embedding1 - embedding2, dim=1)
