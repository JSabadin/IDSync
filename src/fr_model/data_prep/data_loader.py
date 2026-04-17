from torch.utils.data import DataLoader
from .datasets import CFPDataset, LFWDataset, CALFWDataset, CPLFWDataset, WebfaceDataset, CelebADataset, AgeDB30Dataset

def get_loader(dataset_name, dataset_dir, batch_size=32, num_workers=4, augment=False, num_pairs=None, synth_dir = None):
    """
    Create a DataLoader for the specified dataset.

    Args:
        dataset_name (str): Name of the dataset ("Webface", "CFP", "LFW", "CALFW", "CPLFW").
        dataset_dir (str): Path to the dataset directory.
        batch_size (int): Number of samples per batch (default: 32).
        num_workers (int): Number of worker processes for data loading (default: 4).
        augment (bool): Apply augmentations if True (only applicable for "Webface" dataset).
        num_pairs (int): Number of pairs to generate for each dataset (only applicable for "CFP", "LFW", "CALFW", "CPLFW").
        synth_dir (str): Path to the synthetic dataset directory (only applicable for "celebA" dataset).

    Returns:
        DataLoader: A DataLoader object for the specified dataset.
    """
    if dataset_name == "webface":
        dataset = WebfaceDataset(root_dir=dataset_dir, augment=augment)
    elif dataset_name == "celeba":
        dataset = CelebADataset(real_root_dir=dataset_dir, augment=augment, synthetic_root_dir=synth_dir)
    elif dataset_name == "cfp-fp":
        dataset = CFPDataset(dataset_dir=dataset_dir, num_pairs=num_pairs)
    elif dataset_name == "lfw":
        dataset = LFWDataset(dataset_dir=dataset_dir, num_pairs=num_pairs)
    elif dataset_name == "calfw":
        dataset = CALFWDataset(dataset_dir=dataset_dir, num_pairs=num_pairs)
    elif dataset_name == "cplfw":
        dataset = CPLFWDataset(dataset_dir=dataset_dir, num_pairs=num_pairs)
    elif dataset_name == "agedb":
        dataset = AgeDB30Dataset(dataset_dir=dataset_dir, num_pairs=num_pairs)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=(dataset_name in ["webface", "celeba"]),  # Only shuffle for training dataset
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    return data_loader