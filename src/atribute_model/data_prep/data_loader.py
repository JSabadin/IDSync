from torch.utils.data import DataLoader
from src.atribute_model.data_prep.dataset import CelebAMultitaskDataset, WebfaceDataset05, WebfaceDataset21

def get_data_loaders(
    dataset_type: str,
    image_dir: str,
    id_file: str = None,
    attr_file: str = None,
    mapping_file: str = None,
    batch_size: int = 32,
    num_workers: int = 8
):
    """
    Creates DataLoaders for train, val, and test splits for supported datasets.

    Args:
        dataset_type (str): Type of dataset ("celebA" or "casiawebface").
        image_dir (str): Path to the dataset directory containing images.
        id_file (str, optional): Path to the file containing ID labels (for CelebA only).
        attr_file (str, optional): Path to the file containing attribute labels (for CelebA only).
        batch_size (int): Number of samples per batch. Default is 32.
        num_workers (int): Number of subprocesses to use for data loading. Default is 8.

    Returns:
        dict: A dictionary containing DataLoaders for "train", "val", and "test".
    """
    if dataset_type not in {"celebA", "webface", "webface21"}:
        raise ValueError(f"Unsupported dataset type: {dataset_type}. Use 'celebA', 'webface', or 'webface21'.")
    
    splits = ["train", "val", "test"]
    loaders = {}

    if dataset_type == "celebA":
        if not (id_file and attr_file):
            raise ValueError("For CelebA, 'id_file' and 'attr_file' are required.")
        dataset_class = CelebAMultitaskDataset
        dataset_args = {"image_dir": image_dir, "id_file": id_file, "attr_file": attr_file}
    elif dataset_type == "webface05":
        dataset_class = WebfaceDataset05
        dataset_args = {"root_dir": image_dir}
    elif dataset_type == "webface21":
        dataset_class = WebfaceDataset21
        dataset_args = {"root_dir": image_dir, "mapping_file": mapping_file}

    for split in splits:
        dataset = dataset_class(**dataset_args, split=split)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            persistent_workers=True,
            pin_memory=True
        )
    
    return loaders
