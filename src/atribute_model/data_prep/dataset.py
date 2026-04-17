import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from src.atribute_model.data_prep.augmentations import get_transforms, get_augmentations	
from collections import defaultdict
import random
import json
from typing import Any, Dict, List, Optional, Tuple, Union

from src.common.utils import get_logger


logger = get_logger()

class CelebAMultitaskDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        id_file: str,
        attr_file: str,
        split: str = "train",
        seed: int = 42,
        split_path: Optional[str] = None,
        augment: bool = True,
    ) -> None:
        """
        Args:
            image_dir (str): Path to the directory containing images.
            id_file (str): Path to the file containing ID labels.
            attr_file (str): Path to the file containing attribute labels.
            split (str): The subset to load ("train", "val", "test").
            seed (int): Random seed for reproducibility of splits.
        """
        self.image_dir = image_dir
        self.id_labels = self._load_id_labels(id_file)
        self.attr_labels, self.attr_names = self._load_attr_labels(attr_file)
        self.split = split
        self.transform = get_transforms(use_center_crop=True)
        self.augmentations = get_augmentations() if augment and split == "train" else None 
        self.n_skiped_ids = 0

        self.image_files = self._create_split(split, seed, split_path=split_path)

    def _load_id_labels(self, id_file):
        """Load ID labels from the ID file."""
        id_labels = {}
        with open(id_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                image_file, id_label = parts[0], int(parts[1])
                id_labels[image_file] = id_label - 1  # Subtract 1 to make IDs 0-indexed
        return id_labels

    def _load_attr_labels(self, attr_file):
        """Load attribute labels from the attribute file."""
        with open(attr_file, "r") as f:
            lines = f.readlines()
            attr_names = lines[1].strip().split()
            attr_labels = {}
            for line in lines[2:]:
                parts = line.strip().split()
                image_file = parts[0]
                labels = torch.tensor([(int(label) + 1) // 2 for label in parts[1:]], dtype=torch.float32)  # Transform -1/1 to 0/1
                attr_labels[image_file] = labels
        return attr_labels, attr_names

    def print_images_by_id(self, id_label):
        """Prints image filenames associated with a specific ID."""
        for image_file, current_id in self.id_labels.items():
            if current_id == id_label:
                print(image_file)

    def _create_split(self, split, seed, split_path=None):
        """Create splits where each ID has one image in val, one in test, and the rest in train.
        """
        random.seed(seed)
        id_to_images = defaultdict(list)

        for image_file, id_label in self.id_labels.items():
            id_to_images[id_label].append(image_file)

        valid_id_to_images = {}
        id_mapping = {}
        new_id = 0 

        for id_label, image_list in sorted(id_to_images.items()):
            if len(image_list) < 10:
                self.n_skiped_ids += 1
            else:
                valid_id_to_images[new_id] = image_list
                id_mapping[id_label] = new_id
                new_id += 1

        self.id_labels = {
            image_file: id_mapping[old_id]
            for image_file, old_id in self.id_labels.items()
            if old_id in id_mapping
        }

        train_files, val_files, test_files = [], [], []
        split_mapping = {}
        for id_label, image_list in valid_id_to_images.items():
            random.shuffle(image_list)
            val_files.append(image_list[0])  # First image for val
            test_files.append(image_list[1])  # Second image for test
            train_files.extend(image_list[2:])  # Remaining images for train

            if split_path is not None:
                split_mapping[id_label] = {
                    "train": image_list[2:],
                    "val": [image_list[0]],
                    "test": [image_list[1]],
                }

        logger.info(f"Number of skipped IDs: {self.n_skiped_ids}")
        logger.info(f"Total valid IDs: {len(valid_id_to_images)}")

        if split_path is not None:
            with open(split_path, "w") as json_file:
                json.dump(split_mapping, json_file, indent=4)
            logger.info(f"Split mapping saved to {split_path}")

        if split == "train":
            return train_files
        elif split == "val":
            return val_files
        elif split == "test":
            return test_files
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Return an image, its ID label, and its attribute labels."""
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.augmentations:
            image = self.augmentations(image)
        id_label = self.id_labels[image_file]
        attr_label = self.attr_labels[image_file]
        
        return image, id_label, attr_label


class WebfaceDataset05(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        augment: bool = True,
    ) -> None:
        """
        Args:
            root_dir (str): Path to the root directory of the Webface dataset.
            split (str): Either "train" or "val" to determine the split.
            augment (bool): Apply augmentations if True.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = get_transforms(use_center_crop=False) # No center crop for Webface
        self.augmentations = get_augmentations() if augment and split == "train" else None
        self.classes = sorted(
            [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))], 
            key=lambda x: int(x)  # Convert to integer for numeric sort
        )
        self.image_paths = []
        self.labels = []

        for idx, class_dir in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_path):
                images = [
                    os.path.join(class_path, img_name)
                    for img_name in os.listdir(class_path)
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]
                if len(images) == 0:
                    continue
                if split == "train":
                    self.image_paths.extend(images[:-1])
                    self.labels.extend([idx] * (len(images) - 1))
                elif split == "val" or split == "test":
                    self.image_paths.append(images[-1])
                    self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        image = self.transform(image)

        if self.augmentations:
            image = self.augmentations(image)

        return image, label


class WebfaceDataset21(Dataset):
    def __init__(
        self,
        root_dir: str,
        mapping_file: str,
        split: str = "train",
        augment: bool = True,
    ) -> None:
        """
        Args:
            root_dir (str): Path to the root directory of the Webface dataset.
            split (str): Either "train" or "val" to determine the split.
            augment (bool): Apply augmentations if True.
            mapping_file (str): JSON file containing folder to class mappings, since the dataset is filtered to obtain just the ID's with more than 10 images.
        """
        import json
        self.root_dir = root_dir
        self.split = split
        self.transform = get_transforms(use_center_crop=False)
        self.augmentations = get_augmentations() if augment and split == "train" else None

        with open(mapping_file, 'r') as json_file:
            self.class_mapping = json.load(json_file)

        self.image_paths = []
        self.labels = []
        for cls_name, cls_idx in self.class_mapping.items():
            folder_path = os.path.join(root_dir, cls_name)
            images = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            if len(images) >= 10:
                if split == "train":
                    self.image_paths.extend(images[:-1])
                    self.labels.extend([cls_idx] * (len(images) - 1))
                elif split == "val":
                    self.image_paths.append(images[-1])
                    self.labels.append(cls_idx)

        logger.info(f"Number of classes: {len(self.class_mapping)}")
        logger.info(f"Number of images in {split} split: {len(self.image_paths)}")


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.augmentations:
            image = self.augmentations(image)
        label = self.labels[idx]
        return image, label