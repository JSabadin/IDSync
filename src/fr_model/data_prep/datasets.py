import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import random
from .augmentations import get_transforms, get_augmentations
import numpy as np

class BaseDataset(Dataset):
    def __init__(self, dataset_dir, image_extension=".jpg", num_pairs=12000, seed=141):
        """
        Args:
            dataset_dir (str): Path to the dataset directory.
            image_extension (str): Image file extension (default: ".jpg").
            num_pairs (int): Number of verification pairs to create (default: 12000).
            seed (int): Seed for reproducibility (default: 42).
        """
        self.dataset_dir = dataset_dir
        self.image_extension = image_extension
        self.transforms = get_transforms()
        self._set_seed(seed)
        self.images_by_person = self._load_image_paths()
        self.pairs, self.labels = self._create_pairs(num_pairs)

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path = self.pairs[idx]
        label = self.labels[idx]
        img1, img2 = Image.open(img1_path).convert('RGB'), Image.open(img2_path).convert('RGB')

        img1, img2 = self.transforms(img1), self.transforms(img2)

        return (img1, img2), torch.tensor(label, dtype=torch.float32)

    def _load_image_paths(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def _create_pairs(self, num_pairs):
        raise NotImplementedError("This method should be overridden by subclasses.")



class AgeDB30Dataset(BaseDataset):
    def __init__(self, dataset_dir, image_extension=".jpg", num_pairs=6000):
        super().__init__(dataset_dir, image_extension, num_pairs)

    def _load_image_paths(self):
        images = {}
        ages = {}
        
        for img_file in sorted(os.listdir(self.dataset_dir)):
            if img_file.endswith(self.image_extension):
                parts = img_file.split('_')
                age = int(parts[0])
                person_name = parts[1]
                img_path = os.path.join(self.dataset_dir, img_file)
                
                if person_name not in images:
                    images[person_name] = []
                    ages[person_name] = []
                images[person_name].append(img_path)
                ages[person_name].append(age)
        
        self.images_by_person = images
        self.ages_by_person = ages
        return images

    def _create_pairs(self, num_pairs):
        people = list(self.images_by_person.keys())
        pairs, labels = [], []
        half_pairs = num_pairs // 2
        
        for _ in range(half_pairs//2):
            person = random.choice(people)
            if len(self.images_by_person[person]) < 2:
                continue
            
            sorted_images = sorted(zip(self.images_by_person[person], self.ages_by_person[person]), key=lambda x: x[1])
            
            for i in range(len(sorted_images)):
                for j in range(i + 1, len(sorted_images)):
                    age_diff = abs(sorted_images[j][1] - sorted_images[i][1])
                    
                    if age_diff < 30:
                        pairs.append((sorted_images[i][0], sorted_images[j][0]))
                        labels.append(1)
                        break
                if len(pairs) >= half_pairs:
                    break
            
            if len(pairs) >= half_pairs:
                break
        
        # Limit to half_pairs
        pairs = pairs[:half_pairs]
        labels = labels[:half_pairs]
        
        for _ in range(half_pairs):
            person1, person2 = random.sample(people, 2)
            img1 = random.choice(self.images_by_person[person1])
            img2 = random.choice(self.images_by_person[person2])
            pairs.append((img1, img2))
            labels.append(0)
        
        return pairs, labels


class CFPDataset(BaseDataset):
    def __init__(self, dataset_dir, image_extension=".jpg", num_pairs=12000):
        """
        Args:
            dataset_dir (str): Path to the dataset directory.
            image_extension (str): Image file extension (default: ".jpg").
            num_pairs (int): Number of verification pairs to create (default: 12000).
        """
        super().__init__(dataset_dir, image_extension, num_pairs)

    def _load_image_paths(self):
        """Load images organized by person with frontal and profile subdirectories."""
        images = {}
        for person in sorted(os.listdir(self.dataset_dir)):
            person_dir = os.path.join(self.dataset_dir, person)
            if not os.path.isdir(person_dir):
                continue
            frontal = [os.path.join(person_dir, 'frontal', img) for img in os.listdir(os.path.join(person_dir, 'frontal')) if img.endswith(self.image_extension)]
            profile = [os.path.join(person_dir, 'profile', img) for img in os.listdir(os.path.join(person_dir, 'profile')) if img.endswith(self.image_extension)]
            if frontal and profile:
                images[person] = {'frontal': frontal, 'profile': profile}
        return images

    def _create_pairs(self, num_pairs):
        """Create equal numbers of positive and negative pairs."""
        people = list(self.images_by_person.keys())
        pairs, labels = [], []
        half_pairs = num_pairs // 2

        for _ in range(half_pairs):
            person = random.choice(people)
            frontal, profile = self.images_by_person[person]['frontal'], self.images_by_person[person]['profile']
            if random.choice([True, False]):
                pairs.append((random.choice(frontal), random.choice(profile)))
            else:
                img1, img2 = random.sample(frontal + profile, 2)
                pairs.append((img1, img2))
            labels.append(1)

        for _ in range(half_pairs):
            person1, person2 = random.sample(people, 2)
            pairs.append((random.choice(self.images_by_person[person1]['frontal']), random.choice(self.images_by_person[person2]['profile'])))
            labels.append(0)

        return pairs, labels



class LFWDataset(BaseDataset):
    def __init__(self, dataset_dir, image_extension=".jpg", num_pairs=6000):
        """
        Args:
            dataset_dir (str): Path to the dataset directory.
            image_extension (str): Image file extension (default: ".jpg").
            num_pairs (int): Number of verification pairs to create (default: 6000).
        """
        super().__init__(dataset_dir, image_extension, num_pairs)

    def _load_image_paths(self):
        """Load images directly from each person's folder."""
        images = {}
        for person in sorted(os.listdir(self.dataset_dir)):
            person_dir = os.path.join(self.dataset_dir, person)
            if not os.path.isdir(person_dir):
                continue
            image_files = [os.path.join(person_dir, img) for img in os.listdir(person_dir) if img.endswith(self.image_extension)]
            if image_files:
                images[person] = image_files
        return images

    def _create_pairs(self, num_pairs):
        """Create pairs randomly from available images of each person."""
        people = list(self.images_by_person.keys())
        pairs, labels = [], []
        half_pairs = num_pairs // 2

        # Creating positive pairs
        for _ in range(half_pairs):
            person = random.choice(people)
            if len(self.images_by_person[person]) >= 2:
                img1, img2 = random.sample(self.images_by_person[person], 2)
                pairs.append((img1, img2))
                labels.append(1)

        # Creating negative pairs
        for _ in range(half_pairs):
            person1, person2 = random.sample(people, 2)
            img1 = random.choice(self.images_by_person[person1])
            img2 = random.choice(self.images_by_person[person2])
            pairs.append((img1, img2))
            labels.append(0)
        
        return pairs, labels


class CALFWDataset(BaseDataset):
    def __init__(self, dataset_dir, image_extension=".jpg", num_pairs=6000):
        """
        Args:
            dataset_dir (str): Path to the dataset directory.
            image_extension (str): Image file extension (default: ".jpg").
            num_pairs (int): Number of verification pairs to create (default: 12000).
        """
        super().__init__(dataset_dir, image_extension, num_pairs)

    def _load_image_paths(self):
        """Load images and categorize them by extracted person ID from filenames."""
        images = {}
        for img_filename in os.listdir(self.dataset_dir):
            if img_filename.endswith(self.image_extension):
                person_id = img_filename.rsplit('_', 1)[0]  # Split on last underscore to isolate ID
                if person_id not in images:
                    images[person_id] = []
                images[person_id].append(os.path.join(self.dataset_dir, img_filename))
        return images

    def _create_pairs(self, num_pairs):
        """Create pairs randomly from available images of each person."""
        people = list(self.images_by_person.keys())
        pairs, labels = [], []
        half_pairs = num_pairs // 2

        # Creating positive pairs
        for _ in range(half_pairs):
            person = random.choice(people)
            if len(self.images_by_person[person]) >= 2:
                img1, img2 = random.sample(self.images_by_person[person], 2)
                pairs.append((img1, img2))
                labels.append(1)

        # Creating negative pairs
        for _ in range(half_pairs):
            person1, person2 = random.sample(people, 2)
            img1 = random.choice(self.images_by_person[person1])
            img2 = random.choice(self.images_by_person[person2])
            pairs.append((img1, img2))
            labels.append(0)

        return pairs, labels


class CPLFWDataset(BaseDataset):
    def __init__(self, dataset_dir, image_extension=".jpg", num_pairs=6000):
        """
        Args:
            dataset_dir (str): Path to the dataset directory.
            image_extension (str): Image file extension (default: ".jpg").
            num_pairs (int): Number of verification pairs to create (default: 12000).
        """
        super().__init__(dataset_dir, image_extension, num_pairs)

    def _load_image_paths(self):
        """Load images and categorize them by extracted person ID from filenames."""
        images = {}
        for img_filename in os.listdir(self.dataset_dir):
            if img_filename.endswith(self.image_extension):
                person_id = img_filename.rsplit('_', 1)[0]  # Split on last underscore to isolate ID
                if person_id not in images:
                    images[person_id] = []
                images[person_id].append(os.path.join(self.dataset_dir, img_filename))
        return images

    def _create_pairs(self, num_pairs):
        """Create pairs randomly from available images of each person."""
        people = list(self.images_by_person.keys())
        pairs, labels = [], []
        half_pairs = num_pairs // 2

        # Creating positive pairs
        for _ in range(half_pairs):
            person = random.choice(people)
            if len(self.images_by_person[person]) >= 2:
                img1, img2 = random.sample(self.images_by_person[person], 2)
                pairs.append((img1, img2))
                labels.append(1)

        # Creating negative pairs
        for _ in range(half_pairs):
            person1, person2 = random.sample(people, 2)
            img1 = random.choice(self.images_by_person[person1])
            img2 = random.choice(self.images_by_person[person2])
            pairs.append((img1, img2))
            labels.append(0)

        return pairs, labels

    def __len__(self):
        return len(self.pairs)




class WebfaceDataset(Dataset):
    def __init__(self, root_dir, augment=False):
        """
        Args:
            root_dir (str): Path to the root directory of the Webface dataset.
            transform (callable, optional): Transformations to be applied on a sample.
            augment (bool): Apply augmentations if True.
        """
        self.root_dir = root_dir
        self.transform = get_transforms()
        self.augmentations = get_augmentations() if augment else None
        self.classes = os.listdir(root_dir)  # Each subfolder corresponds to a unique identity
        self.image_paths = []
        self.labels = []

        for idx, class_dir in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(img_path)
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
    

class CelebADataset(Dataset):
    def __init__(self, real_root_dir=None, synthetic_root_dir=None, augment=True):
        """
        Args:
            real_root_dir (str, optional): Path to the root directory of the real faces. Default is None.
            synthetic_root_dir (str, optional): Path to the root directory of the synthetic faces. Default is None.
            augment (bool): Apply augmentations if True.
        """
        self.real_root_dir = real_root_dir
        self.synthetic_root_dir = synthetic_root_dir
        self.transform = get_transforms()
        self.augmentations = get_augmentations() if augment else None

        self.real_image_paths, self.real_labels = [], []
        if real_root_dir:
            self.real_image_paths, self.real_labels = self._load_data(real_root_dir, max_classes=100)

        self.synthetic_image_paths, self.synthetic_labels = [], []
        if synthetic_root_dir:
            self.synthetic_image_paths, self.synthetic_labels = self._load_data(synthetic_root_dir)

        self.image_paths = self.real_image_paths + self.synthetic_image_paths
        self.labels = self.real_labels + self.synthetic_labels

        if not self.image_paths:
            raise ValueError("At least one of real_root_dir or synthetic_root_dir must be provided and contain data.")

    def _load_data(self, root_dir, max_classes=None):
        """Helper function to load image paths and labels from the given directory."""
        image_paths = []
        labels = []
        classes = sorted(os.listdir(root_dir))  # Ensure consistent order of IDs
        if max_classes is not None:
            classes = classes[:max_classes]

        for idx, class_dir in enumerate(classes):
            class_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(img_path)
                        labels.append(idx)

        return image_paths, labels

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



# Or you can use bin files for datasets from Insightface.

# class AgeDB30Dataset(BaseDataset):
#     def __init__(self, dataset_dir, image_extension=".jpg", num_pairs=6000):
#         super().__init__(dataset_dir, image_extension=image_extension, num_pairs=num_pairs)

#     def _load_image_paths(self):
#         bin_path = os.path.join(self.dataset_dir, "agedb_30.bin")
#         with open(bin_path, 'rb') as f:
#             self.bins, self.issame_list = pickle.load(f, encoding='bytes')

#         self.images = []
#         for b in self.bins:
#             img_data = np.frombuffer(b, dtype=np.uint8)
#             img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
#             self.images.append(img)

#         return {}

#     def _create_pairs(self, num_pairs):
#         pairs, labels = [], []
#         for idx, same in enumerate(self.issame_list):
#             if (2 * idx + 1) < len(self.images):
#                 pairs.append((2 * idx, 2 * idx + 1))
#                 labels.append(int(same))
#             if len(pairs) >= num_pairs:
#                 break
#         return pairs, labels

#     def __getitem__(self, idx):
#         idx1, idx2 = self.pairs[idx]
#         label = self.labels[idx]
#         # Convert from OpenCV BGR to PIL RGB
#         img1 = Image.fromarray(cv2.cvtColor(self.images[idx1], cv2.COLOR_BGR2RGB))
#         img2 = Image.fromarray(cv2.cvtColor(self.images[idx2], cv2.COLOR_BGR2RGB))
#         # Apply transforms
#         img1 = self.transforms(img1)
#         img2 = self.transforms(img2)
#         return (img1, img2), torch.tensor(label, dtype=torch.float32)
    
# class CFPDataset(BaseDataset):
#     def __init__(self, dataset_dir, image_extension=".jpg", num_pairs=6000):
#         super().__init__(dataset_dir, image_extension=image_extension, num_pairs=num_pairs)

#     def _load_image_paths(self):
#         bin_path = os.path.join(self.dataset_dir, "cfp_fp.bin")
#         with open(bin_path, 'rb') as f:
#             self.bins, self.issame_list = pickle.load(f, encoding='bytes')

#         self.images = []
#         for b in self.bins:
#             img_data = np.frombuffer(b, dtype=np.uint8)
#             img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
#             self.images.append(img)

#         return {}

#     def _create_pairs(self, num_pairs):
#         pairs, labels = [], []
#         for idx, same in enumerate(self.issame_list):
#             if (2 * idx + 1) < len(self.images):
#                 pairs.append((2 * idx, 2 * idx + 1))
#                 labels.append(int(same))
#             if len(pairs) >= num_pairs:
#                 break
#         return pairs, labels

#     def __getitem__(self, idx):
#         idx1, idx2 = self.pairs[idx]
#         label = self.labels[idx]
#         # Convert from OpenCV BGR to PIL RGB
#         img1 = Image.fromarray(cv2.cvtColor(self.images[idx1], cv2.COLOR_BGR2RGB))
#         img2 = Image.fromarray(cv2.cvtColor(self.images[idx2], cv2.COLOR_BGR2RGB))
#         # Apply transforms
#         img1 = self.transforms(img1)
#         img2 = self.transforms(img2)
#         return (img1, img2), torch.tensor(label, dtype=torch.float32)



# class LFWDataset(BaseDataset):
#     def __init__(self, dataset_dir, image_extension=".jpg", num_pairs=6000):
#         super().__init__(dataset_dir, image_extension=image_extension, num_pairs=num_pairs)

#     def _load_image_paths(self):
#         bin_path = os.path.join(self.dataset_dir, "lfw.bin")
#         with open(bin_path, 'rb') as f:
#             self.bins, self.issame_list = pickle.load(f, encoding='bytes')

#         self.images = []
#         for b in self.bins:
#             img_data = np.frombuffer(b, dtype=np.uint8)
#             img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
#             self.images.append(img)

#         return {}

#     def _create_pairs(self, num_pairs):
#         pairs, labels = [], []
#         for idx, same in enumerate(self.issame_list):
#             if (2 * idx + 1) < len(self.images):
#                 pairs.append((2 * idx, 2 * idx + 1))
#                 labels.append(int(same))
#             if len(pairs) >= num_pairs:
#                 break
#         return pairs, labels

#     def __getitem__(self, idx):
#         idx1, idx2 = self.pairs[idx]
#         label = self.labels[idx]
#         # Convert from OpenCV BGR to PIL RGB
#         img1 = Image.fromarray(cv2.cvtColor(self.images[idx1], cv2.COLOR_BGR2RGB))
#         img2 = Image.fromarray(cv2.cvtColor(self.images[idx2], cv2.COLOR_BGR2RGB))
#         # Apply transforms
#         img1 = self.transforms(img1)
#         img2 = self.transforms(img2)
#         return (img1, img2), torch.tensor(label, dtype=torch.float32)
    




# class CPLFWDataset(BaseDataset):
#     def __init__(self, dataset_dir, image_extension=".jpg", num_pairs=6000):
#         super().__init__(dataset_dir, image_extension=image_extension, num_pairs=num_pairs)

#     def _load_image_paths(self):
#         bin_path = os.path.join(self.dataset_dir, "cplfw.bin")
#         with open(bin_path, 'rb') as f:
#             self.bins, self.issame_list = pickle.load(f, encoding='bytes')

#         self.images = []
#         for b in self.bins:
#             img_data = np.frombuffer(b, dtype=np.uint8)
#             img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
#             self.images.append(img)

#         return {}

#     def _create_pairs(self, num_pairs):
#         pairs, labels = [], []
#         for idx, same in enumerate(self.issame_list):
#             if (2 * idx + 1) < len(self.images):
#                 pairs.append((2 * idx, 2 * idx + 1))
#                 labels.append(int(same))
#             if len(pairs) >= num_pairs:
#                 break
#         return pairs, labels

#     def __getitem__(self, idx):
#         idx1, idx2 = self.pairs[idx]
#         label = self.labels[idx]
#         # Convert from OpenCV BGR to PIL RGB
#         img1 = Image.fromarray(cv2.cvtColor(self.images[idx1], cv2.COLOR_BGR2RGB))
#         img2 = Image.fromarray(cv2.cvtColor(self.images[idx2], cv2.COLOR_BGR2RGB))
#         # Apply transforms
#         img1 = self.transforms(img1)
#         img2 = self.transforms(img2)
#         return (img1, img2), torch.tensor(label, dtype=torch.float32)
    



# class CALFWDataset(BaseDataset):
#     def __init__(self, dataset_dir, image_extension=".jpg", num_pairs=6000):
#         super().__init__(dataset_dir, image_extension=image_extension, num_pairs=num_pairs)

#     def _load_image_paths(self):
#         bin_path = os.path.join(self.dataset_dir, "calfw.bin")
#         with open(bin_path, 'rb') as f:
#             self.bins, self.issame_list = pickle.load(f, encoding='bytes')

#         self.images = []
#         for b in self.bins:
#             img_data = np.frombuffer(b, dtype=np.uint8)
#             img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
#             self.images.append(img)

#         return {}

#     def _create_pairs(self, num_pairs):
#         pairs, labels = [], []
#         for idx, same in enumerate(self.issame_list):
#             if (2 * idx + 1) < len(self.images):
#                 pairs.append((2 * idx, 2 * idx + 1))
#                 labels.append(int(same))
#             if len(pairs) >= num_pairs:
#                 break
#         return pairs, labels

#     def __getitem__(self, idx):
#         idx1, idx2 = self.pairs[idx]
#         label = self.labels[idx]
#         # Convert from OpenCV BGR to PIL RGB
#         img1 = Image.fromarray(cv2.cvtColor(self.images[idx1], cv2.COLOR_BGR2RGB))
#         img2 = Image.fromarray(cv2.cvtColor(self.images[idx2], cv2.COLOR_BGR2RGB))
#         # Apply transforms
#         img1 = self.transforms(img1)
#         img2 = self.transforms(img2)
#         return (img1, img2), torch.tensor(label, dtype=torch.float32)



# class CFPDataset(BaseDataset):
#     def __init__(self, dataset_dir, image_extension=".jpg", num_pairs=6000):
#         super().__init__(dataset_dir, image_extension=image_extension, num_pairs=num_pairs)

#     def _load_image_paths(self):
#         bin_path = os.path.join(self.dataset_dir, "cfp_fp.bin")
#         with open(bin_path, 'rb') as f:
#             self.bins, self.issame_list = pickle.load(f, encoding='bytes')

#         self.images = []
#         for b in self.bins:
#             img_data = np.frombuffer(b, dtype=np.uint8)
#             img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
#             self.images.append(img)

#         return {}

#     def _create_pairs(self, num_pairs):
#         pairs, labels = [], []
#         for idx, same in enumerate(self.issame_list):
#             if (2 * idx + 1) < len(self.images):
#                 pairs.append((2 * idx, 2 * idx + 1))
#                 labels.append(int(same))
#             if len(pairs) >= num_pairs:
#                 break
#         return pairs, labels

#     def __getitem__(self, idx):
#         idx1, idx2 = self.pairs[idx]
#         label = self.labels[idx]
#         # Convert from OpenCV BGR to PIL RGB
#         img1 = Image.fromarray(cv2.cvtColor(self.images[idx1], cv2.COLOR_BGR2RGB))
#         img2 = Image.fromarray(cv2.cvtColor(self.images[idx2], cv2.COLOR_BGR2RGB))
#         # Apply transforms
#         img1 = self.transforms(img1)
#         img2 = self.transforms(img2)
#         return (img1, img2), torch.tensor(label, dtype=torch.float32)