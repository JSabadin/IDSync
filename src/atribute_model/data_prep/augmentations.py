import torchvision.transforms as transforms
from PIL import Image
from typing import List, Tuple

def get_transforms(use_center_crop: bool = False) -> transforms.Compose:
    """
    Function to return the transformations to be applied on the input image.
    
    Args:
        use_center_crop (bool): Whether to apply a center crop. CELEB-A specific.
    
    Returns:
        torchvision.transforms.Compose: Composed transformations to be applied.
    """
    transform_list = []
    
    if use_center_crop:
        transform_list.append(transforms.CenterCrop((178, 178)))  # CELEB-A Specific
    
    transform_list.extend([
        transforms.Resize((112, 112)),  # Resize to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    return transforms.Compose(transform_list)

def adjust_jpeg_quality(img: Image.Image, quality: int = 75) -> Image.Image:
    return transforms.functional.adjust_jpeg_quality(img, quality)

def get_augmentations() -> transforms.Compose:
    """
    Function to return the augmentations to be applied on the input image.
    Returns:
        torchvision.transforms.Compose: Composed augmentations to be applied.
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.05),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.15, contrast=0.3, saturation=0.1, hue=0.1)], p=0.5),
        transforms.RandomGrayscale(p=0.01),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.1),
        transforms.RandomRotation(degrees=10),
        transforms.RandomResizedCrop(size=(112, 112), scale=(0.8, 1.0), ratio=(0.9, 1.1)),

    ])