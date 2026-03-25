"""Image transforms for OCT en face image quality assessment.

Final released preprocessing assumptions:
- input resized to 480 x 480
- grayscale loaded as 1 channel
- normalized with mean=[0.5], std=[0.5]
- repeated to 3 channels for EfficientNetV2-L
"""

from __future__ import annotations

import torch
import torchvision.transforms as T


MEAN = [0.5]
STD = [0.5]
IMAGE_SIZE = 480
PRE_CROP_SIZE = 512


def to_3ch_tensor(x1c: torch.Tensor) -> torch.Tensor:
    """
    Repeat a [1, H, W] grayscale tensor into [3, H, W].
    """
    if x1c.ndim != 3 or x1c.shape[0] != 1:
        raise ValueError(f"Expected tensor shape [1, H, W], got {tuple(x1c.shape)}")
    return x1c.repeat(3, 1, 1)


def get_training_transform(image_size: int = IMAGE_SIZE) -> T.Compose:
    """
    Training-time augmentation from notebook Cell 12.
    Included here for provenance, even if the public repo is inference-focused.
    """
    return T.Compose([
        T.Resize((PRE_CROP_SIZE, PRE_CROP_SIZE)),
        T.RandomRotation(12, fill=0),
        T.RandomResizedCrop(image_size, scale=(0.90, 1.0), ratio=(0.95, 1.05)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.3),
        T.Grayscale(num_output_channels=1),
        T.RandomApply([T.ColorJitter(brightness=0.10)], p=0.5),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.6))], p=0.3),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
        T.Lambda(to_3ch_tensor),
    ])


def get_inference_transform(image_size: int = IMAGE_SIZE) -> T.Compose:
    """
    Inference/evaluation transform matching the released model.
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
        T.Lambda(to_3ch_tensor),
    ])


# Alias for clarity in evaluation code
get_test_transform = get_inference_transform
