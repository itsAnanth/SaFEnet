import os
import shutil
import torch
import kagglehub
from PIL import Image, ImageFilter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def download_cifake(data_dir):
    """
    Downloads the CIFAKE dataset from Kaggle using kagglehub
    and moves it to the specified data_dir.
    """
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    if os.path.exists(train_dir) and os.path.exists(test_dir):
        print(f"Dataset already found in {data_dir}. Skipping download.")
        return

    print("Downloading CIFAKE dataset from Kaggle...")
    # download dataset to kagglehub cache
    download_path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")

    print(f"Downloaded to cache: {download_path}. Moving to {data_dir}...")
    os.makedirs(data_dir, exist_ok=True)

    for item in os.listdir(download_path):
        s = os.path.join(download_path, item)
        d = os.path.join(data_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

    print("Dataset download and extraction complete!")


class GaussianBlurTransform:
    """Applies a slight Gaussian blur to simulate compression noise."""
    def __init__(self, radius=0.5):
        self.radius = radius

    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(radius=self.radius))

    def __repr__(self):
        return f"GaussianBlurTransform(radius={self.radius})"


# ----------------------------------------------------------------------- #
# Transforms                                                               #
# ----------------------------------------------------------------------- #
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.05),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
    transforms.RandomApply([GaussianBlurTransform(radius=0.5)], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def _split_train_val(data_dir, val_split):
    """
    Loads data_dir twice (once per transform) and returns Subsets with
    matching indices so train gets augmentations and val does not.
    """
    aug_ds   = datasets.ImageFolder(root=data_dir, transform=TRAIN_TRANSFORM)
    clean_ds = datasets.ImageFolder(root=data_dir, transform=VAL_TRANSFORM)
    n = len(aug_ds)
    n_val = int(val_split * n)
    idx = torch.randperm(n).tolist()
    return Subset(aug_ds, idx[n_val:]), Subset(clean_ds, idx[:n_val]), aug_ds.classes


def get_genimage_dataloaders(data_dir, batch_size=32, num_workers=4, val_split=0.2):
    """
    Loads a single Genimage generator directory (e.g. imagenet_ai_0419_biggan).
    Expects data_dir to have 'train' and 'val' subfolders, each with 'ai' and 'nature' class dirs.
    Splits 'train' into train/val sets and uses the provided 'val' folder as the test set.
    """
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    train_dataset, val_dataset, classes = _split_train_val(train_dir, val_split)
    test_dataset = datasets.ImageFolder(root=val_dir, transform=VAL_TRANSFORM)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, classes


def get_dataloaders(data_dir, batch_size=32, num_workers=4, val_split=0.2):
    """
    Expects data_dir to have subfolders 'train' and 'test' (or 'val').
    If val_split is > 0 and no separate val/test dir exists, it splits the train set into train and val.
    """

    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    val_dir = os.path.join(data_dir, 'val')

    if os.path.exists(test_dir):
        train_dataset, val_dataset, classes = _split_train_val(train_dir, val_split)
        test_loader = DataLoader(
            datasets.ImageFolder(root=test_dir, transform=VAL_TRANSFORM),
            batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
    elif os.path.exists(val_dir):
        print("Genimage dataset structure detected. Using 'val' folder as test set.")
        train_dataset = datasets.ImageFolder(root=train_dir, transform=TRAIN_TRANSFORM)
        val_dataset   = datasets.ImageFolder(root=val_dir,   transform=VAL_TRANSFORM)
        classes = train_dataset.classes
        test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        train_dataset, val_dataset, classes = _split_train_val(train_dir, val_split)
        test_loader = None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if test_loader is None:
        test_loader = val_loader

    return train_loader, val_loader, test_loader, classes
