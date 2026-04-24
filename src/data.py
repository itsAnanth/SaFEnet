import os
import shutil
import kagglehub
from PIL import Image, ImageFilter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

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

def get_dataloaders(data_dir, batch_size=32, num_workers=4, val_split=0.2):
    """
    Expects data_dir to have subfolders 'train' and 'test' (or 'val').
    If val_split is > 0 and no separate val/test dir exists, it splits the train set into train and val.
    """
    
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    val_dir = os.path.join(data_dir, 'val')
    
    full_train_dataset = datasets.ImageFolder(root=train_dir, transform=TRAIN_TRANSFORM)
    
    if os.path.exists(test_dir):
        eval_dataset = datasets.ImageFolder(root=test_dir, transform=VAL_TRANSFORM)
        train_dataset, val_dataset = random_split(full_train_dataset, [len(full_train_dataset) - int(val_split * len(full_train_dataset)), int(val_split * len(full_train_dataset))])
        test_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    elif os.path.exists(val_dir):
        print("Genimage dataset structure detected. Using 'val' folder as test set.")
        val_dataset = datasets.ImageFolder(root=val_dir, transform=VAL_TRANSFORM)
        train_dataset = full_train_dataset
        test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers) # Use val as test
    else:
        num_train = len(full_train_dataset)
        num_val = int(val_split * num_train)
        num_train = num_train - num_val
        train_dataset, val_dataset = random_split(full_train_dataset, [num_train, num_val])
        test_loader = None
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    if test_loader is None:
        test_loader = val_loader
        
    return train_loader, val_loader, test_loader, full_train_dataset.classes
