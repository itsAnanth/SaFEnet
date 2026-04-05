import os
import shutil
import kagglehub
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

def get_dataloaders(data_dir, batch_size=32, num_workers=4, val_split=0.2):
    """
    Expects data_dir to have subfolders 'train' and 'test'.
    If val_split is > 0, it splits the train set into train and val.
    """
    
    # ResNet expects 224x224 and ImageNet normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    full_train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    
    num_train = len(full_train_dataset)
    num_val = int(val_split * num_train)
    num_train = num_train - num_val
    
    train_dataset, val_dataset = random_split(full_train_dataset, [num_train, num_val])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, full_train_dataset.classes
