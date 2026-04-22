import random
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse

from src.data import get_dataloaders, download_cifake
from src.model import get_safenet, get_baseline_resnet
from src.utils import Checkpointer, FocalLoss

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(data_dir, num_epochs=10, batch_size=32, lr=1e-3, num_workers=4):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    
    # Ensure dataset is downloaded
    download_cifake(data_dir)
    
    # Setup DataLoaders
    print("Loading data...")
    train_loader, val_loader, test_loader, classes = get_dataloaders(data_dir, batch_size, num_workers)
    print(f"Classes found: {classes}")
    
    # Setup Model
    model = get_safenet(branches=('spatial', ), num_classes=len(classes))
    model = model.to(device)
    
    # Setup Loss and Optimizer
    criterion = FocalLoss(gamma=2.0)
    # Optimize all parameters that require gradients (the grad CNN branch and the new fusion FC)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    # Setup LR Scheduler
    # Reduces learning rate by a factor of 0.1 if val_loss doesn't improve for 2 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    
    # Setup Checkpointer
    checkpointer = Checkpointer()
    
    best_val_acc = 0.0
    
    # Training Loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 10)
        
        # Training Phase
        model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        corrects = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels.data)
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = corrects.double() / len(val_loader.dataset)
        
        print(f"Train Loss: {epoch_train_loss:.4f}")
        print(f"Val Loss:   {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")
        
        # Scheduler Step
        # Pass the validation loss to the scheduler to dynamically tune the LR
        scheduler.step(epoch_val_loss)
        
        # Checkpointing
        is_best = epoch_val_acc > best_val_acc
        if is_best:
            best_val_acc = epoch_val_acc
            
        checkpointer.log_metrics(epoch_train_loss, epoch_val_loss, epoch_val_acc.item(), epoch)
        checkpointer.save_checkpoint(model, optimizer, epoch, is_best)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CIFAKE Classifier")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to CIFAKE dataset root (should contain train and test folders)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    
    args = parser.parse_args()
    train_model(args.data_dir, num_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, num_workers=args.num_workers)
