import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchmetrics.classification import BinaryF1Score, BinaryAUROC
from torch.amp import autocast, GradScaler
from dataclasses import dataclass, asdict
from src.data import get_dataloaders, download_cifake
from src.utils import Checkpointer
from src.losses import FocalLoss
from src.config import TrainConfig
from src.models.ResNet18 import get_baseline_resnet
from src.models.SaFEnet import get_safenet
from src.models.Ladevic import get_ladevic

from sklearn.metrics import f1_score, roc_auc_score

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

def get_model(config):
    if config.model == "safenet":
        model = get_safenet(num_classes=2)
    elif config.name == "resnet":
        model = get_baseline_resnet(num_classes=2)
    elif config.name == "ladevic":
        model = get_ladevic(num_classes=2)
    return model


def train_model(config: TrainConfig):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    
    # Setup DataLoaders
    print("Loading data...")
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        config.data_dir, config.batch_size, config.num_workers
    )
    print(f"Classes found: {classes}")
    
    # Setup Model
    model = get_model(config)
    model = model.to(device)
    
    # Setup Loss and Optimizer
    if config.loss == "focal":
        criterion = FocalLoss(gamma=2.0, num_classes=len(classes))
    else:
        criterion = nn.CrossEntropyLoss()
    # Optimize all parameters that require gradients (the grad CNN branch and the new fusion FC)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)
    
    # Setup LR Scheduler
    # Reduces learning rate by a factor of 0.1 if val_loss doesn't improve for 2 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    
    # Setup Checkpointer
    checkpointer = Checkpointer(config=asdict(config))

    # Setup f1 and auc
    f1 = BinaryF1Score().to(device)
    auc = BinaryAUROC().to(device)
    
    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
    
    best_val_acc = 0.0
    
    # Training Loop
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        print("-" * 10)
        
        # Training Phase
        model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        corrects = 0
        
        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                
                # Apply softmax to get probabilities for AUC
                probs = torch.nn.functional.softmax(outputs, dim=1)
                
                corrects += torch.sum(preds == labels.data)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                # Store probabilities for the positive class (class 1)
                if probabilities := probs.shape[1] > 1:
                    all_probs.extend(probs[:, 1].cpu().numpy())
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = corrects.double() / len(val_loader.dataset)
        
        # Calculate F1 Score and AUC
        epoch_f1 = f1(torch.tensor(all_preds), torch.tensor(all_labels)).item()
        epoch_auc = auc(torch.tensor(all_probs), torch.tensor(all_labels)).item()

        
        print(f"Train Loss: {epoch_train_loss:.4f}")
        print(f"Val Loss:   {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f} | F1: {epoch_f1:.4f} | AUC: {epoch_auc:.4f}")
        
        # Scheduler Step
        # Pass the validation loss to the scheduler to dynamically tune the LR
        scheduler.step(epoch_val_loss)
        
        # Checkpointing
        is_best = epoch_val_acc > best_val_acc
        if is_best:
            best_val_acc = epoch_val_acc
            
        checkpointer.log_metrics(epoch_train_loss, epoch_val_loss, epoch_val_acc.item(), epoch_f1, epoch_auc, epoch)
        checkpointer.save_checkpoint(model, optimizer, epoch, is_best)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CIFAKE Classifier")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to CIFAKE dataset root (should contain train and test folders)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--loss", type=str, default="focal", choices=["focal", "bce", "ce"], help="Loss function to use")
    parser.add_argument("--model", type=str, default="safenet", choices=["safenet", "resnet"], help="Model architecture to use")
    args = parser.parse_args()
    config = TrainConfig(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        loss=args.loss,
        model=args.model
    )
    train_model(config)
