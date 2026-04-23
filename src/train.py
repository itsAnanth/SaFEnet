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
from src.data import get_dataloaders
from src.utils import Checkpointer
from src.losses import FocalLoss
from src.config import TrainConfig
from src.models.ResNet18 import get_resnet18
from src.models.proto import get_safenet, get_param_groups
from src.models.Ladevic import get_ladevic
from src.models.MobileNetv2 import get_MobileNetV2


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
    optimizer = None

    if config.model == "safenet":
        model = get_safenet(num_classes=2)
        # get_param_groups returns groups with differential LRs.
        # base_lr should be ~1e-3 so aux branches get 5e-4 (enough to train from scratch)
        # and backbone fine-tune gets 1e-4.
        param_groups = get_param_groups(model, base_lr=config.lr)
        optimizer = optim.Adam(param_groups)

    elif config.model == "resnet18":
        model = get_resnet18(num_classes=2)
        optimizer = optim.Adam(model.parameters(), lr=config.lr)  # BUG FIX: was missing model.parameters()

    elif config.model == "ladevic":
        model = get_ladevic(num_classes=2)
        optimizer = optim.Adam(model.parameters(), lr=config.lr)  # BUG FIX: was missing model.parameters()

    elif config.model == "mobilenetv2":
        model = get_MobileNetV2(num_classes=2)
        optimizer = optim.Adam(model.parameters(), lr=config.lr)  # BUG FIX: was missing model.parameters()

    else:
        raise ValueError(f"Unknown model: {config.model}")

    return model, optimizer


def log_gradient_norms(model, optimizer):
    """
    Print the gradient norm per optimizer param group.
    This tells you whether each branch (backbone / aux / head) is actually
    receiving meaningful gradients — a flat norm means the branch isn't learning.
    """
    norms = {}
    for group in optimizer.param_groups:
        name = group.get('name', 'unnamed')
        total_norm = 0.0
        for p in group['params']:
            if p.grad is not None:
                total_norm += p.grad.detach().norm(2).item() ** 2
        norms[name] = total_norm ** 0.5
    norm_str = ' | '.join(f"{k}: {v:.4f}" for k, v in norms.items())
    print(f"  Grad norms → {norm_str}")
    return norms


def train_model(config: TrainConfig):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        config.data_dir, config.batch_size, config.num_workers
    )
    print(f"Classes found: {classes}")

    model, optimizer = get_model(config)
    model = model.to(device)

    print(f"\nOptimizer param groups:")
    for g in optimizer.param_groups:
        n_params = sum(p.numel() for p in g['params'])
        print(f"  [{g.get('name', '?')}]  lr={g['lr']:.2e}  params={n_params:,}")

    if config.loss == "focal":
        criterion = FocalLoss(gamma=2.0, num_classes=len(classes))
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Cosine annealing is more stable than ReduceLROnPlateau for multi-branch models.
    # ReduceLROnPlateau with patience=2 collapses the LR too early when aux branches
    # haven't yet converged, starving them of the learning signal they need.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-6
    )

    checkpointer = Checkpointer(config=asdict(config))
    f1_metric = BinaryF1Score().to(device)
    auc_metric = BinaryAUROC().to(device)
    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    best_val_acc  = 0.0
    history = {
        'train_loss': [], 'val_loss': [], 'val_acc': [],
        'val_f1': [], 'val_auc': [], 'lr': [], 'grad_norms': []
    }

    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        print("-" * 50)

        # ------------------------------------------------------------------ #
        # Training Phase                                                      #
        # ------------------------------------------------------------------ #
        model.train()
        running_loss = 0.0
        epoch_grad_norms = None

        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc="Training")):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            # Unscale before reading grad norms (only log on last batch of epoch)
            scaler.unscale_(optimizer)

            
            if batch_idx == len(train_loader) - 1:
                epoch_grad_norms = log_gradient_norms(model, optimizer)

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)

        # ------------------------------------------------------------------ #
        # Validation Phase                                                    #
        # ------------------------------------------------------------------ #
        model.eval()
        val_loss  = 0.0
        corrects  = 0
        all_labels, all_preds, all_probs = [], [], []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss    = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                corrects += (preds == labels).sum()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # BUG FIX: removed broken walrus operator

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc  = (corrects.double() / len(val_loader.dataset)).item()
        epoch_f1  = f1_metric(torch.tensor(all_preds), torch.tensor(all_labels)).item()
        epoch_auc = auc_metric(torch.tensor(all_probs), torch.tensor(all_labels)).item()

        current_lrs = [g['lr'] for g in optimizer.param_groups]
        print(f"Train Loss: {epoch_train_loss:.4f}")
        print(f"Val   Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc:.4f} | F1: {epoch_f1:.4f} | AUC: {epoch_auc:.4f}")
        lr_summary = {g.get('name', '?'): f"{g['lr']:.2e}" for g in optimizer.param_groups}
        print(f"LRs: {lr_summary}")

        # Log to history for curve plotting
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['val_f1'].append(epoch_f1)
        history['val_auc'].append(epoch_auc)
        history['lr'].append(current_lrs)
        if epoch_grad_norms:
            history['grad_norms'].append(epoch_grad_norms)

        scheduler.step()

        is_best = epoch_val_acc > best_val_acc
        if is_best:
            best_val_acc = epoch_val_acc
            print(f"  ✓ New best val acc: {best_val_acc:.4f}")

        checkpointer.log_metrics(
            epoch_train_loss, epoch_val_loss, epoch_val_acc, epoch_f1, epoch_auc, epoch
        )
        checkpointer.save_checkpoint(model, optimizer, epoch, is_best)

    # ---------------------------------------------------------------------- #
    # Diagnosis Summary                                                       #
    # ---------------------------------------------------------------------- #
    print("\n" + "=" * 60)
    print("TRAINING DIAGNOSIS")
    print("=" * 60)

    train_losses = history['train_loss']
    val_losses   = history['val_loss']
    val_accs     = history['val_acc']

    final_gap = val_losses[-1] - train_losses[-1]
    if final_gap > 0.1:
        print("⚠ OVERFITTING  — val loss is significantly above train loss.")
        print("  → Try: stronger dropout, weight decay, more augmentation.")
    elif val_accs[-1] < val_accs[max(0, len(val_accs)//2)]:
        print("⚠ VAL ACCURACY DECLINED late in training.")
        print("  → LR may have been too high. Try lower base_lr or more epochs.")
    else:
        print("✓ No obvious overfitting detected.")

    if history['grad_norms']:
        last_norms = history['grad_norms'][-1]
        print("\nFinal epoch gradient norms:")
        for k, v in last_norms.items():
            flag = " ← DEAD (not learning!)" if v < 1e-4 else ""
            print(f"  {k}: {v:.6f}{flag}")

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print("=" * 60)

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AIGC Detector")
    parser.add_argument("--data_dir",    type=str,   required=True)
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1e-3,
                        help="Base LR. SaFENet aux branches get 0.5x, backbone gets 0.1x.")
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--loss",        type=str,   default="focal",
                        choices=["focal", "bce", "ce"])
    parser.add_argument("--model",       type=str,   default="safenet",
                        choices=["safenet", "resnet18", "mobilenetv2", "ladevic"])
    args   = parser.parse_args()
    config = TrainConfig(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        loss=args.loss,
        model=args.model,
    )
    train_model(config)