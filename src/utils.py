import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for hard example mining.
    Down-weights well-classified examples and focuses on the hard, misclassified ones.
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculate standard cross entropy loss (unreduced)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get the true class probabilities (pt)
        pt = torch.exp(-ce_loss)
        
        # Calculate focal loss factor: (1 - pt)^gamma
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class Checkpointer:
    def __init__(self, base_dir="checkpoints"):
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(base_dir, self.timestamp)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.metrics = {
            "model_name": None,
            "description": None,
            "train_loss": [],
            "val_loss": [],
            "accuracy": []
        }
        
        self.metrics["model_name"] = str(input("[CHECKPOINTER] Enter the model name: "))
        self.metrics["description"] = str(input("[CHECKPOINTER] Enter a description for this training run: "))
        print(f"Checkpointer initialized at: {self.log_dir}")

    def log_metrics(self, train_loss, val_loss, accuracy, epoch):
        self.metrics["train_loss"].append(train_loss)
        self.metrics["val_loss"].append(val_loss)
        self.metrics["accuracy"].append(accuracy)
        
        # Save metrics to a json file for easy tracking
        metrics_path = os.path.join(self.log_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=4)
            
    def save_checkpoint(self, model, optimizer, epoch, is_best=False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": self.metrics
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(self.log_dir, "checkpoint_latest.pt")
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.log_dir, "checkpoint_best.pt")
            torch.save(checkpoint, best_path)
            print(f"Saved new best checkpoint at epoch {epoch}")
    
    @staticmethod
    def load_metrics(path=None):

        metrics_path = os.path.join(path, 'metrics.json')
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        return metrics

