import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

class Checkpointer:
    def __init__(self, base_dir="checkpoints", config=None):
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(base_dir, self.timestamp)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.metrics = {
            "model_name": None,
            "description": None,
            "config": config,
            "train_loss": [],
            "val_loss": [],
            "accuracy": [],
            "f1_score": [],
            "auc": []
        }
        
        self.metrics["description"] = str(input("[CHECKPOINTER] Enter a description for this training run: "))
        metrics_path = os.path.join(self.log_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=4)
        print(f"Checkpointer initialized at: {self.log_dir}")

    def log_metrics(self, train_loss, val_loss, accuracy, f1_score, auc, epoch):
        self.metrics["train_loss"].append(train_loss)
        self.metrics["val_loss"].append(val_loss)
        self.metrics["accuracy"].append(accuracy)
        self.metrics["f1_score"].append(f1_score)
        self.metrics["auc"].append(auc)
        
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

