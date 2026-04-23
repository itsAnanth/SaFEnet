import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for hard example mining.
    Down-weights well-classified examples and focuses on the hard, misclassified ones.
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', num_classes=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        # Calculate standard cross entropy loss (unreduced)
        _base_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get the true class probabilities (pt)
        pt = torch.exp(-_base_loss)
        
        # Calculate focal loss factor: (1 - pt)^gamma
        focal_loss = self.alpha * (1 - pt) ** self.gamma * _base_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss