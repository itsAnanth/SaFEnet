import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

VALID_BRANCHES = {'spatial', 'gradient', 'frequency'}

# Branch output dimensions before projection
BRANCH_DIMS = {
    'spatial':   512,
    'gradient':  32,
    'frequency': 32,
}


class CrossBranchAttention(nn.Module):
    def __init__(self, active_branches, feat_dim=128, num_heads=4):
        """
        Projects each active branch to feat_dim, then applies multi-head
        self-attention across them so each branch can attend to the others.

        If only one branch is active, attention is a no-op and we just
        return the single projected token directly.
        """
        super(CrossBranchAttention, self).__init__()

        self.active_branches = active_branches
        self.feat_dim = feat_dim
        n = len(active_branches)

        # One projection layer per active branch
        self.projections = nn.ModuleDict({
            branch: nn.Linear(BRANCH_DIMS[branch], feat_dim)
            for branch in active_branches
        })

        # Attention only makes sense with 2+ tokens
        self.use_attn = n > 1
        if self.use_attn:
            self.attn = nn.MultiheadAttention(
                embed_dim=feat_dim,
                num_heads=num_heads,
                batch_first=True
            )
            self.norm = nn.LayerNorm(feat_dim)

    def forward(self, branch_features: dict):
        """
        branch_features: dict mapping branch name → tensor (B, branch_dim)
                         Must contain exactly the active branches.
        Returns: (B, len(active_branches) * feat_dim)
        """
        # Project each branch in a fixed order for reproducibility
        projected = [
            self.projections[b](branch_features[b])
            for b in self.active_branches
        ]

        if not self.use_attn:
            # Single branch — just return the projection directly
            return projected[0]

        # Stack → (B, N, feat_dim)
        tokens = torch.stack(projected, dim=1)

        # Cross-branch attention + residual + norm
        attended, _ = self.attn(tokens, tokens, tokens)
        attended = self.norm(attended + tokens)

        # Flatten → (B, N * feat_dim)
        return attended.flatten(1)


class SaFENet(nn.Module):
    def __init__(
        self,
        num_classes=2,
        branches=('spatial', 'gradient', 'frequency'),
        feat_dim=128,
        num_heads=4,
    ):
        """
        Args:
            num_classes:  Number of output classes (default 2).
            branches:     Any non-empty subset of {'spatial', 'gradient', 'frequency'}.
                          Controls which feature branches are built and used.
                          Examples:
                            ('spatial',)                         -- baseline only
                            ('spatial', 'frequency')             -- no gradient branch
                            ('spatial', 'gradient', 'frequency') -- full model
            feat_dim:     Common projection dimension for cross-branch attention.
            num_heads:    Number of attention heads (must divide feat_dim).
        """
        super(SaFENet, self).__init__()

        # Validate and deduplicate while preserving order
        branches = list(dict.fromkeys(branches))
        assert len(branches) >= 1, "Provide at least one branch."
        assert all(b in VALID_BRANCHES for b in branches), \
            f"Invalid branch(es). Choose from {VALID_BRANCHES}."

        self.active_branches = branches

        # ------------------------------------------------------------------ #
        # 1. Spatial Branch — ResNet-18 Backbone                             #
        # ------------------------------------------------------------------ #
        if 'spatial' in branches:
            weights = models.ResNet18_Weights.DEFAULT
            self.resnet = models.resnet18(weights=weights)
            for param in self.resnet.parameters():
                param.requires_grad = False
            for param in self.resnet.layer4.parameters():
                param.requires_grad = True
            self.resnet.fc = nn.Identity()   # outputs 512-d

        # ------------------------------------------------------------------ #
        # 2. Gradient Edge CNN Branch                                         #
        # ------------------------------------------------------------------ #
        if 'gradient' in branches:
            self.grad_cnn = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )   # outputs 32-d

        # ------------------------------------------------------------------ #
        # 3. Frequency Spectrum CNN Branch                                    #
        # ------------------------------------------------------------------ #
        if 'frequency' in branches:
            self.freq_cnn = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )   # outputs 32-d

        # ------------------------------------------------------------------ #
        # 4. Cross-Branch Attention Fusion                                    #
        # ------------------------------------------------------------------ #
        self.cross_attn = CrossBranchAttention(
            active_branches=branches,
            feat_dim=feat_dim,
            num_heads=num_heads,
        )

        # Output dim: N branches x feat_dim  (or just feat_dim if N==1)
        n = len(branches)
        fused_dim = feat_dim * n if n > 1 else feat_dim

        # ------------------------------------------------------------------ #
        # 5. Classifier Head                                                  #
        # ------------------------------------------------------------------ #
        self.fc = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        # ------------------------------------------------------------------ #
        # Static Sobel Filters (only registered when gradient branch active) #
        # ------------------------------------------------------------------ #
        if 'gradient' in branches:
            sobel_x = torch.tensor([[[[-1., 0., 1.],
                                       [-2., 0., 2.],
                                       [-1., 0., 1.]]]]).float()
            sobel_y = torch.tensor([[[[-1., -2., -1.],
                                       [ 0.,  0.,  0.],
                                       [ 1.,  2.,  1.]]]]).float()
            self.register_buffer('sobel_x', sobel_x)
            self.register_buffer('sobel_y', sobel_y)

    # ---------------------------------------------------------------------- #
    # Helper Methods                                                          #
    # ---------------------------------------------------------------------- #

    def get_grayscale(self, x):
        """RGB -> grayscale using BT.601 luminance weights."""
        return (0.299 * x[:, 0:1, :, :]
              + 0.587 * x[:, 1:2, :, :]
              + 0.114 * x[:, 2:3, :, :])

    def compute_gradients(self, gray):
        """Sobel gradient magnitude map."""
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def compute_frequency_spectrum(self, gray):
        """Log-scaled 2-D FFT magnitude spectrum."""
        fft2      = torch.fft.fft2(gray)
        shifted   = torch.fft.fftshift(fft2, dim=(-2, -1))
        magnitude = torch.abs(shifted)
        return torch.log(magnitude + 1e-6)

    # ---------------------------------------------------------------------- #
    # Forward Pass                                                            #
    # ---------------------------------------------------------------------- #

    def forward(self, x):
        branch_features = {}

        if 'spatial' in self.active_branches:
            branch_features['spatial'] = self.resnet(x)              # (B, 512)

        # Only compute grayscale if an auxiliary branch needs it
        if 'gradient' in self.active_branches or 'frequency' in self.active_branches:
            gray = self.get_grayscale(x)

        if 'gradient' in self.active_branches:
            branch_features['gradient'] = self.grad_cnn(
                self.compute_gradients(gray)
            )                                                         # (B, 32)

        if 'frequency' in self.active_branches:
            branch_features['frequency'] = self.freq_cnn(
                self.compute_frequency_spectrum(gray)
            )                                                         # (B, 32)

        # Cross-branch attention fusion (identity projection if single branch)
        f_fused = self.cross_attn(branch_features)

        return self.fc(f_fused)

    def __repr__(self):
        return (f"SaFENet(branches={self.active_branches}, "
                f"num_classes={self.fc[-1].out_features})")


# --------------------------------------------------------------------------- #
# Factory helpers                                                              #
# --------------------------------------------------------------------------- #

def get_safenet(branches=('spatial', 'gradient', 'frequency'), num_classes=2, feat_dim=128):
    """
    Convenience factory. Pass any subset of branches for ablation studies.

    Usage examples:
        get_safenet()                                         # full model
        get_safenet(branches=['spatial'])                     # spatial only
        get_safenet(branches=['spatial', 'frequency'])        # no gradient
        get_safenet(branches=['spatial', 'gradient'])         # no frequency
        get_safenet(branches=['gradient', 'frequency'])       # no spatial backbone
    """
    return SaFENet(num_classes=num_classes, branches=branches, feat_dim=feat_dim)


def get_baseline_resnet(num_classes=2):
    """
    Standard ResNet-18 baseline with the same backbone configuration
    (frozen trunk, unfrozen layer4) for an apples-to-apples comparison.
    """
    weights = models.ResNet18_Weights.DEFAULT
    model   = models.resnet18(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes)
    )

    return model


# --------------------------------------------------------------------------- #
# Quick sanity check                                                           #
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    configs = [
        ('spatial',),
        ('gradient',),
        ('frequency',),
        ('spatial', 'gradient'),
        ('spatial', 'frequency'),
        ('gradient', 'frequency'),
        ('spatial', 'gradient', 'frequency'),
    ]

    dummy = torch.randn(2, 3, 32, 32)

    for cfg in configs:
        model = get_safenet(branches=cfg)
        out   = model(dummy)
        print(f"branches={list(cfg):<45} output shape: {out.shape}")