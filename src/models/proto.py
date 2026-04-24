"""
SaFENet — Spatial and Frequency Extraction Network
Improved architecture with:
  - MobileNetV2 spatial backbone + CBAM spatial attention
  - Multi-scale Sobel gradients (3x3, 5x5, 7x7) + gradient direction channel
  - Frequency branch with phase channel + radial power spectrum feature
  - True cross-modal attention (each branch queries the others) + FFN block
  - Learnable modality type embeddings to distinguish branch tokens
  - Gated fusion instead of naive flatten
  - Two-layer MLP classifier with GELU + dropout + label smoothing support
  - Differential learning rate helper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import MobileNet_V2_Weights

VALID_BRANCHES = {'spatial', 'gradient', 'frequency'}

# Branch output dimensions BEFORE projection (updated for new branches)
BRANCH_DIMS = {
    'spatial':   1280,   # MobileNetV2 final features
    'gradient':  128,    # multi-scale + direction CNN → 128-d
    'frequency': 128,    # magnitude + phase + radial CNN → 128-d
}


# CBAM — Convolutional Block Attention Module                                 
# Applies channel + spatial attention with negligible parameter cost.         
# Reference: Woo et al., ECCV 2018                                            
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        # Spatial attention
        self.spatial_conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size,
            padding=kernel_size // 2, bias=False
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # --- Channel attention ---
        avg = self.channel_fc(self.avg_pool(x).view(B, C))
        mx  = self.channel_fc(self.max_pool(x).view(B, C))
        ch_attn = torch.sigmoid(avg + mx).view(B, C, 1, 1)
        x = x * ch_attn

        # --- Spatial attention ---
        avg_s = x.mean(dim=1, keepdim=True)
        max_s = x.max(dim=1, keepdim=True).values
        sp_attn = torch.sigmoid(self.spatial_conv(torch.cat([avg_s, max_s], dim=1)))
        return x * sp_attn


# True Cross-Modal Attention Fusion (CMAF)                                   
# Each modality attends to the OTHER modalities (not itself).                 
# Includes learnable modality type embeddings + post-attention FFN.           
class CMAF(nn.Module):
    def __init__(self, active_branches, feat_dim=128, num_heads=4, ffn_mult=2, dropout=0.1):
        super().__init__()
        self.active_branches = active_branches
        self.feat_dim = feat_dim
        n = len(active_branches)

        # Project each branch into the shared latent space
        self.projections = nn.ModuleDict({
            b: nn.Sequential(
                nn.Linear(BRANCH_DIMS[b], feat_dim),
                nn.LayerNorm(feat_dim),
            )
            for b in active_branches
        })

        # Learnable modality type embeddings — tells the model which token
        # came from which branch (analogous to BERT token type IDs)
        self.modality_embeddings = nn.ParameterDict({
            b: nn.Parameter(torch.zeros(feat_dim))
            for b in active_branches
        })
        for emb in self.modality_embeddings.values():
            nn.init.normal_(emb.data, std=0.02)

        self.use_attn = n > 1
        if self.use_attn:
            # True cross-attention: each branch as Query, others as Key/Value
            self.cross_attns = nn.ModuleDict({
                b: nn.MultiheadAttention(
                    embed_dim=feat_dim,
                    num_heads=num_heads,
                    batch_first=True,
                    dropout=dropout,
                )
                for b in active_branches
            })
            self.attn_norms = nn.ModuleDict({
                b: nn.LayerNorm(feat_dim) for b in active_branches
            })

            # Post-attention FFN (standard transformer block component)
            ffn_dim = feat_dim * ffn_mult
            self.ffns = nn.ModuleDict({
                b: nn.Sequential(
                    nn.Linear(feat_dim, ffn_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ffn_dim, feat_dim),
                    nn.Dropout(dropout),
                )
                for b in active_branches
            })
            self.ffn_norms = nn.ModuleDict({
                b: nn.LayerNorm(feat_dim) for b in active_branches
            })

            # Gated fusion: learned soft weighting of each branch's contribution
            # instead of naive concatenation
            self.gate = nn.Sequential(
                nn.Linear(feat_dim * n, n),
                nn.Softmax(dim=-1)
            )

        # Output dim
        self.out_dim = feat_dim  # always feat_dim after gated fusion

    def forward(self, branch_features: dict):
        """
        branch_features: {branch_name: (B, branch_dim)}
        Returns: (B, feat_dim)
        """
        # 1. Project + add modality embeddings
        projected = {}
        for b in self.active_branches:
            p = self.projections[b](branch_features[b])          # (B, feat_dim)
            p = p + self.modality_embeddings[b].unsqueeze(0)     # broadcast add
            projected[b] = p

        if not self.use_attn:
            return projected[self.active_branches[0]]

        # 2. True cross-modal attention:
        #    For branch b: Query=b, Key/Value=all OTHER branches
        attended = {}
        for b in self.active_branches:
            q = projected[b].unsqueeze(1)                         # (B, 1, feat_dim)
            others = [projected[o] for o in self.active_branches if o != b]
            kv = torch.stack(others, dim=1)                       # (B, n-1, feat_dim)

            out, _ = self.cross_attns[b](q, kv, kv)              # (B, 1, feat_dim)
            out = out.squeeze(1)                                  # (B, feat_dim)

            # Residual + LayerNorm
            out = self.attn_norms[b](out + projected[b])

            # FFN + residual + LayerNorm
            out = self.ffn_norms[b](out + self.ffns[b](out))

            attended[b] = out

        # 3. Gated fusion — soft-weight each branch's attended representation
        stacked = torch.stack(
            [attended[b] for b in self.active_branches], dim=1
        )                                                         # (B, n, feat_dim)

        # Gate input: concatenation of all attended features
        gate_input = stacked.flatten(1)                           # (B, n * feat_dim)
        gates = self.gate(gate_input)                             # (B, n)

        # Weighted sum across branches
        fused = (stacked * gates.unsqueeze(-1)).sum(dim=1)        # (B, feat_dim)
        return fused


# Multi-Scale Sobel Gradient Branch                                           
# Computes gradients at 3 kernel sizes and stacks magnitude + direction.      
class GradientBranch(nn.Module):
    def __init__(self):
        super().__init__()
        # Register Sobel kernels at three scales as buffers (non-trainable)
        for size in [3, 5, 7]:
            kx, ky = self._make_sobel(size)
            self.register_buffer(f'sobel_x_{size}', kx)
            self.register_buffer(f'sobel_y_{size}', ky)

        # Input: 6 channels (magnitude + direction × 3 scales)
        self.cnn = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )  # → 128-d

    @staticmethod
    def _make_sobel(size):
        """
        Construct Sobel kernels of exactly (size x size).

        The smoothing vector is the (size-1)-th row of Pascal's triangle
        (length = size). The derivative vector is the finite-difference of
        the (size-1)-th row, giving length = size. Both are correct.
        """
        import numpy as np

        # Smoothing kernel: binomial coefficients of degree (size-1), length=size
        smooth = np.array([1.0])
        for _ in range(size - 1):
            smooth = np.convolve(smooth, [1.0, 1.0])
        # smooth is now length-size

        # Derivative kernel: first difference of binomial coefficients, length=size
        # d/dx of binomial(n) = binomial(n-1) convolved with [1, -1],
        # which gives a length-size vector when n = size-2.
        deriv_base = np.array([1.0])
        for _ in range(size - 2):
            deriv_base = np.convolve(deriv_base, [1.0, 1.0])
        deriv = np.convolve(deriv_base, [1.0, -1.0])
        # deriv is now length-size

        assert len(smooth) == size and len(deriv) == size, \
            f"Kernel size mismatch: smooth={len(smooth)}, deriv={len(deriv)}, expected={size}"

        kx = np.outer(smooth, deriv).astype(np.float32)   # (size, size)
        ky = kx.T.copy()
        kx = torch.tensor(kx).unsqueeze(0).unsqueeze(0)   # (1, 1, size, size)
        ky = torch.tensor(ky).unsqueeze(0).unsqueeze(0)
        return kx, ky

    def forward(self, gray):
        """gray: (B, 1, H, W)"""
        maps = []
        for size in [3, 5, 7]:
            kx = getattr(self, f'sobel_x_{size}')
            ky = getattr(self, f'sobel_y_{size}')
            # padding = size // 2 guarantees output spatial size == input size
            # for any odd kernel (same-padding convention)
            p = size // 2
            gx = F.conv2d(gray, kx, padding=p)             # (B, 1, H, W)
            gy = F.conv2d(gray, ky, padding=p)             # (B, 1, H, W)
            magnitude = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
            direction = torch.atan2(gy, gx + 1e-8)
            maps.extend([magnitude, direction])

        return self.cnn(torch.cat(maps, dim=1))             # (B, 128)


# Enhanced Frequency Branch                                                   
# Magnitude + phase spectrum, plus a 1-D radial power spectrum feature.      
class FrequencyBranch(nn.Module):
    def __init__(self, num_radial_bins=32):
        super().__init__()
        self.num_radial_bins = num_radial_bins

        # Input: 2 channels (log-magnitude + phase)
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )  # → 96-d

        # Radial spectrum MLP: compresses binned 1-D power spectrum → 32-d
        self.radial_mlp = nn.Sequential(
            nn.Linear(num_radial_bins, 64),
            nn.GELU(),
            nn.Linear(64, 32),
        )

        # Fuse CNN features + radial features → 128-d
        self.fuse = nn.Sequential(
            nn.Linear(96 + 32, 128),
            nn.GELU(),
        )

    def _radial_power_spectrum(self, gray):
        """
        Compute azimuthally-averaged (radial) power spectrum.
        Captures spectral decay differences between real and synthetic images
        as described by Durall et al. (CVPR 2020).

        Returns: (B, num_radial_bins)
        """
        B, _, H, W = gray.shape
        fft2    = torch.fft.fft2(gray.squeeze(1))                 # (B, H, W)
        shifted = torch.fft.fftshift(fft2, dim=(-2, -1))
        power   = torch.abs(shifted) ** 2                         # (B, H, W)

        cy, cx = H // 2, W // 2
        y_idx  = torch.arange(H, device=gray.device).float() - cy
        x_idx  = torch.arange(W, device=gray.device).float() - cx
        # Radial distance for each pixel
        radius = torch.sqrt(
            y_idx.unsqueeze(1) ** 2 + x_idx.unsqueeze(0) ** 2
        )  # (H, W)
        max_r  = radius.max()
        # Bin each pixel into one of num_radial_bins equal-width rings
        bin_idx = (radius / (max_r + 1e-6) * self.num_radial_bins).long()
        bin_idx = bin_idx.clamp(0, self.num_radial_bins - 1)      # (H, W)

        # Accumulate mean power per radial bin
        radial = torch.zeros(B, self.num_radial_bins, device=gray.device)
        flat_power = power.view(B, -1)                             # (B, H*W)
        flat_bin   = bin_idx.view(-1)                              # (H*W,)
        radial.scatter_add_(1, flat_bin.unsqueeze(0).expand(B, -1), flat_power)

        # Normalize by bin count to get mean power
        counts = torch.bincount(flat_bin, minlength=self.num_radial_bins).float()
        radial = radial / (counts.unsqueeze(0) + 1e-6)

        return torch.log(radial + 1e-6)     

    def forward(self, gray):
        """gray: (B, 1, H, W)"""
        fft2 = torch.fft.fft2(gray)
        shifted = torch.fft.fftshift(fft2, dim=(-2, -1))

        log_mag = torch.log(torch.abs(shifted) + 1e-6)         # (B,1,H,W)
        phase = torch.angle(shifted)                           # (B,1,H,W)

        # 2-channel input: [log magnitude, phase]
        freq_input = torch.cat([log_mag, phase], dim=1)        # (B,2,H,W)

        cnn_feat = self.cnn(freq_input)                        # (B, 96)
        radial_feat = self.radial_mlp(
            self._radial_power_spectrum(gray)
        )                                                      # (B, 32)

        return self.fuse(torch.cat([cnn_feat, radial_feat], dim=1))  # (B, 128)


# SaFENet — Full Model                                                        
class SaFENet(nn.Module):
    def __init__(
        self,
        num_classes=2,
        branches=('spatial', 'gradient', 'frequency'),
        feat_dim=128,
        num_heads=4,
        dropout=0.3,
    ):
        """
        Args:
            num_classes:  Output classes (default 2 for binary detection).
            branches:     Any non-empty subset of {'spatial','gradient','frequency'}.
            feat_dim:     Common projection dimension in CMAF.
            num_heads:    Attention heads (must divide feat_dim).
            dropout:      Dropout rate in classifier.
        """
        super().__init__()

        branches = list(dict.fromkeys(branches))
        assert len(branches) >= 1
        assert all(b in VALID_BRANCHES for b in branches)
        self.active_branches = branches

        # * SPATIAL BACKBONE BRANCH *
        # uses unfrozen MobileNetV2 + CBAM attention + global pooling
        if 'spatial' in branches:
            mobilenet = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

            # Detach features from the rest of mobilenet cleanly
            self.spatial_features = mobilenet.features

            # Fully unfreeze — get_param_groups handles the lower LR
            for p in self.spatial_features.parameters():
                p.requires_grad = True

            del mobilenet  # explicitly free the orphaned classifier weights

            self.spatial_cbam = CBAM(channels=1280, reduction=16)
            self.spatial_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )    
        
        # * GRADIENT BRANCH *
        # computes multi-scale Sobel gradients + direction, processed by a small CNN
        if 'gradient' in branches:
            self.gradient_branch = GradientBranch()

        # * FREQUENCY BRANCH *
        # computes log-magnitude + phase + radial spectrum features, processed by a small CNN
        if 'frequency' in branches:
            self.frequency_branch = FrequencyBranch()

        # * CMAF *
        # Fuses the active branches with true cross-modal attention + FFN + gated fusion
        self.cmaf = CMAF(
            active_branches=branches,
            feat_dim=feat_dim,
            num_heads=num_heads,
        )

        # ------------------------------------------------------------------ #
        # 5. Classifier Head — 2-layer MLP with GELU + dropout               #
        # ------------------------------------------------------------------ #
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim // 2, num_classes),
        )

    # ---------------------------------------------------------------------- #
    # Helpers                                                                 #
    # ---------------------------------------------------------------------- #
    @staticmethod
    def _to_gray(x):
        """BT.601 luminance: RGB → grayscale (B,1,H,W)."""
        return (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])

    # ---------------------------------------------------------------------- #
    # Forward                                                                 #
    # ---------------------------------------------------------------------- #
    def forward(self, x):
        branch_features = {}

        if 'spatial' in self.active_branches:
            feat = self.spatial_features(x)          # (B, 1280, H', W')
            feat = self.spatial_cbam(feat)            # attention-weighted
            branch_features['spatial'] = self.spatial_pool(feat)  # (B, 1280)

        # Compute grayscale only once if needed by auxiliary branches
        if 'gradient' in self.active_branches or 'frequency' in self.active_branches:
            gray = self._to_gray(x)                  # (B, 1, H, W)

        if 'gradient' in self.active_branches:
            branch_features['gradient'] = self.gradient_branch(gray)  # (B, 128)

        if 'frequency' in self.active_branches:
            branch_features['frequency'] = self.frequency_branch(gray)  # (B, 128)

        fused = self.cmaf(branch_features)            # (B, feat_dim)
        return self.classifier(fused)                 # (B, num_classes)

    def __repr__(self):
        return (f"SaFENet(branches={self.active_branches}, "
                f"num_classes={self.classifier[-1].out_features})")


# =========================================================================== #
# Differential Learning Rate Helper                                           #
# Returns param groups with different LRs per component.                     #
# Usage: optimizer = Adam(get_param_groups(model), lr=1e-3)                  #
# =========================================================================== #
def get_param_groups(model, base_lr=1e-3):
    """
    Applies differential learning rates:
      - Spatial backbone fine-tuned layers:  base_lr * 0.1
      - Gradient / frequency CNNs:           base_lr * 0.5
      - CMAF + classifier:                   base_lr
    """
    backbone_params, aux_params, head_params = [], [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'spatial_features' in name or 'spatial_cbam' in name or 'spatial_pool' in name:
            backbone_params.append(param)
        elif 'gradient_branch' in name or 'frequency_branch' in name:
            aux_params.append(param)
        else:
            head_params.append(param)

    return [
        {'params': backbone_params, 'lr': base_lr * 0.3,  'name': 'spatial_backbone'},
        {'params': aux_params,      'lr': base_lr * 0.5,  'name': 'aux_branches'},
        {'params': head_params,     'lr': base_lr,         'name': 'cmaf_classifier'},
    ]


# =========================================================================== #
# Factory                                                                     #
# =========================================================================== #
def get_safenet(
    branches=('spatial', 'gradient', 'frequency'),
    num_classes=2,
    feat_dim=128,
):
    """
    Convenience factory for full model or ablation variants.

    Examples:
        get_safenet()                                          # full model
        get_safenet(branches=['spatial'])                      # spatial only
        get_safenet(branches=['spatial', 'frequency'])         # no gradient
        get_safenet(branches=['spatial', 'gradient'])          # no frequency
        get_safenet(branches=['gradient', 'frequency'])        # no backbone
    """
    return SaFENet(num_classes=num_classes, branches=branches, feat_dim=feat_dim)


def clip_gradients(optimizer):            # Clip each group separately — tighter on backbone, looser on aux
    torch.nn.utils.clip_grad_norm_(
        [p for g in optimizer.param_groups
        if g.get('name') == 'spatial_backbone'
        for p in g['params']],
        max_norm=1.0
    )
    torch.nn.utils.clip_grad_norm_(
        [p for g in optimizer.param_groups
        if g.get('name') in ('aux_branches', 'cmaf_classifier')
        for p in g['params']],
        max_norm=5.0
    )

def aux_Warmup(epoch, model: SaFENet, AUX_WARMUP_EPOCHS=3):
            # Freeze backbone during warmup so aux branches develop independently
        if epoch < AUX_WARMUP_EPOCHS:
            for p in model.spatial_features.parameters():
                p.requires_grad = False
            for p in model.spatial_cbam.parameters():
                p.requires_grad = False
        elif epoch == AUX_WARMUP_EPOCHS:
            # Unfreeze and let differential LR take over
            for p in model.spatial_features.parameters():
                p.requires_grad = True
            for p in model.spatial_cbam.parameters():
                p.requires_grad = True
            print(f"Epoch {epoch+1}: spatial backbone unfrozen")

# =========================================================================== #
# Quick sanity check                                                          #
# =========================================================================== #
if __name__ == '__main__':
    model = get_safenet()
    model.eval()

    dummy = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)

    print(model)
    print(f"\nInput:  {dummy.shape}")
    print(f"Output: {out.shape}")

    total   = sum(p.numel() for p in model.parameters())
    trained = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params:     {total:,}")
    print(f"Trainable params: {trained:,}")

    # Verify differential LR groups
    groups = get_param_groups(model)
    for g in groups:
        n = sum(p.numel() for p in g['params'])
        print(f"  [{g['name']}]  lr={g['lr']:.4f}  params={n:,}")