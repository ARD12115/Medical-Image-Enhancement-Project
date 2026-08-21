"""
Segmentation Loss Functions for Brain Tumor Segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        inter = (probs_flat * targets_flat).sum(dim=1)
        denom = probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth
        dice_per_sample = (2.0 * inter + self.smooth) / denom
        return 1.0 - dice_per_sample.mean()


class FocalLoss(nn.Module):
    """
    Binary focal loss — down-weights easy (background) pixels
    and focuses training on hard/rare pixels (tiny lesions).
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class BoundaryLoss(nn.Module):
    """Boundary loss using distance transform for better boundary delineation."""
    def __init__(self):
        super().__init__()

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # probs: (B, 1, H, W) in [0,1], targets: (B, 1, H, W) in {0,1}
        with torch.no_grad():
            # Compute boundary mask using gradient
            kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                    dtype=torch.float32, device=targets.device).view(1, 1, 3, 3)
            kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                    dtype=torch.float32, device=targets.device).view(1, 1, 3, 3)
            tx = F.conv2d(targets, kernel_x, padding=1)
            ty = F.conv2d(targets, kernel_y, padding=1)
            boundary_mask = ((tx**2 + ty**2).sqrt() > 0.5).float()
            # Distance transform approximation
            boundary_dist = F.max_pool2d(boundary_mask, 3, stride=1, padding=1) - boundary_mask
            boundary_dist = (boundary_dist > 0).float()
        
        # Weighted BCE with boundary emphasis
        weight = 1.0 + boundary_dist * 4.0  # 5x weight near boundaries
        loss = F.binary_cross_entropy_with_logits(torch.zeros_like(targets), targets, weight=weight, reduction='mean')
        return loss


class CombinedLoss(nn.Module):
    """Dice + BCE, with optional Focal and Boundary terms for tiny-lesion class imbalance."""
    def __init__(self, use_focal: bool = False, focal_weight: float = 0.3,
                 use_boundary: bool = False, boundary_weight: float = 0.1):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.use_focal = use_focal
        self.focal_weight = focal_weight
        self.focal = FocalLoss() if use_focal else None
        self.use_boundary = use_boundary
        self.boundary_weight = boundary_weight
        if use_boundary:
            self.boundary_loss = BoundaryLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = 0.5 * self.dice(logits, targets) + 0.5 * self.bce(logits, targets)
        if self.focal is not None:
            loss = (1 - self.focal_weight) * loss + self.focal_weight * self.focal(logits, targets)
        if self.use_boundary:
            probs = torch.sigmoid(logits)
            loss += self.boundary_weight * self.boundary_loss(probs, targets)
        return loss


# Convenience function
def get_segmentation_loss(use_focal=False, focal_weight=0.3, use_boundary=False, boundary_weight=0.1):
    """Factory function to create segmentation loss"""
    return CombinedLoss(
        use_focal=use_focal,
        focal_weight=focal_weight,
        use_boundary=use_boundary,
        boundary_weight=boundary_weight
    )