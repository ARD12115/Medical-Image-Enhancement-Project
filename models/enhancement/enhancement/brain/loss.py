"""
Enhancement Loss Functions for Brain MRI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pytorch_msssim import ssim as pytorch_ssim_fn
    HAS_PYTORCH_MSSSIM = True
except ImportError:
    HAS_PYTORCH_MSSSIM = False

try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False


class EnhancementLoss(nn.Module):
    """
    Enhanced Combined Loss with:
    - L1 loss (pixel-wise)
    - SSIM (structural similarity)
    - Optional LPIPS (perceptual)
    - Charbonnier (robust L1)
    - Edge-aware (Sobel)
    """
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.1, delta=0.1,
                 use_lpips=True, use_charbonnier=True, use_edge=True,
                 device=None):
        super().__init__()
        self.alpha = alpha      # L1
        self.beta = beta        # SSIM
        self.gamma = gamma      # LPIPS
        self.delta = delta      # Charbonnier/Edge
        self.use_lpips = use_lpips
        self.use_charbonnier = use_charbonnier
        self.use_edge = use_edge
        self.device = device

        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

        if use_lpips and HAS_LPIPS:
            self.lpips = lpips.LPIPS(net='alex').to(device)
            self.lpips.eval()

        if use_edge and device is not None:
            self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                        dtype=torch.float32, device=device).view(1, 1, 3, 3)
            self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                        dtype=torch.float32, device=device).view(1, 1, 3, 3)

    def charbonnier_loss(self, pred, target, eps=1e-3):
        diff = pred - target
        # Clamp to prevent numerical instability
        diff = torch.clamp(diff, -1e4, 1e4)
        return torch.sqrt(diff ** 2 + eps ** 2).mean()

    def edge_loss(self, pred, target):
        pred_x = F.conv2d(pred, self.sobel_x, padding=1)
        pred_y = F.conv2d(pred, self.sobel_y, padding=1)
        target_x = F.conv2d(target, self.sobel_x, padding=1)
        target_y = F.conv2d(target, self.sobel_y, padding=1)
        pred_edge = torch.sqrt(pred_x**2 + pred_y**2 + 1e-8)
        target_edge = torch.sqrt(target_x**2 + target_y**2 + 1e-8)
        return F.l1_loss(pred_edge, target_edge)

    def forward(self, pred, target):
        loss = self.alpha * self.l1(pred, target)

        # SSIM
        try:
            if HAS_PYTORCH_MSSSIM:
                ssim_val = pytorch_ssim_fn(pred, target, data_range=1.0, size_average=True)
                if torch.isfinite(ssim_val):
                    loss += self.beta * (1.0 - ssim_val)
        except Exception:
            pass

        # LPIPS
        if self.use_lpips and self.gamma > 0 and HAS_LPIPS and hasattr(self, 'lpips') and self.lpips is not None:
            try:
                p3 = pred.repeat(1, 3, 1, 1) * 2 - 1
                t3 = target.repeat(1, 3, 1, 1) * 2 - 1
                lp = self.lpips(p3, t3).mean()
                if torch.isfinite(lp):
                    loss += self.gamma * lp
            except Exception:
                pass

        # Charbonnier
        if self.use_charbonnier and self.delta > 0:
            loss += self.delta * self.charbonnier_loss(pred, target)

        # Edge-aware
        if self.use_edge and self.delta > 0 and hasattr(self, 'sobel_x'):
            loss += self.delta * self.edge_loss(pred, target)

        return loss