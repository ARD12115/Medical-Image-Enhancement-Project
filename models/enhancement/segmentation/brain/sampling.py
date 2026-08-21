"""
Foreground-Weighted Sampling for Segmentation
"""

import torch
from torch.utils.data import WeightedRandomSampler
import numpy as np


def compute_sample_weights(dataset, fg_weight: float = 3.0, bg_weight: float = 1.0):
    """
    Compute per-sample weights for WeightedRandomSampler.
    Slices containing lesion/ROI pixels get fg_weight, pure-background slices get bg_weight.
    """
    weights = []
    n_fg = 0
    
    for _, m in dataset.samples:
        has_fg = float(np.asarray(m, dtype=np.float32).sum()) > 0.0
        weights.append(fg_weight if has_fg else bg_weight)
        n_fg += int(has_fg)
    
    print(f"[sampling] Foreground-containing slices: {n_fg}/{len(dataset.samples)} "
          f"({100.0 * n_fg / max(1, len(dataset.samples)):.1f}%) — "
          f"fg_weight={fg_weight}, bg_weight={bg_weight}")
    
    return weights


def create_fg_sampler(dataset, fg_weight: float = 3.0, bg_weight: float = 1.0):
    """Create WeightedRandomSampler for foreground-weighted sampling"""
    weights = compute_sample_weights(dataset, fg_weight, bg_weight)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler