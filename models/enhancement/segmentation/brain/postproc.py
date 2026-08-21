"""
Post-Processing for Segmentation: Threshold Sweep & Blob Removal
"""

import torch
import torch.nn.functional as F
import numpy as np

try:
    from scipy import ndimage as _ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from scipy.spatial.distance import directed_hausdorff
    HAS_HAUSDORFF = True
except ImportError:
    HAS_HAUSDORFF = False


def dice_score(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-5) -> float:
    inter = (pred * gt).sum()
    return float((2 * inter + smooth) / (pred.sum() + gt.sum() + smooth))


def jaccard_score(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-5) -> float:
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum() - inter
    return float((inter + smooth) / (union + smooth))


def remove_small_blobs(mask: np.ndarray, min_area: int = 6) -> np.ndarray:
    """
    Drop connected components smaller than min_area pixels.
    Cheap, effective clean-up of noise-driven false-positive specks.
    """
    if not HAS_SCIPY or min_area <= 0:
        return mask
    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return mask
    labeled, num = _ndimage.label(m)
    if num == 0:
        return mask
    sizes = _ndimage.sum(m, labeled, index=np.arange(1, num + 1))
    keep = np.zeros_like(m)
    for comp_id, size in enumerate(sizes, start=1):
        if size >= min_area:
            keep[labeled == comp_id] = 1
    return keep.astype(mask.dtype)


def hausdorff_distance(pred: np.ndarray, gt: np.ndarray) -> float:
    """Approximate Hausdorff distance using boundary points."""
    if not HAS_HAUSDORFF:
        return float("nan")
    p_pts = np.argwhere(pred > 0)
    g_pts = np.argwhere(gt > 0)
    if p_pts.size == 0 or g_pts.size == 0:
        return float("nan")
    d1 = directed_hausdorff(p_pts, g_pts)[0]
    d2 = directed_hausdorff(g_pts, p_pts)[0]
    return float(max(d1, d2))


def predict_prob(model, x: torch.Tensor, use_tta: bool = False) -> torch.Tensor:
    """Sigmoid probability map, optionally averaged over TTA flips."""
    if not use_tta:
        return torch.sigmoid(model(x))
    p0 = torch.sigmoid(model(x))
    p1 = torch.sigmoid(model(torch.flip(x, [-1])))
    p1 = torch.flip(p1, [-1])
    p2 = torch.sigmoid(model(torch.flip(x, [-2])))
    p2 = torch.flip(p2, [-2])
    return (p0 + p1 + p2) / 3.0


def sweep_best_threshold(model, loader, use_tta: bool = False,
                         thresholds: list = None) -> tuple:
    """
    Sweep candidate thresholds on validation set and return the one
    that maximizes mean Dice. Probabilities computed once, reused across thresholds.
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.30, 0.71, 0.05)]
    model.eval()
    cached = []
    for imgs, masks in loader:
        imgs = imgs.to(imgs.device, memory_format=torch.channels_last)
        probs = predict_prob(model, imgs, use_tta=use_tta).cpu().numpy()
        gts = masks.numpy()
        for p, g in zip(probs, gts):
            cached.append((p[0], g[0]))

    best_t, best_dice = 0.5, -1.0
    dice_at_default = None
    for t in thresholds:
        dices = [dice_score((p > t).astype(np.float32), g) for p, g in cached]
        mean_d = float(np.mean(dices)) if dices else 0.0
        if abs(t - 0.5) < 1e-6:
            dice_at_default = mean_d
        if mean_d > best_dice:
            best_dice, best_t = mean_d, float(t)
    ref = f"{dice_at_default:.4f}" if dice_at_default is not None else "n/a"
    print(f"  [threshold-sweep] Best threshold={best_t:.2f}  Dice={best_dice:.4f}  (fixed-0.5 Dice={ref})")
    return best_t, best_dice