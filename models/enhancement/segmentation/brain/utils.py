"""
Segmentation Utilities for Brain MRI
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False


def compute_volume_percentiles(vol: np.ndarray, low_p: float = 0.5, high_p: float = 99.5) -> dict:
    """Compute percentile-based normalization parameters"""
    low = float(np.percentile(vol, low_p))
    high = float(np.percentile(vol, high_p))
    if high <= low:
        high = low + 1e-6
    return {"low": low, "high": high}


def normalize_volume(vol: np.ndarray, params: dict) -> np.ndarray:
    """Normalize volume using percentile parameters"""
    low, high = params["low"], params["high"]
    return np.clip((vol - low) / (high - low), 0.0, 1.0).astype(np.float32)


def invert_normalize(vol_norm: np.ndarray, params: dict) -> np.ndarray:
    """Denormalize volume back to original intensity range"""
    low, high = params["low"], params["high"]
    return (vol_norm * (high - low) + low).astype(np.float32)


def save_nifti_float32(data: np.ndarray, reference_nifti_path: str, out_path: str):
    """Save array as NIfTI using reference file's affine/header"""
    if not HAS_NIBABEL:
        raise RuntimeError("nibabel required for NIfTI I/O")
    ref = nib.load(reference_nifti_path)
    nib.save(nib.Nifti1Image(data.astype(np.float32), ref.affine, ref.header), out_path)


def save_nifti(data: np.ndarray, reference_path: str, out_path: str):
    """Save segmentation mask as NIfTI"""
    if not HAS_NIBABEL:
        raise RuntimeError("nibabel required for NIfTI I/O")
    ref = nib.load(reference_path)
    nib.save(nib.Nifti1Image(data.astype(np.float32), ref.affine, ref.header), out_path)


def load_nifti(path: str):
    """Load NIfTI file"""
    if not HAS_NIBABEL:
        raise RuntimeError("nibabel required for NIfTI I/O")
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    return img, data