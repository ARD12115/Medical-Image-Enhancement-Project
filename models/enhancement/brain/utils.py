"""
Enhancement Model Utilities for Brain MRI
"""

import torch
import numpy as np
import cv2

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False


def compute_volume_percentiles(vol: np.ndarray, low_p: float = 0.5, high_p: float = 99.5) -> dict:
    """Compute percentile-based normalization parameters"""
    v = np.asarray(vol, dtype=np.float32)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    low = float(np.percentile(v, low_p))
    high = float(np.percentile(v, high_p))
    if high <= low:
        high = low + 1e-6
    return {"low": low, "high": high}


def normalize_volume(vol: np.ndarray, params: dict) -> np.ndarray:
    """Normalize volume using percentile parameters"""
    low, high = params["low"], params["high"]
    v = np.asarray(vol, dtype=np.float32)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    denom = (high - low) if (high - low) != 0 else 1e-6
    v = np.clip((v - low) / denom, 0.0, 1.0)
    return v.astype(np.float32)


def invert_normalize(vol_norm: np.ndarray, params: dict) -> np.ndarray:
    """Denormalize volume back to original intensity range"""
    low, high = params["low"], params["high"]
    return (vol_norm * (high - low) + low).astype(np.float32)


def save_nifti_float32(data: np.ndarray, reference_nifti_path: str, out_path: str):
    """Save array as NIfTI using reference file's affine/header"""
    if not HAS_NIBABEL:
        raise RuntimeError("nibabel required to save NIfTI files")
    ref = nib.load(reference_nifti_path)
    new_img = nib.Nifti1Image(data.astype(np.float32), ref.affine, ref.header)
    nib.save(new_img, out_path)


def preprocess_slice(slice_2d: np.ndarray, size: int = 224) -> np.ndarray:
    """Preprocess a single 2D slice for model input"""
    # Normalize to [0, 1]
    slice_norm = slice_2d.astype(np.float32)
    slice_norm = np.nan_to_num(slice_norm, nan=0.0, posinf=0.0, neginf=0.0)
    minv, maxv = slice_norm.min(), slice_norm.max()
    if maxv > minv + 1e-12:
        slice_norm = (slice_norm - minv) / (maxv - minv)
    else:
        slice_norm = np.zeros_like(slice_norm)
    slice_norm = np.clip(slice_norm, 0.0, 1.0)
    
    # Resize to model input size
    slice_resized = cv2.resize(slice_norm, (size, size), interpolation=cv2.INTER_LINEAR)
    return slice_resized.astype(np.float32)


def postprocess_slice(slice_2d: np.ndarray, orig_shape: tuple) -> np.ndarray:
    """Postprocess model output slice back to original dimensions"""
    return cv2.resize(slice_2d, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_LINEAR)


def compute_psnr_ssim(pred: np.ndarray, target: np.ndarray) -> tuple:
    """Compute PSNR and SSIM metrics"""
    from skimage.metrics import peak_signal_noise_ratio as calc_psnr
    from skimage.metrics import structural_similarity as calc_ssim
    
    PSNR_CAP_DB = 60.0
    pred_u8, target_u8 = None, None
    try:
        p = np.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
        t = np.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0)
        p = np.clip(p, 0.0, 1.0)
        t = np.clip(t, 0.0, 1.0)
        pred_u8 = (p * 255.0).round().astype(np.uint8)
        target_u8 = (t * 255.0).round().astype(np.uint8)
        psnr = calc_psnr(target_u8, pred_u8, data_range=255)
        if not np.isfinite(psnr):
            psnr = PSNR_CAP_DB
    except Exception:
        psnr = 0.0
    
    try:
        if pred_u8 is not None and target_u8 is not None:
            ssim = calc_ssim(target_u8, pred_u8, data_range=255)
        else:
            ssim = 0.0
        if not np.isfinite(ssim):
            ssim = 0.0
    except Exception:
        ssim = 0.0
    
    return float(psnr), float(ssim)