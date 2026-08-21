"""
Spine Segmentation - Re-exports Brain Components with Spine-Specific Config

Spine segmentation shares architecture with brain but may use different
hyperparameters for vertebrae/disc segmentation.
"""

from .brain import (
    SegmentationUNet,
    UNet,
    AttentionGate,
    BlurPool2d,
    DiceLoss,
    FocalLoss,
    BoundaryLoss,
    CombinedLoss,
    compute_sample_weights,
    remove_small_blobs,
    predict_prob,
    sweep_best_threshold,
    save_segmentation_grid,
    save_loss_curve,
    infer_volume,
    compute_volume_percentiles,
    normalize_volume,
    invert_normalize,
    save_nifti_float32,
    save_nifti,
    load_nifti,
    dice_score,
    jaccard_score,
    hausdorff_distance,
    BlurPool2d,
    AttentionGate,
)

# Spine-specific defaults
SPINE_DEFAULTS = {
    "base_ch": 64,
    "dropout": 0.1,
    "use_blurpool": True,
    "use_attention": True,
    "deep_supervision": True,
    "fg_weight": 3.0,
    "min_blob_area": 6,
}

def get_spine_unet(**kwargs):
    """Create U-Net with spine-optimized defaults"""
    from .brain import SegmentationUNet
    config = {**SPINE_DEFAULTS, **kwargs}
    return SegmentationUNet(**config)


__all__ = [
    "SegmentationUNet",
    "UNet",
    "AttentionGate",
    "BlurPool2d",
    "DiceLoss",
    "FocalLoss",
    "BoundaryLoss",
    "CombinedLoss",
    "compute_sample_weights",
    "remove_small_blobs",
    "predict_prob",
    "sweep_best_threshold",
    "save_segmentation_grid",
    "save_loss_curve",
    "infer_volume",
    "compute_volume_percentiles",
    "normalize_volume",
    "invert_normalize",
    "save_nifti_float32",
    "save_nifti",
    "load_nifti",
    "dice_score",
    "jaccard_score",
    "hausdorff_distance",
    "BlurPool2d",
    "AttentionGate",
    "get_spine_unet",
    "SPINE_DEFAULTS",
]