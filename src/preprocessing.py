"""
preprocessing.py — MedhaDrishti
Denoise → Rescale → CLAHE for Brain/Spine MRI NIfTI volumes.

Usage:
    python src/preprocessing.py --data data/brain --out data/brain_preprocessed
    python src/preprocessing.py --demo              # synthetic data test
"""

import argparse
import os
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

# ─────────────────────────── demo ──────────────────────────────

def make_demo_volume(shape=(256, 256, 30), seed=0) -> np.ndarray:
    """Synthetic noisy MRI-like volume."""
    rng = np.random.default_rng(seed)
    vol = np.zeros(shape, dtype=np.float32)
    cy, cx = shape[0] // 2, shape[1] // 2
    Y, X = np.ogrid[:shape[0], :shape[1]]
    brain_mask = ((X - cx)**2 / 90**2 + (Y - cy)**2 / 110**2) <= 1
    for z in range(shape[2]):
        sl = rng.integers(20, 80, size=shape[:2]).astype(np.float32)
        sl[brain_mask] += rng.integers(80, 160)
        noise = rng.normal(0, 20, shape[:2]).astype(np.float32)
        sl = np.clip(sl + noise, 0, 255)
        vol[:, :, z] = sl
    return vol

# ─────────────────────── preprocessing ops ─────────────────────

def denoise_slice(sl: np.ndarray) -> np.ndarray:
    """Non-Local Means denoising (best for MRI)."""
    u8 = sl.astype(np.uint8)
    return cv2.fastNlMeansDenoising(u8, h=10, templateWindowSize=7, searchWindowSize=21)


def rescale_slice(sl: np.ndarray) -> np.ndarray:
    """Normalize to [0, 255] uint8."""
    sl = sl.astype(np.float32)
    mn, mx = sl.min(), sl.max()
    if mx > mn:
        sl = (sl - mn) / (mx - mn) * 255.0
    return sl.astype(np.uint8)


def clahe_slice(sl: np.ndarray) -> np.ndarray:
    """Apply CLAHE for contrast enhancement — preferred for MRI."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(sl.astype(np.uint8))


def preprocess_slice(sl: np.ndarray) -> np.ndarray:
    """Full preprocessing pipeline for one slice."""
    sl = rescale_slice(sl)      # normalize first
    sl = denoise_slice(sl)      # NLM denoise
    sl = clahe_slice(sl)        # CLAHE
    return sl


def preprocess_volume(vol: np.ndarray) -> np.ndarray:
    """Apply preprocessing to every axial slice of a 3-D volume."""
    out = np.zeros_like(vol, dtype=np.uint8)
    for z in range(vol.shape[2]):
        out[:, :, z] = preprocess_slice(vol[:, :, z])
    return out

# ──────────────────────── visualisation ────────────────────────

def save_before_after(raw_sl: np.ndarray, proc_sl: np.ndarray,
                      scan_id: str, out_dir: str):
    """Side-by-side before/after figure for the report."""
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Preprocessing: {scan_id}", fontsize=11, fontweight="bold")
    for ax, img, title in zip(axes,
                               [raw_sl, proc_sl],
                               ["Raw (before)", "Preprocessed (after)"]):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{scan_id}_before_after.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path

# ────────────────────────────── I/O ────────────────────────────

def process_nifti(in_path: str, out_path: str, scan_id: str, plot_dir: str):
    """Load → preprocess → save NIfTI. Returns stats dict."""
    img = nib.load(in_path)
    data = img.get_fdata().astype(np.float32)
    if data.ndim == 4:
        data = data[..., 0]

    print(f"  Volume shape: {data.shape}")
    proc = preprocess_volume(data)

    # Save before/after for the middle axial slice
    mid = data.shape[2] // 2
    raw_sl = rescale_slice(data[:, :, mid])
    plot_path = save_before_after(raw_sl, proc[:, :, mid], scan_id, plot_dir)
    print(f"  Before/after plot -> {plot_path}")

    # Save processed NIfTI (preserve affine + header)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    new_img = nib.Nifti1Image(proc.astype(np.float32), img.affine, img.header)
    nib.save(new_img, out_path)
    print(f"  Saved preprocessed volume -> {out_path}")

    return {"scan_id": scan_id, "shape": list(data.shape), "preprocessed": out_path}


def process_demo(out_dir: str, plot_dir: str):
    """Run preprocessing on synthetic volumes."""
    print("[preprocessing] DEMO mode — using synthetic volumes.")
    records = []
    for i in range(3):
        scan_id = f"demo_{i:03d}"
        vol = make_demo_volume(seed=i)
        proc = preprocess_volume(vol)
        mid = vol.shape[2] // 2
        raw_sl = rescale_slice(vol[:, :, mid])
        plot_path = save_before_after(raw_sl, proc[:, :, mid], scan_id, plot_dir)
        print(f"  [{scan_id}] Before/after -> {plot_path}")
        # Save as numpy for downstream use
        np.save(os.path.join(out_dir, f"{scan_id}.npy"), proc)
        records.append({"scan_id": scan_id, "shape": list(vol.shape)})
    return records

# ────────────────────────────── main ───────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MedhaDrishti — Preprocessing")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--out", type=str, default="data/preprocessed")
    parser.add_argument("--plot-dir", type=str, default="results/plots/preprocessing")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    if args.demo:
        records = process_demo(args.out, args.plot_dir)
    else:
        if not HAS_NIBABEL:
            raise ImportError("nibabel not installed. Run: pip install nibabel")
        if args.data is None:
            raise ValueError("--data is required")
        nii_files = sorted(
            f for f in os.listdir(args.data) if f.endswith(".nii") or f.endswith(".nii.gz")
        )
        if not nii_files:
            raise FileNotFoundError(f"No NIfTI files found in {args.data}")
        print(f"[preprocessing] Found {len(nii_files)} volumes.")
        records = []
        for fname in nii_files:
            scan_id = fname.replace(".nii.gz", "").replace(".nii", "")
            in_path = os.path.join(args.data, fname)
            out_path = os.path.join(args.out, fname)
            print(f"\n  Processing {fname} …")
            rec = process_nifti(in_path, out_path, scan_id, args.plot_dir)
            records.append(rec)

    manifest = os.path.join(args.out, "manifest.json")
    with open(manifest, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\n[preprocessing] Done. Manifest -> {manifest}")


if __name__ == "__main__":
    main()
