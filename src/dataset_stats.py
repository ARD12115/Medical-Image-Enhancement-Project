"""
dataset_stats.py — MedhaDrishti
Compute per-scan contrast, sharpness, and noise statistics for Brain/Spine MRI.

Usage:
    python src/dataset_stats.py --data data/brain --out results/brain_stats.csv
    python src/dataset_stats.py --data data/spine --out results/spine_stats.csv
    python src/dataset_stats.py --demo              # synthetic data, no real NIfTI needed
"""

import argparse
import os
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Optional nibabel — skipped in demo mode
try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

# ─────────────────────────── helpers ───────────────────────────

def load_nifti_mid_slice(path: str) -> np.ndarray:
    """Load a NIfTI file and return the middle axial slice as uint8."""
    img = nib.load(path)
    data = img.get_fdata()
    if data.ndim == 4:          # 4-D (time series) → take first volume
        data = data[..., 0]
    mid = data.shape[2] // 2    # middle axial index
    sl = data[:, :, mid]
    sl = sl.astype(np.float32)
    if sl.max() > sl.min():
        sl = (sl - sl.min()) / (sl.max() - sl.min()) * 255
    return sl.astype(np.uint8)


def michelson_contrast(img: np.ndarray) -> float:
    """Michelson contrast on a normalised slice."""
    i_max = float(img.max())
    i_min = float(img.min())
    denom = i_max + i_min
    return (i_max - i_min) / denom if denom > 0 else 0.0


def laplacian_sharpness(img: np.ndarray) -> float:
    """Variance of the Laplacian — higher = sharper."""
    lap = cv2.Laplacian(img.astype(np.float64), cv2.CV_64F)
    return float(lap.var())


def background_noise(img: np.ndarray, crop: int = 20) -> float:
    """Std-dev of a top-left corner crop (assumed homogeneous background)."""
    region = img[:crop, :crop].astype(np.float64)
    return float(region.std())


# ──────────────────────────── demo ─────────────────────────────

def make_demo_slices(n: int = 6) -> list:
    """Generate synthetic grayscale slices mimicking T1/T2/FLAIR MRI appearance."""
    rng = np.random.default_rng(42)
    slices = []
    modalities = ["T1", "T2", "FLAIR", "T1", "T2", "STIR"]
    for i in range(n):
        base = rng.integers(30, 200, size=(256, 256), dtype=np.uint8)
        # Add a bright oval 'brain' region
        cy, cx = 128, 128
        Y, X = np.ogrid[:256, :256]
        mask = ((X - cx)**2 / 90**2 + (Y - cy)**2 / 110**2) <= 1
        base[mask] = np.clip(base[mask] + rng.integers(40, 80), 0, 255)
        # Add Gaussian noise
        noise = rng.normal(0, 15, base.shape).astype(np.int16)
        sl = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        slices.append({"scan_id": f"demo_{i:03d}", "modality": modalities[i % len(modalities)], "slice": sl})
    return slices


# ────────────────────────── core logic ─────────────────────────

def analyse_slice(scan_id: str, modality: str, sl: np.ndarray) -> dict:
    return {
        "scan_id": scan_id,
        "modality": modality,
        "contrast": round(michelson_contrast(sl), 4),
        "sharpness": round(laplacian_sharpness(sl), 4),
        "noise": round(background_noise(sl), 4),
    }


def save_annotated_plot(sl: np.ndarray, stats: dict, out_path: str):
    """Save a single annotated slice image."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(sl, cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"{stats['scan_id']} | {stats['modality']}", fontsize=9)
    ax.set_xlabel(
        f"Contrast={stats['contrast']:.3f}  Sharpness={stats['sharpness']:.1f}  Noise={stats['noise']:.2f}",
        fontsize=7
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close(fig)


def save_summary_plot(df: pd.DataFrame, out_dir: str):
    """Bar charts for contrast, sharpness, noise grouped by modality."""
    os.makedirs(out_dir, exist_ok=True)
    metrics = ["contrast", "sharpness", "noise"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("MRI Dataset Statistics — MedhaDrishti", fontsize=12, fontweight="bold")
    colors = ["#4C9BE8", "#E86B4C", "#4CE878"]
    for ax, metric, color in zip(axes, metrics, colors):
        grouped = df.groupby("modality")[metric].mean()
        bars = ax.bar(grouped.index, grouped.values, color=color, edgecolor="white", linewidth=0.8)
        ax.set_title(metric.capitalize(), fontsize=10)
        ax.set_ylabel(metric)
        ax.set_xlabel("Modality")
        ax.spines[["top", "right"]].set_visible(False)
        for bar, val in zip(bars, grouped.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "stats_summary.png"), dpi=150)
    plt.close(fig)
    print(f"  [stats] Summary plot saved -> {out_dir}/stats_summary.png")


# ────────────────────────────── main ───────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MedhaDrishti — Dataset Statistics")
    parser.add_argument("--data", type=str, default=None,
                        help="Directory containing .nii / .nii.gz files")
    parser.add_argument("--out", type=str, default="results/stats.csv",
                        help="Output CSV path")
    parser.add_argument("--plot-dir", type=str, default="results/plots/stats",
                        help="Directory for annotated slice images")
    parser.add_argument("--demo", action="store_true",
                        help="Use synthetic data (no real NIfTI files needed)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else "results", exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    records = []

    if args.demo:
        print("[dataset_stats] Running in DEMO mode with synthetic data.")
        entries = make_demo_slices(n=6)
        for entry in entries:
            stats = analyse_slice(entry["scan_id"], entry["modality"], entry["slice"])
            records.append(stats)
            plot_path = os.path.join(args.plot_dir, f"{entry['scan_id']}.png")
            save_annotated_plot(entry["slice"], stats, plot_path)
            print(f"  [stats] {stats}")
    else:
        if not HAS_NIBABEL:
            raise ImportError("nibabel not installed. Run: pip install nibabel")
        if args.data is None:
            raise ValueError("--data required when not in --demo mode")
        nii_files = sorted(
            f for f in os.listdir(args.data) if f.endswith(".nii") or f.endswith(".nii.gz")
        )
        if not nii_files:
            raise FileNotFoundError(f"No .nii/.nii.gz files found in {args.data}")
        print(f"[dataset_stats] Found {len(nii_files)} volumes in {args.data}")
        for fname in nii_files:
            fpath = os.path.join(args.data, fname)
            scan_id = fname.replace(".nii.gz", "").replace(".nii", "")
            # Try to guess modality from filename
            modality = "unknown"
            for tag in ["T1", "T2", "FLAIR", "STIR"]:
                if tag.lower() in fname.lower():
                    modality = tag
                    break
            print(f"  Loading {fname} …")
            try:
                sl = load_nifti_mid_slice(fpath)
                stats = analyse_slice(scan_id, modality, sl)
                records.append(stats)
                plot_path = os.path.join(args.plot_dir, f"{scan_id}.png")
                save_annotated_plot(sl, stats, plot_path)
                print(f"  [stats] {stats}")
            except Exception as e:
                print(f"  [WARN] Failed to process {fname}: {e}")

    if not records:
        print("[dataset_stats] No records produced.")
        return

    df = pd.DataFrame(records)
    df.to_csv(args.out, index=False)
    print(f"\n[dataset_stats] CSV saved -> {args.out}")
    print(df.to_string(index=False))

    # Save summary bar charts
    save_summary_plot(df, args.plot_dir)

    # Also write JSON sidecar
    json_out = args.out.replace(".csv", ".json")
    with open(json_out, "w") as f:
        json.dump(records, f, indent=2)
    print(f"[dataset_stats] JSON saved -> {json_out}")


if __name__ == "__main__":
    main()
