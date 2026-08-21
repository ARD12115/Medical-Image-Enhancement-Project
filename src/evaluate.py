"""
evaluate.py — MedhaDrishti
Consolidates results from enhancement + segmentation into results/results.json
and generates a final comparison report figure.

Usage:
    python src/evaluate.py --models models/ --data data/brain_preprocessed --out results/results.json
    python src/evaluate.py --demo
"""

import argparse
import os
import json
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

try:
    from enhancement_model  import ResNet18Enhancer, DemoDataset as DemoEnhDataset, compute_psnr_ssim
    from segmentation_model import UNet, DemoSegDataset, dice_score, jaccard_score
    IMPORTS_OK = True
except ImportError:
    IMPORTS_OK = False
    print("[evaluate] WARNING: Could not import model modules. Run from project root.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════════════════ INFERENCE HELPERS ═════════════════════

@torch.no_grad()
def run_enhancement_inference(model, dataset, n_samples: int = 20):
    """Run enhancement model on n_samples and return per-sample metrics."""
    model.eval()
    records = []
    indices = np.linspace(0, len(dataset) - 1, min(n_samples, len(dataset)), dtype=int)
    for i, idx in enumerate(indices):
        noisy, clean = dataset[idx]
        pred = model(noisy.unsqueeze(0).to(DEVICE)).cpu().squeeze().numpy()
        clean_np = clean.squeeze().numpy()
        noisy_np = noisy.squeeze().numpy()
        psnr_enhanced, ssim_enhanced = compute_psnr_ssim(pred,     clean_np)
        psnr_baseline, ssim_baseline = compute_psnr_ssim(noisy_np, clean_np)
        records.append({
            "sample_id": int(idx),
            "baseline_psnr": round(psnr_baseline, 3),
            "baseline_ssim": round(ssim_baseline, 4),
            "enhanced_psnr": round(psnr_enhanced, 3),
            "enhanced_ssim": round(ssim_enhanced, 4),
        })
    return records


@torch.no_grad()
def run_segmentation_inference(model, dataset, n_samples: int = 20):
    """Run U-Net on n_samples and return per-sample Dice/Jaccard."""
    model.eval()
    records = []
    indices = np.linspace(0, len(dataset) - 1, min(n_samples, len(dataset)), dtype=int)
    for i, idx in enumerate(indices):
        img, gt = dataset[idx]
        logit = model(img.unsqueeze(0).to(DEVICE))
        pred  = (torch.sigmoid(logit) > 0.5).float().cpu().squeeze().numpy()
        gt_np = gt.squeeze().numpy()
        records.append({
            "sample_id": int(idx),
            "dice":    round(dice_score(pred, gt_np), 4),
            "jaccard": round(jaccard_score(pred, gt_np), 4),
        })
    return records


# ════════════════════════ FINAL REPORT FIGURE ══════════════════

def save_final_report_figure(
    enh_model, enh_ds,
    seg_model, seg_ds,
    enh_records, seg_records,
    out_path: str,
):
    """
    A single publication-ready figure for the report & PPT:
    Row 1: Enhancement comparison (4 samples)
    Row 2: Segmentation comparison (4 samples)
    Row 3: Metric bar charts
    """
    fig = plt.figure(figsize=(20, 18))
    fig.patch.set_facecolor("#0F1117")
    gs = gridspec.GridSpec(3, 8, figure=fig, hspace=0.45, wspace=0.15)

    title_kw = dict(color="white", fontsize=11, fontweight="bold")
    label_kw = dict(color="#AAAAAA", fontsize=8)

    # ── Row 1: Enhancement ──────────────────────────────────
    fig.text(0.01, 0.97, "MRI ENHANCEMENT  —  ResNet18 Encoder + Decoder", fontsize=13,
             fontweight="bold", color="#4C9BE8", va="top")
    enh_idx = np.linspace(0, len(enh_ds) - 1, 4, dtype=int)
    enh_model.eval()
    with torch.no_grad():
        for col, idx in enumerate(enh_idx):
            noisy, clean = enh_ds[idx]
            pred = enh_model(noisy.unsqueeze(0).to(DEVICE)).cpu().squeeze().numpy()
            psnr, ssim = compute_psnr_ssim(pred, clean.squeeze().numpy())
            for sub_col, (img, lbl) in enumerate([
                (noisy.squeeze().numpy(), "Input"),
                (pred,                    f"Enhanced\n▲PSNR={psnr:.1f}dB"),
            ]):
                ax = fig.add_subplot(gs[0, col * 2 + sub_col])
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                ax.set_title(lbl, **label_kw, pad=3)
                ax.axis("off")
                ax.set_facecolor("#0F1117")

    # ── Row 2: Segmentation ─────────────────────────────────
    fig.text(0.01, 0.65, "ROI SEGMENTATION  —  U-Net (4-level, Dice+BCE loss)", fontsize=13,
             fontweight="bold", color="#E86B4C", va="top")
    seg_idx = np.linspace(0, len(seg_ds) - 1, 4, dtype=int)
    seg_model.eval()
    with torch.no_grad():
        for col, idx in enumerate(seg_idx):
            img, gt = seg_ds[idx]
            logit = seg_model(img.unsqueeze(0).to(DEVICE))
            pred  = (torch.sigmoid(logit) > 0.5).float().cpu().squeeze().numpy()
            d = dice_score(pred, gt.squeeze().numpy())
            j = jaccard_score(pred, gt.squeeze().numpy())
            for sub_col, (overlay, mask, lbl) in enumerate([
                (img.squeeze().numpy(), gt.squeeze().numpy(),   "Ground Truth"),
                (img.squeeze().numpy(), pred,                   f"Prediction\nDice={d:.3f} IoU={j:.3f}"),
            ]):
                ax = fig.add_subplot(gs[1, col * 2 + sub_col])
                ax.imshow(overlay, cmap="gray", vmin=0, vmax=1)
                ax.imshow(mask, cmap="Reds", alpha=0.45, vmin=0, vmax=1)
                ax.set_title(lbl, **label_kw, pad=3)
                ax.axis("off")
                ax.set_facecolor("#0F1117")

    # ── Row 3: Metric charts ─────────────────────────────────
    # Enhancement metrics
    ax_psnr = fig.add_subplot(gs[2, :2])
    ax_ssim = fig.add_subplot(gs[2, 2:4])
    ax_dice = fig.add_subplot(gs[2, 4:6])
    ax_jacc = fig.add_subplot(gs[2, 6:])

    def styled_bar(ax, labels, values, title, color, ylabel):
        bars = ax.bar(labels, values, color=color, edgecolor="#333", linewidth=0.5)
        ax.set_title(title, color="white", fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, color="#AAAAAA", fontsize=8)
        ax.set_facecolor("#1A1D27")
        ax.tick_params(colors="#AAAAAA", labelsize=7)
        ax.spines[:].set_edgecolor("#333")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.yaxis.set_tick_params(color="#444")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                    f"{val:.2f}", ha="center", va="bottom", color="white", fontsize=8)

    psnr_base = np.mean([r["baseline_psnr"] for r in enh_records])
    psnr_enh  = np.mean([r["enhanced_psnr"] for r in enh_records])
    ssim_base = np.mean([r["baseline_ssim"] for r in enh_records])
    ssim_enh  = np.mean([r["enhanced_ssim"] for r in enh_records])
    mean_dice = np.mean([r["dice"]    for r in seg_records])
    mean_jacc = np.mean([r["jaccard"] for r in seg_records])

    styled_bar(ax_psnr, ["Baseline", "Enhanced"], [psnr_base, psnr_enh],
               "PSNR (dB) ↑", ["#555", "#4C9BE8"], "dB")
    styled_bar(ax_ssim, ["Baseline", "Enhanced"], [ssim_base, ssim_enh],
               "SSIM ↑",      ["#555", "#4C9BE8"], "")
    styled_bar(ax_dice, ["Dice"],    [mean_dice], "Dice Coefficient ↑", ["#E86B4C"], "")
    styled_bar(ax_jacc, ["Jaccard"], [mean_jacc], "Jaccard / IoU ↑",   ["#4CE878"], "")

    # Footer
    fig.text(0.5, 0.01,
             f"MedhaDrishti · Yugma TechFest 2.0 · Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",
             ha="center", color="#666", fontsize=8)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Final report figure -> {out_path}")


# ════════════════════════════ MAIN ════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MedhaDrishti — Evaluation")
    parser.add_argument("--models",   type=str, default="models")
    parser.add_argument("--data",     type=str, default=None)
    parser.add_argument("--out",      type=str, default="results/results.json")
    parser.add_argument("--plot-dir", type=str, default="results/plots")
    parser.add_argument("--demo",     action="store_true")
    args = parser.parse_args()

    if not IMPORTS_OK:
        print("[evaluate] ERROR: Cannot import model modules. Run from project root with:")
        print("  python src/evaluate.py --demo")
        return

    os.makedirs(args.plot_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else "results", exist_ok=True)

    # ── Load datasets ────────────────────────────────────────
    if args.demo:
        print("[evaluate] DEMO mode.")
        enh_ds = DemoEnhDataset(n_samples=40,  size=256, seed=42)
        seg_ds = DemoSegDataset(n=40,           size=256, seed=42)
    else:
        from enhancement_model  import MRISliceDataset
        from segmentation_model import MRISegDataset
        if args.data is None:
            raise ValueError("--data is required")
        enh_ds = MRISliceDataset(args.data, size=256)
        seg_ds = MRISegDataset(args.data, size=256, use_synthetic_masks=True)

    # ── Load models ──────────────────────────────────────────
    enh_path = os.path.join(args.models, "enhancer.pt")
    seg_path = os.path.join(args.models, "unet.pt")

    enh_model = ResNet18Enhancer(pretrained=False).to(DEVICE)
    if os.path.exists(enh_path):
        enh_model.load_state_dict(torch.load(enh_path, map_location=DEVICE, weights_only=True))
        print(f"  Loaded enhancement model ← {enh_path}")
    else:
        print(f"  [WARN] {enh_path} not found — using untrained model (run enhancement_model.py first)")

    seg_model = UNet(in_ch=1, base_ch=64).to(DEVICE)
    if os.path.exists(seg_path):
        seg_model.load_state_dict(torch.load(seg_path, map_location=DEVICE, weights_only=True))
        print(f"  Loaded segmentation model ← {seg_path}")
    else:
        print(f"  [WARN] {seg_path} not found — using untrained model (run segmentation_model.py first)")

    # ── Run inference ────────────────────────────────────────
    print("\n[evaluate] Running enhancement inference …")
    enh_records = run_enhancement_inference(enh_model, enh_ds, n_samples=20)
    print("[evaluate] Running segmentation inference …")
    seg_records = run_segmentation_inference(seg_model, seg_ds, n_samples=20)

    # ── Aggregate statistics ──────────────────────────────────
    agg = {
        "generated_at": datetime.now().isoformat(),
        "device": str(DEVICE),
        "enhancement": {
            "n_samples": len(enh_records),
            "mean_baseline_psnr_db": round(float(np.mean([r["baseline_psnr"] for r in enh_records])), 3),
            "mean_baseline_ssim":    round(float(np.mean([r["baseline_ssim"] for r in enh_records])), 4),
            "mean_enhanced_psnr_db": round(float(np.mean([r["enhanced_psnr"] for r in enh_records])), 3),
            "mean_enhanced_ssim":    round(float(np.mean([r["enhanced_ssim"] for r in enh_records])), 4),
            "delta_psnr_db":         round(float(np.mean([r["enhanced_psnr"] - r["baseline_psnr"] for r in enh_records])), 3),
            "delta_ssim":            round(float(np.mean([r["enhanced_ssim"] - r["baseline_ssim"] for r in enh_records])), 4),
            "per_sample": enh_records,
        },
        "segmentation": {
            "n_samples": len(seg_records),
            "mean_dice":    round(float(np.mean([r["dice"]    for r in seg_records])), 4),
            "mean_jaccard": round(float(np.mean([r["jaccard"] for r in seg_records])), 4),
            "per_sample": seg_records,
        },
    }

    # Merge with existing
    existing = {}
    if os.path.exists(args.out):
        try:
            with open(args.out) as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    existing.update(agg)
    with open(args.out, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\n[evaluate] results.json -> {args.out}")

    # ── Console summary table ─────────────────────────────────
    print("\n" + "═" * 52)
    print("  MEDHADRISHTI — FINAL RESULTS SUMMARY")
    print("═" * 52)
    e = agg["enhancement"]
    s = agg["segmentation"]
    print(f"  ENHANCEMENT  (n={e['n_samples']})")
    print(f"    Baseline PSNR : {e['mean_baseline_psnr_db']:.2f} dB")
    print(f"    Enhanced PSNR : {e['mean_enhanced_psnr_db']:.2f} dB  (Δ {e['delta_psnr_db']:+.2f} dB)")
    print(f"    Baseline SSIM : {e['mean_baseline_ssim']:.4f}")
    print(f"    Enhanced SSIM : {e['mean_enhanced_ssim']:.4f}  (Δ {e['delta_ssim']:+.4f})")
    print(f"\n  SEGMENTATION  (n={s['n_samples']})")
    print(f"    Dice    : {s['mean_dice']:.4f}")
    print(f"    Jaccard : {s['mean_jaccard']:.4f}")
    print("═" * 52)

    # ── Final report figure ───────────────────────────────────
    report_fig = os.path.join(args.plot_dir, "final_report_figure.png")
    save_final_report_figure(
        enh_model, enh_ds,
        seg_model, seg_ds,
        enh_records, seg_records,
        report_fig,
    )
    print(f"\n[evaluate] Done. All outputs in {os.path.dirname(args.out)}/")


if __name__ == "__main__":
    main()
