"""
segmentation_model.py — MedhaDrishti
Standard U-Net (4 levels) for Brain/Spine MRI ROI segmentation.
Trains on Brain T2 + tumor class (BRATS format or synthetic demo masks).
Evaluates with Dice & Jaccard.

Usage:
    python src/segmentation_model.py --data data/brain_preprocessed --epochs 15 --out models/unet.pt
    python src/segmentation_model.py --demo --epochs 5
"""

import argparse
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[segmentation] Using device: {DEVICE}")

# ═══════════════════════════ U-NET ════════════════════════════

class ConvBlock(nn.Module):
    """Double Conv → BN → ReLU."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample → concat skip → ConvBlock."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if spatial sizes differ by 1 pixel
        if x.shape != skip.shape:
            x = nn.functional.interpolate(x, size=skip.shape[-2:],
                                           mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """
    Standard 4-level U-Net.
    Input : (B, 1, H, W)  — single-channel MRI slice
    Output: (B, 1, H, W)  — raw logits (apply sigmoid for probability)
    """
    def __init__(self, in_ch: int = 1, base_ch: int = 64):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_ch,       base_ch)        # H     → 64ch
        self.enc2 = ConvBlock(base_ch,     base_ch * 2)    # H/2   → 128ch
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)    # H/4   → 256ch
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8)    # H/8   → 512ch
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 16)  # H/16 → 1024ch

        # Decoder
        self.up4 = UpBlock(base_ch * 16, base_ch * 8,  base_ch * 8)
        self.up3 = UpBlock(base_ch * 8,  base_ch * 4,  base_ch * 4)
        self.up2 = UpBlock(base_ch * 4,  base_ch * 2,  base_ch * 2)
        self.up1 = UpBlock(base_ch * 2,  base_ch,      base_ch)

        # Segmentation head
        self.head = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        # Bottleneck
        b  = self.bottleneck(self.pool(s4))
        # Decoder (with skip connections)
        d = self.up4(b,  s4)
        d = self.up3(d,  s3)
        d = self.up2(d,  s2)
        d = self.up1(d,  s1)
        return self.head(d)   # raw logits


# ═══════════════════════════ LOSS ═════════════════════════════

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        num   = 2 * (probs * targets).sum()
        den   = probs.sum() + targets.sum() + self.smooth
        return 1 - num / den


class CombinedLoss(nn.Module):
    """0.5 * Dice + 0.5 * BCE — robust for imbalanced segmentation."""
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.bce  = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        return 0.5 * self.dice(logits, targets) + 0.5 * self.bce(logits, targets)


# ═══════════════════════════ METRICS ══════════════════════════

def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray, smooth: float = 1e-5) -> float:
    """Dice coefficient. pred_mask and gt_mask are binary (0/1)."""
    intersection = (pred_mask * gt_mask).sum()
    return (2 * intersection + smooth) / (pred_mask.sum() + gt_mask.sum() + smooth)


def jaccard_score(pred_mask: np.ndarray, gt_mask: np.ndarray, smooth: float = 1e-5) -> float:
    """Jaccard / IoU."""
    intersection = (pred_mask * gt_mask).sum()
    union        = pred_mask.sum() + gt_mask.sum() - intersection
    return (intersection + smooth) / (union + smooth)


# ═══════════════════════════ DATASET ══════════════════════════

class MRISegDataset(Dataset):
    """
    Loads (slice, mask) pairs from a preprocessed directory.
    Expects either:
      - NIfTI pairs: <id>_flair.nii.gz (image) + <id>_seg.nii.gz (mask) — BRATS format
      - NPY pairs:   <id>.npy (image volume, already preprocessed)
        Generates synthetic circular masks when no GT is available (for demo of pipeline shape).
    """
    def __init__(self, data_dir: str, size: int = 256, use_synthetic_masks: bool = False):
        self.size = size
        self.samples = []   # list of (image_np, mask_np) in [0,1]
        self._load(data_dir, use_synthetic_masks)

    def _load(self, data_dir: str, synth_masks: bool):
        files = sorted(os.listdir(data_dir))
        npy_files = [f for f in files if f.endswith(".npy") and not f.startswith("mask_")]

        for fname in npy_files:
            vol = np.load(os.path.join(data_dir, fname)).astype(np.float32)
            mask_path = os.path.join(data_dir, "mask_" + fname)

            for z in range(0, vol.shape[2] if vol.ndim == 3 else 1, 3):
                sl = vol[:, :, z] if vol.ndim == 3 else vol
                sl = (sl - sl.min()) / (sl.max() - sl.min() + 1e-6)

                if os.path.exists(mask_path):
                    mask_vol = np.load(mask_path).astype(np.float32)
                    mask = mask_vol[:, :, z] if mask_vol.ndim == 3 else mask_vol
                    mask = (mask > 0.5).astype(np.float32)
                elif synth_masks:
                    mask = self._synthetic_mask(sl.shape, z)
                else:
                    mask = self._synthetic_mask(sl.shape, z)  # fallback

                self.samples.append((sl, mask))

        # BRATS-style NIfTI pairs
        if HAS_NIBABEL:
            flair_files = [f for f in files if "flair" in f.lower() and
                           (f.endswith(".nii") or f.endswith(".nii.gz"))]
            for ff in flair_files:
                seg_fname = ff.replace("flair", "seg").replace("FLAIR", "seg")
                flair_path = os.path.join(data_dir, ff)
                seg_path   = os.path.join(data_dir, seg_fname)
                img_vol = nib.load(flair_path).get_fdata().astype(np.float32)
                if img_vol.ndim == 4:
                    img_vol = img_vol[..., 0]
                has_gt = os.path.exists(seg_path)
                seg_vol = nib.load(seg_path).get_fdata().astype(np.float32) if has_gt else None

                for z in range(0, img_vol.shape[2], 4):
                    sl = img_vol[:, :, z]
                    sl = (sl - sl.min()) / (sl.max() - sl.min() + 1e-6)
                    mask = (seg_vol[:, :, z] > 0).astype(np.float32) if has_gt else \
                            self._synthetic_mask(sl.shape, z)
                    self.samples.append((sl, mask))

    def _synthetic_mask(self, shape, seed=0) -> np.ndarray:
        """Small circular mask for demo / no-GT fallback."""
        rng = np.random.default_rng(seed)
        mask = np.zeros(shape, dtype=np.float32)
        cy = shape[0] // 2 + rng.integers(-30, 30)
        cx = shape[1] // 2 + rng.integers(-30, 30)
        r  = rng.integers(10, 30)
        Y, X = np.ogrid[:shape[0], :shape[1]]
        mask[(X - cx)**2 + (Y - cy)**2 <= r**2] = 1.0
        return mask

    def __len__(self):
        return len(self.samples)

    def _resize(self, arr: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(arr, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(resized).unsqueeze(0).float()

    def __getitem__(self, idx):
        img, mask = self.samples[idx]
        if np.random.rand() > 0.5:
            img  = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
        return self._resize(img), self._resize(mask)


class DemoSegDataset(Dataset):
    """Fully synthetic (slice, mask) pairs for pipeline testing."""
    def __init__(self, n: int = 100, size: int = 256, seed: int = 42):
        self.n = n; self.size = size; self.seed = seed

    def __len__(self): return self.n

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed + idx)
        size = self.size
        # Image: oval brain + lesion blob
        img = np.zeros((size, size), dtype=np.float32)
        cy, cx = size // 2, size // 2
        Y, X = np.ogrid[:size, :size]
        brain = ((X - cx)**2 / (0.35 * size)**2 + (Y - cy)**2 / (0.43 * size)**2) <= 1
        img[brain] = rng.uniform(0.3, 0.7, img[brain].shape)
        # Lesion
        lx = cx + rng.integers(-50, 50)
        ly = cy + rng.integers(-50, 50)
        r  = rng.integers(12, 28)
        lesion_mask = ((X - lx)**2 + (Y - ly)**2) <= r**2
        img[lesion_mask] = rng.uniform(0.75, 1.0, img[lesion_mask].shape)
        img += rng.normal(0, 0.04, img.shape).astype(np.float32)
        img = np.clip(img, 0, 1)

        # Ground truth mask = lesion area
        gt = lesion_mask.astype(np.float32)

        t_img  = torch.from_numpy(img).unsqueeze(0).float()
        t_mask = torch.from_numpy(gt).unsqueeze(0).float()
        return t_img, t_mask


# ════════════════════════ TRAIN / EVAL ════════════════════════

def train_epoch(model, loader, optimizer, criterion, epoch, total):
    model.train()
    total_loss = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    avg = total_loss / len(loader)
    print(f"  Epoch [{epoch}/{total}] Train Loss: {avg:.5f}")
    return avg


@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    dice_list, jacc_list, total_loss = [], [], 0.0
    criterion = CombinedLoss()
    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        logits = model(imgs)
        total_loss += criterion(logits, masks).item()
        preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
        gts   = masks.cpu().numpy()
        for p, g in zip(preds, gts):
            dice_list.append(dice_score(p[0], g[0]))
            jacc_list.append(jaccard_score(p[0], g[0]))
    return np.mean(dice_list), np.mean(jacc_list), total_loss / len(loader)


# ════════════════════════ VISUALISATION ═══════════════════════

def save_segmentation_grid(model, dataset, out_dir: str, n: int = 4):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    indices = np.linspace(0, len(dataset) - 1, n, dtype=int)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    fig.suptitle("ROI Segmentation — MedhaDrishti\n(Left: Input | Centre: Predicted Mask | Right: Ground Truth)",
                 fontsize=12, fontweight="bold")
    with torch.no_grad():
        for row, idx in enumerate(indices):
            img, gt = dataset[idx]
            logit = model(img.unsqueeze(0).to(DEVICE))
            pred  = (torch.sigmoid(logit) > 0.5).float().cpu().squeeze().numpy()
            img_np = img.squeeze().numpy()
            gt_np  = gt.squeeze().numpy()
            d = dice_score(pred, gt_np)
            j = jaccard_score(pred, gt_np)
            for col, (vis, title) in enumerate([
                (img_np,         "Input Slice"),
                (pred,           f"Prediction\nDice={d:.3f} Jaccard={j:.3f}"),
                (gt_np,          "Ground Truth"),
            ]):
                ax = axes[row, col]
                ax.imshow(img_np, cmap="gray", vmin=0, vmax=1)
                if col in (1, 2):
                    ax.imshow(vis, cmap="Reds", alpha=0.5, vmin=0, vmax=1)
                ax.set_title(title, fontsize=9)
                ax.axis("off")
    plt.tight_layout()
    out = os.path.join(out_dir, "segmentation_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Segmentation grid -> {out}")


def save_loss_curve(train_losses, val_losses, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="Train Loss", color="#4C9BE8")
    ax.plot(val_losses,   label="Val Loss",   color="#E86B4C", linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Combined Loss")
    ax.set_title("Segmentation Model — Training Curve")
    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = os.path.join(out_dir, "segmentation_loss_curve.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Loss curve -> {out}")


# ════════════════════════════ MAIN ════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MedhaDrishti — Segmentation Model")
    parser.add_argument("--data",       type=str,   default=None)
    parser.add_argument("--out",        type=str,   default="models/unet.pt")
    parser.add_argument("--results",    type=str,   default="results/results.json")
    parser.add_argument("--plot-dir",   type=str,   default="results/plots/segmentation")
    parser.add_argument("--epochs",     type=int,   default=15)
    parser.add_argument("--batch-size", type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--img-size",   type=int,   default=256)
    parser.add_argument("--val-split",  type=float, default=0.15)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--demo",       action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else "models", exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────
    if args.demo:
        print("[segmentation] DEMO mode — synthetic dataset.")
        full_ds = DemoSegDataset(n=100, size=args.img_size, seed=args.seed)
    else:
        if args.data is None:
            raise ValueError("--data is required")
        full_ds = MRISegDataset(args.data, size=args.img_size, use_synthetic_masks=True)
        if len(full_ds) == 0:
            raise RuntimeError(f"No slices loaded from {args.data}.")

    n_val = max(1, int(len(full_ds) * args.val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"  Train: {n_train} slices | Val: {n_val} slices")

    # ── Model ─────────────────────────────────────────────────
    model     = UNet(in_ch=1, base_ch=64).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.5)
    criterion = CombinedLoss()

    # ── Baseline ─────────────────────────────────────────────
    base_dice, base_jacc, _ = eval_epoch(model, val_loader)
    print(f"\n  Baseline -> Dice: {base_dice:.4f} | Jaccard: {base_jacc:.4f}")

    # ── Training ─────────────────────────────────────────────
    train_losses, val_losses = [], []
    best_dice = 0.0
    t0 = time.time()
    print(f"\n[segmentation] Training for {args.epochs} epochs ...")
    for epoch in range(1, args.epochs + 1):
        t_loss = train_epoch(model, train_loader, optimizer, criterion, epoch, args.epochs)
        v_dice, v_jacc, v_loss = eval_epoch(model, val_loader)
        scheduler.step()
        train_losses.append(t_loss); val_losses.append(v_loss)
        print(f"               Val   -> Dice: {v_dice:.4f} | Jaccard: {v_jacc:.4f}")
        if v_dice > best_dice:
            best_dice = v_dice
            _, best_jacc, _ = eval_epoch(model, val_loader)
            torch.save(model.state_dict(), args.out)
            print(f"               [OK] New best Dice — model saved -> {args.out}")

    elapsed = time.time() - t0
    print(f"\n[segmentation] Training complete in {elapsed/60:.1f} min.")
    print(f"  Best  -> Dice: {best_dice:.4f} | Jaccard: {best_jacc:.4f}")
    print(f"  Baseline -> Dice: {base_dice:.4f} | Jaccard: {base_jacc:.4f}")

    # -- Visualise --------------------------------------------
    model.load_state_dict(torch.load(args.out, map_location=DEVICE, weights_only=True))
    save_segmentation_grid(model, full_ds, args.plot_dir, n=4)
    save_loss_curve(train_losses, val_losses, args.plot_dir)

    # -- results.json -----------------------------------------
    result = {
        "module": "segmentation",
        "model": "UNet-4level",
        "device": str(DEVICE),
        "roi": "tumor (brain)",
        "modality": "T2/FLAIR",
        "epochs_trained": args.epochs,
        "train_samples": n_train,
        "val_samples": n_val,
        "baseline_dice":    round(float(base_dice), 4),
        "baseline_jaccard": round(float(base_jacc), 4),
        "best_dice":        round(float(best_dice), 4),
        "best_jaccard":     round(float(best_jacc), 4),
        "training_time_min": round(float(elapsed / 60), 2),
    }

    all_results = {}
    if os.path.exists(args.results):
        try:
            with open(args.results) as f:
                all_results = json.load(f)
        except Exception:
            all_results = {}
    all_results["segmentation"] = result
    with open(args.results, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[segmentation] Results -> {args.results}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
