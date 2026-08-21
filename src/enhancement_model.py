"""
enhancement_model.py — MedhaDrishti
ResNet18 encoder (ImageNet-pretrained) + lightweight decoder for MRI enhancement.
Evaluates with PSNR & SSIM. Writes results to results/results.json.

Usage:
    python src/enhancement_model.py --data data/brain_preprocessed --epochs 20 --out models/enhancer.pt
    python src/enhancement_model.py --demo --epochs 5
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
import torchvision.models as models
import torchvision.transforms.functional as TF
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[enhancement] Using device: {DEVICE}")

# ═══════════════════════════ MODEL ════════════════════════════

class DecoderBlock(nn.Module):
    """Upsample → Conv → BN → ReLU."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResNet18Enhancer(nn.Module):
    """
    ResNet18 encoder (pretrained ImageNet) + 4-block decoder.
    Input : (B, 1, H, W)  — single-channel MRI slice
    Output: (B, 1, H, W)  — enhanced slice
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # Adapt first conv to accept 1-channel input (average the 3-channel weights)
        w = backbone.conv1.weight.data  # (64, 3, 7, 7)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone.conv1.weight.data = w.mean(dim=1, keepdim=True)

        # Encoder stages
        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # /2  → 64ch
        self.pool = backbone.maxpool                                              # /4
        self.enc1 = backbone.layer1   # /4  → 64ch
        self.enc2 = backbone.layer2   # /8  → 128ch
        self.enc3 = backbone.layer3   # /16 → 256ch
        self.enc4 = backbone.layer4   # /32 → 512ch

        # Decoder: 32× → 16× → 8× → 4× → 2× → 1×
        self.dec4 = DecoderBlock(512, 256)
        self.dec3 = DecoderBlock(256, 128)
        self.dec2 = DecoderBlock(128, 64)
        self.dec1 = DecoderBlock(64, 32)
        self.dec0 = DecoderBlock(32, 16)

        # Final head → single channel in [0,1]
        self.head = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encoder
        e0 = self.enc0(x)       # H/2
        p  = self.pool(e0)      # H/4
        e1 = self.enc1(p)       # H/4
        e2 = self.enc2(e1)      # H/8
        e3 = self.enc3(e2)      # H/16
        e4 = self.enc4(e3)      # H/32

        # Decoder (no skip connections to keep it lightweight & fast)
        d = self.dec4(e4)       # H/16
        d = self.dec3(d)        # H/8
        d = self.dec2(d)        # H/4
        d = self.dec1(d)        # H/2
        d = self.dec0(d)        # H

        # Resize to exactly match input if off by a pixel
        if d.shape[-2:] != x.shape[-2:]:
            d = nn.functional.interpolate(d, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return self.head(d)


# ═══════════════════════════ DATASET ══════════════════════════

class MRISliceDataset(Dataset):
    """
    Loads grayscale MRI slices from a directory of .npy or NIfTI files.
    Input  = noisy raw slice (slightly degraded)
    Target = CLAHE-preprocessed clean slice
    """
    def __init__(self, data_dir: str, size: int = 256, augment: bool = True):
        self.size = size
        self.augment = augment
        self.samples = []   # list of (noisy_np, clean_np)
        self._load(data_dir)

    def _load(self, data_dir: str):
        files = sorted(os.listdir(data_dir))
        for fname in files:
            path = os.path.join(data_dir, fname)
            if fname.endswith(".npy"):
                vol = np.load(path)          # already preprocessed
                self._extract_slices(vol, "clean")
            elif HAS_NIBABEL and (fname.endswith(".nii") or fname.endswith(".nii.gz")):
                img = nib.load(path)
                vol = img.get_fdata().astype(np.float32)
                if vol.ndim == 4:
                    vol = vol[..., 0]
                self._extract_slices(vol, "nifti")

    def _extract_slices(self, vol: np.ndarray, source: str):
        """Extract every 3rd axial slice to avoid redundancy."""
        for z in range(0, vol.shape[2], 3):
            sl_clean = vol[:, :, z].astype(np.float32)
            if sl_clean.max() > sl_clean.min():
                sl_clean = (sl_clean - sl_clean.min()) / (sl_clean.max() - sl_clean.min())
            else:
                continue
            # Simulate noisy input by adding Gaussian noise
            rng = np.random.default_rng(z)
            noise = rng.normal(0, 0.05, sl_clean.shape).astype(np.float32)
            sl_noisy = np.clip(sl_clean + noise, 0, 1)
            self.samples.append((sl_noisy, sl_clean))

    def __len__(self):
        return len(self.samples)

    def _resize(self, arr: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(arr, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(resized).unsqueeze(0).float()   # (1, H, W)

    def __getitem__(self, idx):
        noisy, clean = self.samples[idx]
        if self.augment and np.random.rand() > 0.5:
            noisy = np.fliplr(noisy).copy()
            clean = np.fliplr(clean).copy()
        return self._resize(noisy), self._resize(clean)


# ═══════════════════════ DEMO DATASET ═════════════════════════

class DemoDataset(Dataset):
    """Synthetic MRI-like dataset for pipeline testing without real data."""
    def __init__(self, n_samples: int = 120, size: int = 256, seed: int = 42):
        self.n = n_samples
        self.size = size
        self.rng = np.random.default_rng(seed)

    def _make_slice(self, idx: int):
        rng = np.random.default_rng(idx)
        base = np.zeros((self.size, self.size), dtype=np.float32)
        cy, cx = self.size // 2, self.size // 2
        Y, X = np.ogrid[:self.size, :self.size]
        # Brain oval
        mask = ((X - cx)**2 / (0.35 * self.size)**2 +
                (Y - cy)**2 / (0.43 * self.size)**2) <= 1
        base[mask] = rng.uniform(0.4, 0.8, base[mask].shape)
        # Lesion blob
        lx = cx + rng.integers(-40, 40)
        ly = cy + rng.integers(-40, 40)
        lmask = ((X - lx)**2 + (Y - ly)**2) <= (12**2)
        base[lmask] = rng.uniform(0.8, 1.0, base[lmask].shape)
        return np.clip(base, 0, 1)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        clean = self._make_slice(idx)
        noise = np.random.default_rng(idx + 10000).normal(0, 0.06, clean.shape).astype(np.float32)
        noisy = np.clip(clean + noise, 0, 1)
        t_noisy = torch.from_numpy(noisy).unsqueeze(0).float()
        t_clean = torch.from_numpy(clean).unsqueeze(0).float()
        return t_noisy, t_clean


# ═══════════════════════════ METRICS ══════════════════════════

def compute_psnr_ssim(pred: np.ndarray, target: np.ndarray):
    """pred, target in [0,1]."""
    pred_u8   = (pred   * 255).clip(0, 255).astype(np.uint8)
    target_u8 = (target * 255).clip(0, 255).astype(np.uint8)
    psnr = calc_psnr(target_u8, pred_u8, data_range=255)
    ssim = calc_ssim(target_u8, pred_u8, data_range=255)
    return float(psnr), float(ssim)


# ═══════════════════════════ TRAINING ═════════════════════════

def train(model, loader, optimizer, criterion, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    for noisy, clean in loader:
        noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
        optimizer.zero_grad()
        pred = model(noisy)
        loss = criterion(pred, clean)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg = total_loss / len(loader)
    print(f"  Epoch [{epoch}/{total_epochs}] Train Loss: {avg:.5f}")
    return avg


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    psnr_list, ssim_list, val_loss = [], [], 0.0
    criterion = nn.MSELoss()
    for noisy, clean in loader:
        noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
        pred = model(noisy)
        val_loss += criterion(pred, clean).item()
        for p, t in zip(pred.cpu().numpy(), clean.cpu().numpy()):
            ps, ss = compute_psnr_ssim(p[0], t[0])
            psnr_list.append(ps)
            ssim_list.append(ss)
    return (np.mean(psnr_list), np.mean(ssim_list), val_loss / len(loader))


# ════════════════════════ VISUALISATION ═══════════════════════

def save_comparison_grid(model, dataset, out_dir: str, n: int = 4):
    """Save a grid of noisy → enhanced → target for the report."""
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    indices = np.linspace(0, len(dataset) - 1, n, dtype=int)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    fig.suptitle("MRI Enhancement — MedhaDrishti\n(Left: Noisy Input | Centre: Enhanced | Right: Target)",
                 fontsize=12, fontweight="bold")
    with torch.no_grad():
        for row, idx in enumerate(indices):
            noisy, clean = dataset[idx]
            pred = model(noisy.unsqueeze(0).to(DEVICE)).cpu().squeeze().numpy()
            noisy_np = noisy.squeeze().numpy()
            clean_np = clean.squeeze().numpy()
            ps, ss = compute_psnr_ssim(pred, clean_np)
            for col, (img, title) in enumerate([
                (noisy_np, "Noisy Input"),
                (pred,     f"Enhanced\nPSNR={ps:.1f} SSIM={ss:.3f}"),
                (clean_np, "Target (clean)"),
            ]):
                ax = axes[row, col]
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                ax.set_title(title, fontsize=9)
                ax.axis("off")
    plt.tight_layout()
    out = os.path.join(out_dir, "enhancement_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Comparison grid -> {out}")


def save_loss_curve(train_losses: list, val_losses: list, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="Train Loss", color="#4C9BE8")
    ax.plot(val_losses,   label="Val Loss",   color="#E86B4C", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Enhancement Model — Training Curve")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    out = os.path.join(out_dir, "enhancement_loss_curve.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Loss curve -> {out}")


# ════════════════════════════ MAIN ════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MedhaDrishti — Enhancement Model")
    parser.add_argument("--data",       type=str,   default=None)
    parser.add_argument("--out",        type=str,   default="models/enhancer.pt")
    parser.add_argument("--results",    type=str,   default="results/results.json")
    parser.add_argument("--plot-dir",   type=str,   default="results/plots/enhancement")
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch-size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-3)
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
        print("[enhancement] DEMO mode — synthetic dataset.")
        full_ds = DemoDataset(n_samples=120, size=args.img_size, seed=args.seed)
    else:
        if args.data is None:
            raise ValueError("--data is required")
        full_ds = MRISliceDataset(args.data, size=args.img_size)
        if len(full_ds) == 0:
            raise RuntimeError(f"No slices loaded from {args.data}. Check the directory.")

    n_val = max(1, int(len(full_ds) * args.val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"  Train: {n_train} slices | Val: {n_val} slices")

    # ── Model ─────────────────────────────────────────────────
    model     = ResNet18Enhancer(pretrained=True).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.5)
    criterion = nn.MSELoss()

    # ── Baseline PSNR/SSIM (before training) ──────────────────
    base_psnr, base_ssim, _ = evaluate(model, val_loader)
    print(f"\n  Baseline (pre-training) -> PSNR: {base_psnr:.2f} dB | SSIM: {base_ssim:.4f}")

    # ── Training loop ─────────────────────────────────────────
    train_losses, val_losses = [], []
    best_psnr, best_ssim = 0.0, 0.0
    t0 = time.time()

    print(f"\n[enhancement] Training for {args.epochs} epochs ...")
    for epoch in range(1, args.epochs + 1):
        t_loss = train(model, train_loader, optimizer, criterion, epoch, args.epochs)
        v_psnr, v_ssim, v_loss = evaluate(model, val_loader)
        scheduler.step()
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        print(f"             Val   -> PSNR: {v_psnr:.2f} dB | SSIM: {v_ssim:.4f}")
        if v_psnr > best_psnr:
            best_psnr, best_ssim = v_psnr, v_ssim
            torch.save(model.state_dict(), args.out)
            print(f"             [OK] New best PSNR — model saved -> {args.out}")

    elapsed = time.time() - t0
    print(f"\n[enhancement] Training complete in {elapsed/60:.1f} min.")
    print(f"  Best -> PSNR: {best_psnr:.2f} dB | SSIM: {best_ssim:.4f}")
    print(f"  Baseline -> PSNR: {base_psnr:.2f} dB | SSIM: {base_ssim:.4f}")
    print(f"  Improvement -> dPSNR: {best_psnr - base_psnr:+.2f} dB | dSSIM: {best_ssim - base_ssim:+.4f}")

    # ── Load best weights & visualise ─────────────────────────
    model.load_state_dict(torch.load(args.out, map_location=DEVICE, weights_only=True))
    save_comparison_grid(model, full_ds, args.plot_dir, n=4)
    save_loss_curve(train_losses, val_losses, args.plot_dir)

    # ── Write results.json ────────────────────────────────────
    result = {
        "module": "enhancement",
        "model": "ResNet18Enhancer",
        "device": str(DEVICE),
        "epochs_trained": args.epochs,
        "train_samples": n_train,
        "val_samples": n_val,
        "baseline_psnr_db": round(float(base_psnr), 4),
        "baseline_ssim":    round(float(base_ssim), 4),
        "best_psnr_db":     round(float(best_psnr), 4),
        "best_ssim":        round(float(best_ssim), 4),
        "delta_psnr_db":    round(float(best_psnr - base_psnr), 4),
        "delta_ssim":       round(float(best_ssim - base_ssim), 4),
        "training_time_min": round(float(elapsed / 60), 2),
    }

    os.makedirs(os.path.dirname(args.results) if os.path.dirname(args.results) else "results", exist_ok=True)
    # Merge with existing results.json if present
    all_results = {}
    if os.path.exists(args.results):
        try:
            with open(args.results) as f:
                all_results = json.load(f)
        except Exception:
            all_results = {}
    all_results["enhancement"] = result
    with open(args.results, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[enhancement] Results -> {args.results}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
