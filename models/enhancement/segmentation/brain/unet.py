"""
Brain Segmentation U-Net with BlurPool and Attention Gates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BlurPool2d(nn.Module):
    """
    Anti-alias pooling (Zhang 2019). Applies a fixed Gaussian blur before
    stride-2 downsampling. Drop-in replacement for MaxPool2d.
    Improves shift-invariance for lesion segmentation.
    """
    def __init__(self, channels: int, stride: int = 2):
        super().__init__()
        self.stride = stride
        k = torch.tensor([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]], dtype=torch.float32)
        k = k / k.sum()
        self.register_buffer("kernel", k.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1))
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sharp features first (MaxPool, stride 1), then blur + downsample
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=1, padding=1)
        return nn.functional.conv2d(x, self.kernel, stride=self.stride,
                                    padding=1, groups=self.channels)


class AttentionGate(nn.Module):
    """
    Attention Gate (Oktay et al., 'Attention U-Net', 2018).
    Uses the upsampled decoder features (g) to weight the encoder skip features (x).
    """
    def __init__(self, f_g: int, f_l: int, f_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, bias=True),
            nn.GroupNorm(num_groups=min(32, f_int), num_channels=f_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, bias=True),
            nn.GroupNorm(num_groups=min(32, f_int), num_channels=f_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, bias=True),
            nn.GroupNorm(num_groups=1, num_channels=1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[-2:] != x1.shape[-2:]:
            g1 = F.interpolate(g1, size=x1.shape[-2:],
                               mode="bilinear", align_corners=False)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch), nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample -> Attention-gate the skip connection -> concat -> ConvBlock"""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0,
                 use_attention: bool = True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.attn = AttentionGate(f_g=in_ch // 2, f_l=skip_ch,
                                  f_int=max(8, skip_ch // 2)) if use_attention else None
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        if self.attn is not None:
            skip = self.attn(g=x, x=skip)
        return self.conv(torch.cat([skip, x], dim=1))


class SegmentationUNet(nn.Module):
    """
    4-level U-Net for brain tumor segmentation.
    Upgrades:
      - BlurPool replaces MaxPool2d (anti-aliasing)
      - Attention Gates on every skip connection
      - Deep Supervision: auxiliary outputs at each decoder level
    Input: (B, 1, H, W) -> Output: (B, 1, H, W) logits
    """
    def __init__(self, in_ch: int = 1, base_ch: int = 64, dropout: float = 0.1,
                 use_blurpool: bool = True, use_attention: bool = True,
                 deep_supervision: bool = False):
        super().__init__()
        self.deep_supervision = deep_supervision
        b = base_ch
        
        self.enc1 = ConvBlock(in_ch, b)
        self.enc2 = ConvBlock(b, b * 2)
        self.enc3 = ConvBlock(b * 2, b * 4)
        self.enc4 = ConvBlock(b * 4, b * 8)

        if use_blurpool:
            self.pool1 = BlurPool2d(b)
            self.pool2 = BlurPool2d(b * 2)
            self.pool3 = BlurPool2d(b * 4)
            self.pool4 = BlurPool2d(b * 8)
        else:
            self.pool1 = self.pool2 = self.pool3 = self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(b * 8, b * 16, dropout=dropout)
        self.up4 = UpBlock(b * 16, b * 8, b * 8, dropout=dropout, use_attention=use_attention)
        self.up3 = UpBlock(b * 8, b * 4, b * 4, dropout=dropout, use_attention=use_attention)
        self.up2 = UpBlock(b * 4, b * 2, b * 2, use_attention=use_attention)
        self.up1 = UpBlock(b * 2, b, b, use_attention=use_attention)
        self.head = nn.Conv2d(b, 1, kernel_size=1)
        
        # Deep supervision auxiliary heads
        if deep_supervision:
            self.ds_head4 = nn.Conv2d(b * 8, 1, kernel_size=1)
            self.ds_head3 = nn.Conv2d(b * 4, 1, kernel_size=1)
            self.ds_head2 = nn.Conv2d(b * 2, 1, kernel_size=1)
            self.ds_head1 = nn.Conv2d(b, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool1(s1))
        s3 = self.enc3(self.pool2(s2))
        s4 = self.enc4(self.pool3(s3))
        b = self.bottleneck(self.pool4(s4))
        d4 = self.up4(b, s4)
        d3 = self.up3(d4, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)
        main_out = self.head(d1)
        
        if self.deep_supervision and self.training:
            ds4 = self.ds_head4(d4)
            ds3 = self.ds_head3(d3)
            ds2 = self.ds_head2(d2)
            ds1 = self.ds_head1(d1)
            return main_out, ds4, ds3, ds2, ds1
        return main_out


# Alias for compatibility
UNet = SegmentationUNet