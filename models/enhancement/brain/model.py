"""
Enhancement Model for Brain MRI - ResNet18-UNet Architecture
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ConvBlock(nn.Module):
    """Conv -> GN -> ReLU (GroupNorm for small batch stability)"""
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding, bias=False),
            nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel, padding=padding, bias=False),
            nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    """Upsample -> concat skip -> ConvBlock"""
    def __init__(self, in_ch, out_ch, skip_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        conv_in_ch = out_ch + skip_ch
        self.conv = ConvBlock(in_ch=conv_in_ch, out_ch=out_ch)
    
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class EnhancementModel(nn.Module):
    """
    ResNet18 encoder (pretrained ImageNet) + U-Net decoder with lateral skips
    Input: (B, 1, H, W) -> Output: (B, 1, H, W) enhanced image
    """
    def __init__(self, pretrained: bool = True, base_decoder_ch: int = 128):
        super().__init__()
        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        
        # Adapt first conv to accept 1-channel input
        w = backbone.conv1.weight.data  # (64, 3, 7, 7)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone.conv1.weight.data = w.mean(dim=1, keepdim=True)
        
        # Encoder
        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # /2 -> 64
        self.pool = backbone.maxpool  # /4
        self.enc1 = backbone.layer1    # /4 -> 64
        self.enc2 = backbone.layer2    # /8 -> 128
        self.enc3 = backbone.layer3    # /16 -> 256
        self.enc4 = backbone.layer4    # /32 -> 512
        
        # Bridge conv to reduce channels before decoder
        self.bridge = ConvBlock(512, base_decoder_ch)
        
        # Decoder
        self.up4 = UpBlock(base_decoder_ch, 256, skip_ch=256)  # bridge -> enc3
        self.up3 = UpBlock(256, 128, skip_ch=128)              # dec4 -> enc2
        self.up2 = UpBlock(128, 64,  skip_ch=64)               # dec3 -> enc1
        self.up1 = UpBlock(64, 32,   skip_ch=64)               # dec2 -> enc0
        
        self.head = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        e0 = self.enc0(x)       # H/2
        p  = self.pool(e0)      # H/4
        e1 = self.enc1(p)       # H/4
        e2 = self.enc2(e1)      # H/8
        e3 = self.enc3(e2)      # H/16
        e4 = self.enc4(e3)      # H/32
        
        b = self.bridge(e4)
        
        d4 = self.up4(b, e3)    # H/16
        d3 = self.up3(d4, e2)   # H/8
        d2 = self.up2(d3, e1)   # H/4
        d1 = self.up1(d2, e0)   # H/2
        
        out = nn.functional.interpolate(d1, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(out)


# Alias for compatibility
ResNet18Enhancer = EnhancementModel


def build_enhancer(arch: str = "resnet18", pretrained: bool = True) -> nn.Module:
    """Build enhancement model by architecture name"""
    if arch.lower() == "resnet18":
        return EnhancementModel(pretrained=pretrained)
    elif arch.lower() == "convnext":
        from .convnext_unet import ConvNeXtUNet
        return ConvNeXtUNet(pretrained=pretrained)
    raise ValueError(f"Unknown enhancement architecture: {arch}")