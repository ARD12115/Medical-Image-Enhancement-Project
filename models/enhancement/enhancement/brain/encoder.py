"""
ResNet18 Encoder for Brain Enhancement - ImageNet pretrained backbone
"""

import torch
import torch.nn as nn
import torchvision.models as models


def get_resnet18_encoder(pretrained: bool = True):
    """
    Get ResNet18 encoder with adapted first layer for 1-channel input.
    Returns the encoder components for use in U-Net style decoder.
    """
    backbone = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    )
    
    # Adapt first conv to accept 1-channel input
    w = backbone.conv1.weight.data  # (64, 3, 7, 7)
    backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    backbone.conv1.weight.data = w.mean(dim=1, keepdim=True)
    
    # Return encoder components
    encoder = nn.ModuleDict({
        'enc0': nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu),  # /2 -> 64
        'pool': backbone.maxpool,  # /4
        'enc1': backbone.layer1,   # /4 -> 64
        'enc2': backbone.layer2,   # /8 -> 128
        'enc3': backbone.layer3,   # /16 -> 256
        'enc4': backbone.layer4,   # /32 -> 512
    })
    
    return encoder


class ResNet18Encoder(nn.Module):
    """ResNet18 Encoder module for easy integration"""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.encoder = get_resnet18_encoder(pretrained)
    
    def forward(self, x):
        """Return all encoder features for skip connections"""
        e0 = self.encoder['enc0'](x)       # H/2, 64
        p = self.encoder['pool'](e0)       # H/4
        e1 = self.encoder['enc1'](p)       # H/4, 64
        e2 = self.encoder['enc2'](e1)      # H/8, 128
        e3 = self.encoder['enc3'](e2)      # H/16, 256
        e4 = self.encoder['enc4'](e3)      # H/32, 512
        return e0, e1, e2, e3, e4
    
    def get_feature_channels(self):
        """Return channel counts for each encoder level"""
        return [64, 64, 128, 256, 512]  # e0, e1, e2, e3, e4