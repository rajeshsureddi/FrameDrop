"""Frozen encoders.

DenseEncoder   : runs on EVERY frame at 256x144. MobileNetV3-small up to the
                 stride-16 stage, pooled to a 4x6 grid (keeps coarse layout so
                 pans and jumps move the embedding). 48*4*6 = 1152 dims.
SpatialEncoder : runs only on sampled frames, on a native-resolution fragment.
                 ResNet-50 layer4, mean+std pooled (VSFA-style, 4096 dims);
                 VSFA with these features reached 0.85 SROCC on the study data.

Both are pluggable: swap in a quality-aware backbone (FAST-VQA/DOVER) later.
"""
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_tensor(batch_uint8, device):
    x = torch.from_numpy(np.ascontiguousarray(np.stack(batch_uint8)))
    x = x.to(device).permute(0, 3, 1, 2).float() / 255.0
    return (x - _MEAN.to(device)) / _STD.to(device)


class DenseEncoder(nn.Module):
    dim = 48 * 4 * 6

    def __init__(self):
        super().__init__()
        m = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.DEFAULT)
        self.body = m.features[:9]
        self.pool = nn.AdaptiveAvgPool2d((4, 6))
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):
        return self.pool(self.body(x)).flatten(1)


class SpatialEncoder(nn.Module):
    dim = 4096

    def __init__(self):
        super().__init__()
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
        self.body = nn.Sequential(*list(m.children())[:-2])
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):
        f = self.body(x)
        return torch.cat([f.mean((2, 3)), f.std((2, 3))], 1)
