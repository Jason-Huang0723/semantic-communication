#!/usr/bin/env python3
"""
A clean, self-contained PyTorch implementation of **ResNet-50** from scratch
(without using torchvision.models), supporting both ImageNet-style and
CIFAR-style stems.

- For CIFAR-100 (32×32): use `resnet50_cifar(num_classes=100)`.
- For ImageNet: use `resnet50_imagenet(num_classes=1000)`.

Features
========
1) Bottleneck block with expansion=4 (ResNet-50).
2) Configurable stem:
   - CIFAR: 3×3 conv, stride=1, no maxpool.
   - ImageNet: 7×7 conv, stride=2, followed by 3×3 maxpool stride=2.
3) He (Kaiming) initialization + optional zero-init of residual branch's last BN.
4) Minimal and readable; easy to plug into your training loop.

Usage example
-------------
>>> from resnet50_custom import resnet50_cifar
>>> model = resnet50_cifar(num_classes=100)
>>> x = torch.randn(4, 3, 32, 32)
>>> logits = model(x)   # [4, 100]

"""

from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "Bottleneck",
    "ResNet",
    "resnet50_cifar",
    "resnet50_imagenet",
]


# -----------------------------------------------------------------------------
# Building Blocks
# -----------------------------------------------------------------------------
class ConvBNAct(nn.Module):
    """Conv2d → BatchNorm2d → ReLU (inplace).

    Args:
        in_ch, out_ch: channels
        kernel_size: int
        stride: int
        padding: int
        bias: bool (default False)
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Bottleneck(nn.Module):
    """ResNet Bottleneck block (expansion=4).

    Layout (ResNet v1.5 style with stride applied on the 3×3 conv):
        1×1, reduce → 3×3, stride (maybe) → 1×1, expand

    If shape changes (stride>1 or channels differ), a projection via 1×1 conv is used.
    """

    expansion: int = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        # 1×1 reduce
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # 3×3 conv (spatial)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # 1×1 expand
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out); out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


# -----------------------------------------------------------------------------
# ResNet backbone
# -----------------------------------------------------------------------------
class ResNet(nn.Module):
    """Generic ResNet backbone.

    Args:
        block: block class, e.g., Bottleneck
        layers: list with number of blocks in each stage (e.g., [3,4,6,3] for ResNet-50)
        num_classes: classifier output dim
        in_ch: input channels (default 3)
        stem_type: "cifar" or "imagenet"
        zero_init_residual: if True, zero-initialize the last BN in each block
    """

    def __init__(
        self,
        block: Callable[..., nn.Module],
        layers: List[int],
        num_classes: int = 1000,
        in_ch: int = 3,
        stem_type: str = "imagenet",
        zero_init_residual: bool = True,
    ):
        super().__init__()
        assert stem_type in {"cifar", "imagenet"}

        self.inplanes = 64
        if stem_type == "imagenet":
            # 7×7 conv, stride 2; then 3×3 maxpool stride 2
            self.stem = nn.Sequential(
                nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        else:
            # CIFAR-style: 3×3 conv, stride=1, no maxpool
            self.stem = nn.Sequential(
                nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )

        # Stages
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights(zero_init_residual=zero_init_residual)

    # ---- helpers ----
    def _make_layer(self, block: Callable[..., nn.Module], planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, downsample=None))
        return nn.Sequential(*layers)

    def _init_weights(self, zero_init_residual: bool = True):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.0)

        if zero_init_residual:
            # Zero-initialize the last BN in each bottleneck so residual starts as identity
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0.0)

    # ---- forward ----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)          # (B, 64, H/?, W/?)
        x = self.layer1(x)        # 64 → 256
        x = self.layer2(x)        # 128 → 512
        x = self.layer3(x)        # 256 → 1024
        x = self.layer4(x)        # 512 → 2048
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# -----------------------------------------------------------------------------
# Factory functions
# -----------------------------------------------------------------------------

def _resnet50(stem_type: str, num_classes: int = 1000, in_ch: int = 3) -> ResNet:
    return ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        num_classes=num_classes,
        in_ch=in_ch,
        stem_type=stem_type,
        zero_init_residual=True,
    )


def resnet50_cifar(num_classes: int = 100, in_ch: int = 3) -> ResNet:
    """ResNet-50 with a CIFAR stem (3×3 conv stride=1, no maxpool)."""
    return _resnet50(stem_type="cifar", num_classes=num_classes, in_ch=in_ch)


def resnet50_imagenet(num_classes: int = 1000, in_ch: int = 3) -> ResNet:
    """ResNet-50 with an ImageNet stem (7×7 stride=2 + maxpool)."""
    return _resnet50(stem_type="imagenet", num_classes=num_classes, in_ch=in_ch)


# -----------------------------------------------------------------------------
# Quick self-test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    def count_params(m: nn.Module) -> int:
        return sum(p.numel() for p in m.parameters())

    mode = sys.argv[1] if len(sys.argv) > 1 else "cifar"
    if mode == "cifar":
        net = resnet50_cifar(num_classes=100)
        x = torch.randn(2, 3, 32, 32)
    else:
        net = resnet50_imagenet(num_classes=1000)
        x = torch.randn(2, 3, 224, 224)

    y = net(x)
    print("mode:", mode)
    print("output shape:", tuple(y.shape))
    print("params:", count_params(net))
