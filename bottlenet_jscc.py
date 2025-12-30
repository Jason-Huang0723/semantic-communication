# bottlenet_jscc.py
import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- 可微 AWGN 通道 ----------
class ChannelAWGN(nn.Module):
    def __init__(self, snr_db: float = 14.5, learn_noise: bool = False):
        super().__init__()
        sigma = math.sqrt(1.0 / (10.0 ** (snr_db / 10.0)))  # P=1 假設下的噪聲標準差
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(sigma)), requires_grad=learn_noise)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        sigma = self.log_sigma.exp()
        return s + torch.randn_like(s) * sigma

# ---------- 小型 JSCC 編/解碼器（1×1 為主，輕量對稱） ----------
class JSCCEncoder(nn.Module):
    """
    (B, C, H, W) -> (B, m, 1, 1)
    """
    def __init__(self, in_ch: int, code_dim: int, hidden: int = 512, act: str = "tanh"):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, code_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(code_dim),
            nn.Tanh() if act == "tanh" else nn.Sigmoid(),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        s = self.enc(z)
        s = self.gap(s)  # (B, m, 1, 1)
        # 簡單功率正規化，避免訓練初期發散
        s = s / (s.detach().std(dim=(0,1,2,3), keepdim=True) + 1e-6)
        return s

class JSCCDecoder(nn.Module):
    """
    (B, m, 1, 1) -> (B, C, H, W)
    """
    def __init__(self, out_ch: int, code_dim: int, hidden: int = 512):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(code_dim, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
        )
        self.post = nn.Sequential(
            nn.Conv2d(hidden, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, s_hat: torch.Tensor, out_hw: Tuple[int,int]) -> torch.Tensor:
        x = self.pre(s_hat)                       # (B, hidden, 1, 1)
        x = F.interpolate(x, size=out_hw, mode="nearest")
        x = self.post(x)                          # (B, out_ch, H, W)
        return x

# ---------- 只在「最後一個 conv 層」插入 JSCC+AWGN ----------
class ResNet50LastConvJSCC(nn.Module):
    """
    把 base ResNet-50（CIFAR 版）在最後一個卷積層（layer4 結束）後插入 JSCC + AWGN，
    再接 avgpool / fc。支援端到端重訓。
    """
    def __init__(self, base: nn.Module, code_dim: int = 256, snr_db: float = 0, hidden: int = 512, act: str = "tanh"):
        super().__init__()
        self.base = base

        # 取 backbone 所在裝置
        dev = next(base.parameters()).device

        # 用 dummy 在「最後一層 conv 結束」處量出 (C,H,W)
        was_training = base.training
        base.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32, device=dev)  # CIFAR-100 輸入
            z, (C, H, W) = self._forward_until_last_conv(dummy)
        base.train(was_training)

        self.enc = JSCCEncoder(in_ch=C, code_dim=code_dim, hidden=hidden, act=act)
        self.chn = ChannelAWGN(snr_db=snr_db)
        self.dec = JSCCDecoder(out_ch=C, code_dim=code_dim, hidden=hidden)
        self.out_hw = (H, W)

        # 把 base 的各部分拿出來用
        self.stem   = base.stem
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool= base.avgpool
        self.fc     = base.fc

    @torch.no_grad()
    def _forward_until_last_conv(self, x: torch.Tensor):
        x = self.base.stem(x)
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)     # ← 最後一個 conv stage 結束（avgpool 之前）
        C, H, W = x.shape[1], x.shape[2], x.shape[3]
        return x, (C, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 到最後一個 conv 結束
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 在這裡插入 JSCC + AWGN
        z   = x
        s   = self.enc(z)                 # (B, m, 1, 1)
        # 保證每筆碼字平均功率=1
        B = s.size(0)
        p = s.view(B, -1).pow(2).mean(dim=1, keepdim=True)   # 每筆樣本的平均功率
        scale = (1.0 / (p + 1e-8)).sqrt()
        s = s * scale.view(B, 1, 1, 1)                       # 依 s 的實際形狀調整 view

        r   = self.chn(s)                 # AWGN
        z_h = self.dec(r, self.out_hw)    # (B, C, H, W)

        # 接回 head
        x = self.avgpool(z_h)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
