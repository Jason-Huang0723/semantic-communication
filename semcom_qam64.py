# semcom_qam64.py  — CNN redundancy + 64QAM + CNN decoder（64QAM、預設不做逐樣本功率正規化）
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

# ---------- utils ----------
def snr_db_to_sigma_complex(snr_db: float) -> float:
    """
    定義：複數符號平均功率 P=1，SNR = P / sigma^2  ⇒  sigma = 10^(-SNR/20)
    此 sigma 為複數雜訊的「總」標準差；I/Q 各維的標準差為 sigma/√2。
    """
    return 10.0 ** (-snr_db / 20.0)

def qam64_constellation(device=None, dtype=torch.float32):
    """
    64-QAM 星座 {±1,±3,±5,±7}^2，做「平均能量 = 1」正規化（等同 MATLAB UnitAveragePower=true）
    回傳 shape: (64, 2) ，每列為 [I, Q]
    """
    levels = torch.tensor([-7,-5,-3,-1, 1, 3, 5, 7], dtype=dtype, device=device)
    I, Q = torch.meshgrid(levels, levels, indexing="xy")             # 8x8 → 64
    const = torch.stack([I.reshape(-1), Q.reshape(-1)], dim=1)       # (64,2)
    avg_e = (const.pow(2).sum(dim=1)).mean()                         # E[|s|^2]
    return const / torch.sqrt(avg_e)                                 # (64,2), UnitAvgPower

# ---------- AWGN channel (complex via I/Q) ----------
class ChannelAWGNIQ(nn.Module):
    """
    複數 AWGN：I/Q 各加獨立 N(0, (sigma/√2)^2)，其中 sigma 由 SNR(dB) 依 P=1 的假設換算。
    備註：提供 set_snr_db() 讓外部掃 SNR。
    """
    def __init__(self, snr_db: float):
        super().__init__()
        sigma = snr_db_to_sigma_complex(snr_db)             # 複數總標準差
        sigma_per_dim = sigma / sqrt(2.0)                   # I/Q 各維
        self.register_buffer(
            "log_sigma_per_dim",
            torch.log(torch.tensor(sigma_per_dim, dtype=torch.float32))
        )

    @torch.no_grad()
    def set_snr_db(self, snr_db: float):
        sigma = snr_db_to_sigma_complex(snr_db)
        sigma_per_dim = sigma / sqrt(2.0)
        v = torch.log(torch.tensor(sigma_per_dim, dtype=torch.float32,
                                   device=self.log_sigma_per_dim.device))
        self.log_sigma_per_dim.copy_(v)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        s: (B, n_sym, 2) ，最後一維是 [I, Q]，其平均能量已為 1（星座 LUT 已正規化）。
        回傳：同 shape，I/Q 各自加高斯雜訊。
        """
        sigma = self.log_sigma_per_dim.exp()
        noise = torch.randn_like(s) * sigma
        return s + noise

# ---------- CNN blocks ----------
class ResidualBlock1D(nn.Module):
    def __init__(self, ch, k=3, p=1, s=1):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, kernel_size=k, padding=p, stride=s)
        self.bn1   = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, kernel_size=k, padding=p, stride=1)
        self.bn2   = nn.BatchNorm1d(ch)
        self.act   = nn.ReLU(inplace=True)
        self.down  = None
        if s != 1:
            self.down = nn.Sequential(
                nn.Conv1d(ch, ch, kernel_size=1, stride=s),
                nn.BatchNorm1d(ch)
            )

    def forward(self, x):
        idn = x if self.down is None else self.down(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + idn)

# ---------- CNN redundancy encoder ----------
class RedundancyEncoderCNN(nn.Module):
    """
    (B, m) → 1D-CNN 堆疊 → 線性插值到 n_sym → 1×1 投影成 latent_ch
    輸出 (B, n_sym, latent_ch)
    """
    def __init__(self, code_dim: int, n_sym: int, latent_ch: int = 128, width: int = 128, depth: int = 4):
        super().__init__()
        self.n_sym = n_sym
        self.inp   = nn.Conv1d(1, width, kernel_size=5, padding=2)
        blocks = []
        for _ in range(depth):
            blocks += [ResidualBlock1D(width), ResidualBlock1D(width)]
        self.body = nn.Sequential(*blocks)
        self.proj = nn.Conv1d(width, latent_ch, kernel_size=1)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, code_vec):                 # (B, m)
        x = code_vec.unsqueeze(1)               # (B, 1, m)
        x = self.act(self.inp(x))               # (B, W, m)
        x = self.body(x)                        # (B, W, m)
        x = F.interpolate(x, size=self.n_sym, mode="linear", align_corners=False)  # (B, W, n)
        x = self.proj(x)                        # (B, latent, n)
        return x.permute(0, 2, 1)               # (B, n, latent_ch)

# ---------- 64QAM 機率調變（ST Gumbel-Softmax），預設不做逐樣本功率正規化 ----------
class QAM64StochasticModulator(nn.Module):
    def __init__(self, latent_ch: int, tau: float = 1.0, power_norm: bool = False):
        super().__init__()
        self.tau = tau
        self.power_norm = power_norm
        self.proj = nn.Sequential(
            nn.Linear(latent_ch, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 64)   # 64 類 logits
        )
        self.register_buffer("const", qam64_constellation())  # (64,2)，UnitAvgPower

    def forward(self, z_lat):                                  # (B, n, latent_ch)
        logits = self.proj(z_lat)                              # (B, n, 64)
        # Straight-Through Gumbel-Softmax：前向 one-hot、反向可導
        y = F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1)  # (B, n, 64)
        s = torch.matmul(y, self.const)                        # (B, n, 2) → 64QAM I/Q
        # 不做逐樣本功率規範（星座 LUT 已具有平均能量=1）
        return s, logits

# ---------- CNN channel decoder ----------
class ChannelDecoderCNN(nn.Module):
    def __init__(self, n_sym: int, code_dim: int, width: int = 128, depth: int = 4):
        super().__init__()
        self.inp  = nn.Conv1d(2, width, kernel_size=5, padding=2)
        blocks = []
        for _ in range(depth):
            blocks += [ResidualBlock1D(width), ResidualBlock1D(width)]
        self.body = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(code_dim)
        self.proj = nn.Conv1d(width, 1, kernel_size=1)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, r_iq):                    # (B, n, 2)
        x = r_iq.permute(0, 2, 1)              # (B, 2, n)
        x = self.act(self.inp(x))              # (B, W, n)
        x = self.body(x)                       # (B, W, n)
        x = self.pool(x)                       # (B, W, m)
        x = self.proj(x)                       # (B, 1, m)
        return x.squeeze(1)                    # (B, m)

# ---------- 主系統（JSCC 可訓、backbone 凍結；64QAM + 複數 AWGN） ----------
class FrozenJSCCQAM64System(nn.Module):
    def __init__(self, base_parts: dict, jscc_enc: nn.Module, jscc_dec: nn.Module,
                 out_hw: tuple[int,int], code_dim: int, n_sym: int,
                 snr_db: float = 20.0, hidden: int = 512, tau: float = 1.0):
        super().__init__()
        # 凍結 backbone
        self.stem   = base_parts["stem"];   self.layer1 = base_parts["layer1"]
        self.layer2 = base_parts["layer2"]; self.layer3 = base_parts["layer3"]
        self.layer4 = base_parts["layer4"]; self.avgpool = base_parts["avgpool"]; self.fc = base_parts["fc"]
        for m in [self.stem,self.layer1,self.layer2,self.layer3,self.layer4,self.avgpool,self.fc]:
            for p in m.parameters():
                p.requires_grad = False

        # JSCC（可訓）
        self.jscc_enc = jscc_enc
        self.jscc_dec = jscc_dec

        # 通道側（CNN 冗餘、64QAM 調變、複數 AWGN、CNN 解碼）
        width = 256 if hidden >= 512 else 128
        latent_ch = 128
        self.redu = RedundancyEncoderCNN(code_dim=code_dim, n_sym=n_sym,
                                         latent_ch=latent_ch, width=width, depth=4)
        self.mod  = QAM64StochasticModulator(latent_ch=latent_ch, tau=tau, power_norm=False)
        self.chn  = ChannelAWGNIQ(snr_db=snr_db)
        self.cdec = ChannelDecoderCNN(n_sym=n_sym, code_dim=code_dim, width=width, depth=4)

        self.out_hw   = out_hw
        self.code_dim = code_dim
        self.n_sym    = n_sym

    @torch.no_grad()
    def set_snr_db(self, snr_db: float):
        self.chn.set_snr_db(snr_db)

    def forward(self, x):
        # Backbone conv 到最後一個 stage（固定）
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)

        # JSCC encoder（可訓）
        z = x
        s = self.jscc_enc(z)                         # (B, m, 1, 1)
        code_vec = s.view(s.size(0), -1)             # (B, m)

        # 冗餘 + 64QAM + 複數 AWGN + CNN 通道解碼
        z_lat = self.redu(code_vec)                  # (B, n, latent)
        s_iq, logits64 = self.mod(z_lat)             # (B, n, 2), (B, n, 64)

        # 在 s_iq = ... 之後、r_iq = self.chn(s_iq) 之前
        


        r_iq = self.chn(s_iq)                        # (B, n, 2)
        code_hat = self.cdec(r_iq)                   # (B, m)

        # JSCC decoder → head（固定）
        feat = self.jscc_dec(code_hat.view(code_hat.size(0), -1, 1, 1), self.out_hw)
        x = self.avgpool(feat)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits, {"logits64": logits64}
