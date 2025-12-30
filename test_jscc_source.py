#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_jscc_source.py (SNR sweep 版)
- 載入 train_jscc_source.py 產生的 JSCC checkpoint
- 僅 forward（不更新權重）
- 可在 'val'（使用同一組 val_indices）或 'test' 上評估
- 會在 ckpt 原始 SNR 之外，額外掃一串 SNR（可自訂）

用法：
  # 在 val（同一組 val_indices）上做 SNR 掃描
  python test_jscc_source.py --weights .\ckpts_jscc\jscc_best_val.pt --split val

  # 在 test 上做 SNR 掃描，並自訂 SNR 清單
  python test_jscc_source.py --weights .\ckpts_jscc\jscc_final.pt --split test --snrs "-18,-15,-12,-9,-6,-3,0,3,6,9,12,15,18"
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

from resnet50_custom import resnet50_cifar
from bottlenet_jscc import ResNet50LastConvJSCC


def get_transforms():
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)
    return T.Compose([T.ToTensor(), T.Normalize(mean, std)])


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype=torch.float16, use_amp=True, name="EVAL"):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    correct = 0
    total = 0

    device_type = 'cuda' if device.type == 'cuda' else 'cpu'

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        loss_sum += loss.item() * labels.size(0)
        pred = logits.argmax(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return loss_sum / total, 100.0 * correct / total, total


def _set_channel_snr(model: nn.Module, snr_db: float):
    """
    依 SNR(dB) 設定通道雜訊 sigma。
    - 如果通道有 buffer: sigma → 直接填值
    - 如果通道可學（log_sigma）→ 覆寫參數數值
    """
    sigma = 10.0 ** (-snr_db / 20.0)  # 以單位功率碼字為前提的常見換算
    chn = model.chn if hasattr(model, "chn") else None
    if chn is None:
        raise RuntimeError("模型內找不到通道模組（model.chn）。")

    # 常見兩種寫法都支援
    if hasattr(chn, "sigma") and isinstance(chn.sigma, torch.Tensor):
        with torch.no_grad():
            chn.sigma.fill_(float(sigma))
    elif hasattr(chn, "log_sigma") and isinstance(chn.log_sigma, torch.Tensor):
        with torch.no_grad():
            chn.log_sigma.fill_(float(torch.log(torch.tensor(sigma))))
    else:
        raise RuntimeError("ChannelAWGN 不包含可設定的 sigma / log_sigma 欄位，請檢查通道實作。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True,
                    help="JSCC checkpoint 路徑（jscc_best_val.pt 或 jscc_final.pt）")
    ap.add_argument("--split", type=str, default="test", choices=["val", "test"])
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--amp", type=str, default="fp16", choices=["fp16", "bf16", "off"])
    # 額外要掃的 SNR 清單（逗號分隔）。若沒指定，就用預設那串。
    ap.add_argument("--snrs", type=str,
                    default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18 ,19,20 ,21 ,22,23, 24,25",
                    help="額外掃描的 SNR(dB) 列表（以逗號分隔），例如：\"-12,-8,-4, 0, 4, 8, 12, 16, 20\"")
    args = ap.parse_args()

    ckpt_path = Path(args.weights)
    assert ckpt_path.exists(), f"weights 檔案不存在：{ckpt_path}"

    # 讀 ckpt
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    cfg   = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    # 從 ckpt 讀回 code_dim / snr_db（若無則預設）
    code_dim = int(cfg.get("code_dim", 256))
    base_snr = float(cfg.get("snr_db", 40.0))

    # 組模型：backbone（你的 resnet50_custom）+ JSCC wrapper
    base  = resnet50_cifar(num_classes=100)
    model = ResNet50LastConvJSCC(base, code_dim=code_dim, snr_db=base_snr).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # 載入權重
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[Warn] missing keys:", missing)
    if unexpected:
        print("[Warn] unexpected keys:", unexpected)

    # Device / AMP
    device = next(model.parameters()).device
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "off": torch.float32}[args.amp]
    use_amp = (args.amp != "off" and device.type == "cuda")

    # 資料
    eval_tf = get_transforms()
    if args.split == "val":
        assert "val_indices" in ckpt and ckpt["val_indices"] is not None, \
            "checkpoint 裡沒有 val_indices；請用 train_jscc_source.py 訓練產生。"
        val_indices = ckpt["val_indices"]
        base_train = torchvision.datasets.CIFAR100(root=args.data_root, train=True,  download=True, transform=eval_tf)
        dataset = Subset(base_train, val_indices)
        split_name = "VAL"
    else:
        dataset = torchvision.datasets.CIFAR100(root=args.data_root, train=False, download=True, transform=eval_tf)
        split_name = "TEST"

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)

    # SNR sweep：先跑 ckpt 原本的 SNR，再跑使用者清單
    snr_list = [base_snr]
    if args.snrs:
        try:
            more = [float(s.strip()) for s in args.snrs.split(",") if s.strip() != ""]
            snr_list.extend(more)
        except Exception:
            print("[Warn] 解析 --snrs 失敗，忽略使用者清單。")
    # 去重、保序
    seen = set()
    snr_list = [x for x in snr_list if not (x in seen or seen.add(x))]

    print(f"\n將在 {split_name} 上評估以下 SNR(dB)：{snr_list}\n")

    # 逐一設定通道 SNR 並評估
    for snr_db in snr_list:
        _set_channel_snr(model, snr_db)
        val_loss, val_acc, n = evaluate(model, loader, device, amp_dtype=amp_dtype, use_amp=use_amp, name=split_name)
        print(f"SNR={snr_db:>5.1f} dB  |  {split_name}: loss={val_loss:.4f}  acc={val_acc:.2f}%  n={n}")

if __name__ == "__main__":
    main()
