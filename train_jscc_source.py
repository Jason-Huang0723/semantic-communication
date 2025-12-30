#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_jscc_source.py
- Freeze ResNet-50 backbone (pretrained weights), train ONLY JSCC encoder/decoder at high SNR (near source coding).
- Split CIFAR-100 official train(50k) -> train/val; official test(10k) untouched.
- Save jscc_best_val.pt (best on val) and jscc_final.pt (last snapshot), including val_indices.
"""
"""python train_jscc_source.py --backbone_weights checkpoints/backbone_final.pt --code_dim 128 --snr_db 35.0 """

import argparse, os, csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset, random_split
import torchvision
import torchvision.transforms as T

from resnet50_custom import resnet50_cifar
from bottlenet_jscc import ResNet50LastConvJSCC

def get_transforms():
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    eval_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    return train_tf, eval_tf

def build_loaders(data_root, seed, val_ratio, batch_size, workers):
    train_tf, eval_tf = get_transforms()
    full_train = torchvision.datasets.CIFAR100(root=data_root, train=True,  download=True, transform=train_tf)
    n_total = len(full_train); n_val = int(n_total * val_ratio); n_train = n_total - n_val
    g = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [n_train, n_val], generator=g)
    val_indices = val_set.indices if hasattr(val_set, "indices") else None
    val_set = Subset(
        torchvision.datasets.CIFAR100(root=data_root, train=True, download=False, transform=eval_tf),
        val_indices
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True, persistent_workers=workers>0)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=workers, pin_memory=True, persistent_workers=workers>0)
    return train_loader, val_loader, val_indices

@torch.no_grad()
@torch.no_grad()
def evaluate(model, loader, device, amp=True, amp_dtype=torch.float16):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0; correct = 0; total = 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if amp and device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                logits = model(images)
                loss = criterion(logits, labels)
        else:
            logits = model(images); loss = criterion(logits, labels)
        loss_sum += loss.item() * labels.size(0)
        correct  += (logits.argmax(1) == labels).sum().item()
        total    += labels.size(0)              # ←←← 加上這行
    return loss_sum/total, 100.0*correct/total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone_weights", type=str, required=True, help="Path to pretrained backbone_final.pt")
    ap.add_argument("--code_dim", type=int, default=256, help="JSCC code dimension (smaller = stronger compression)")
    ap.add_argument("--snr_db", type=float, default=40.0, help="High SNR in dB (e.g., 30~40)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--outdir", type=str, default="./ckpts_jscc")
    ap.add_argument("--amp", type=str, default="fp16", choices=["fp16","bf16","off"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "off": torch.float32}[args.amp]
    use_amp = (args.amp != "off" and device.type == "cuda")

    # Data
    train_loader, val_loader, val_indices = build_loaders(
        args.data_root, args.seed, args.val_ratio, args.batch_size, args.workers
    )

    # Backbone + load weights
    base = resnet50_cifar(num_classes=100)
    ckpt = torch.load(args.backbone_weights, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = base.load_state_dict(state, strict=False)
    if unexpected: print("Warning: unexpected backbone keys:", unexpected)
    if missing:   print("Warning: missing backbone keys:", missing)

    # Wrap with JSCC and freeze backbone
    model = ResNet50LastConvJSCC(base, code_dim=args.code_dim, snr_db=args.snr_db).to(device)
    for m in [model.stem, model.layer1, model.layer2, model.layer3, model.layer4, model.avgpool, model.fc]:
        for p in m.parameters():
            p.requires_grad = False
        m.eval()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.outdir) / "train_jscc_log.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch","train_loss","train_acc","val_loss","val_acc","lr"])

    best_acc = 0.0
    best_path = str(Path(args.outdir) / "jscc_best_val.pt")
    final_path = str(Path(args.outdir) / "jscc_final.pt")

    for epoch in range(1, args.epochs+1):
        model.train()
        run_loss = 0.0; run_correct = 0; total = 0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    logits = model(images)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images); loss = criterion(logits, labels)
                loss.backward(); optimizer.step()
            run_loss += loss.item() * labels.size(0)
            run_correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = run_loss/total; train_acc = 100.0*run_correct/total
        val_loss, val_acc = evaluate(model, val_loader, device, amp=use_amp, amp_dtype=amp_dtype)
        scheduler.step()

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.3f}",
                                    f"{val_loss:.6f}", f"{val_acc:.3f}", f"{scheduler.get_last_lr()[0]:.8f}"])
        print(f"[{epoch}/{args.epochs}] train_acc={train_acc:.2f}%  val_acc={val_acc:.2f}%  best={best_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_indices": val_indices,
                "args": vars(args)
            }, best_path)

    torch.save({
        "epoch": args.epochs,
        "state_dict": model.state_dict(),
        "val_acc": best_acc,
        "val_indices": val_indices,
        "args": vars(args)
    }, final_path)

    print(f"Best-on-val: {best_path}")
    print(f"Final      : {final_path}")

if __name__ == "__main__":
    main()
