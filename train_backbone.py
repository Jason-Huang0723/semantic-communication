#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_backbone.py
- 依照 temp.py 的風格載入 CIFAR-100（官方 train/test）
- 將官方 50k train 依 seed 切成 train/val（預設 45k/5k）
- 用 train 更新參數、每個 epoch 在 val 上評估；val 變好就存最佳權重
- checkpoint 會包含：model state_dict、最佳 val 指標、以及 val 的 indices（供 test_backbone 復原同一組驗證集）

用法：
  python train_backbone.py --epochs 100 --batch-size 256 --workers 8 --val-ratio 0.1 --data-root ./data
"""
import argparse, os, math, time, csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset, random_split
import torchvision
import torchvision.transforms as T

# 你的自訂模型（不要動）
from resnet50_custom import resnet50_cifar

def build_model(num_classes=100):
    m = resnet50_cifar(num_classes=num_classes)
    return m

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

def get_datasets(data_root, seed, val_ratio=0.1):
    train_tf, eval_tf = get_transforms()
    full_train = torchvision.datasets.CIFAR100(root=data_root, train=True,  download=True, transform=train_tf)
    test_set   = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True, transform=eval_tf)

    # 切分官方 50k train -> train/val
    n_total = len(full_train)           # 50_000
    n_val   = int(n_total * val_ratio)  # 預設 5_000
    n_train = n_total - n_val

    g = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [n_train, n_val], generator=g)
    # 保存 val 的索引（random_split 內部是 Subset）
    val_indices = val_set.indices if hasattr(val_set, "indices") else None

    # val 需要「乾淨的 eval_tf」，把 subset 的底層 transform 換掉
    # random_split 給的是 Subset(full_train, idxs)，直接替換底層 dataset 的 transform 會同步影響 train；
    # 所以這裡把 val_set 重新包一層，避免互相影響。
    val_set = Subset(
        torchvision.datasets.CIFAR100(root=data_root, train=True, download=False, transform=eval_tf),
        val_indices
    )
    return train_set, val_set, test_set, val_indices

@torch.no_grad()
def evaluate(model, loader, device, amp_dtype=torch.float16):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = total = 0
    loss_sum = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(device.type=="cuda"), dtype=amp_dtype):
            logits = model(images)
            loss = criterion(logits, labels)
        loss_sum += loss.item() * labels.size(0)
        pred = logits.argmax(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return loss_sum/total, 100.0*correct/total

def param_and_grad_norm(model):
    p2 = g2 = 0.0
    for p in model.parameters():
        if p is not None and p.data is not None:
            p2 += p.data.float().norm(2).item()**2
        if p is not None and p.grad is not None:
            g2 += p.grad.data.float().norm(2).item()**2
    return math.sqrt(p2), math.sqrt(g2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=str, default='./data')
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=0.1)
    ap.add_argument('--weight-decay', type=float, default=5e-4)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--opt', type=str, default='sgd', choices=['sgd','adamw'])
    ap.add_argument('--amp', type=str, default='fp16', choices=['fp16','bf16','off'])
    ap.add_argument('--val-ratio', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--outdir', type=str, default='checkpoints')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    amp_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'off': torch.float32}[args.amp]
    use_amp = (args.amp != 'off' and device.type == 'cuda')

    # data
    train_set, val_set, test_set, val_indices = get_datasets(args.data_root, args.seed, args.val_ratio)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, persistent_workers=args.workers>0)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True, persistent_workers=args.workers>0)

    # model/opt
    model = build_model(100).to(device)
    if args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    log_csv = Path(args.outdir) / "train_log.csv"
    with open(log_csv, 'w', newline='') as f:
        csv.writer(f).writerow(["epoch","train_loss","train_acc","val_loss","val_acc","lr","p_l2","g_l2"])

    best_acc = 0.0
    best_path = str(Path(args.outdir) / "backbone_best_val.pt")
    final_path = str(Path(args.outdir) / "backbone_final.pt")

    for epoch in range(1, args.epochs+1):
        model.train()
        run_loss = 0.0; run_correct = 0; total = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
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

        train_loss = run_loss/total
        train_acc  = 100.0*run_correct/total
        val_loss, val_acc = evaluate(model, val_loader, device, amp_dtype=amp_dtype if use_amp else torch.float32)
        scheduler.step()

        p_l2, g_l2 = param_and_grad_norm(model)
        with open(log_csv, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.3f}",
                                    f"{val_loss:.6f}", f"{val_acc:.3f}",
                                    f"{scheduler.get_last_lr()[0]:.8f}", f"{p_l2:.6f}", f"{g_l2:.6f}"])
        print(f"[{epoch}/{args.epochs}] train_acc={train_acc:.2f}%  val_acc={val_acc:.2f}%  best={best_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_indices": val_indices,  # ★ 儲存驗證集索引
                "args": vars(args)
            }, best_path)

    # 最後快照
    torch.save({
        "epoch": args.epochs,
        "state_dict": model.state_dict(),
        "val_acc": best_acc,
        "val_indices": val_indices,
        "args": vars(args)
    }, final_path)
    print(f"最佳：{best_path}\n最後：{final_path}")

if __name__ == "__main__":
    main()
