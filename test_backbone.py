#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_backbone.py
- 載入 train_backbone.py 產生的 checkpoint
- 只 forward，不更新權重
- 可在 'val'（用訓練時同一組索引）或 'test' 上評估

用法：
  python test_backbone.py --weights checkpoints/backbone_best_val.pt --split val
  python test_backbone.py --weights checkpoints/backbone_final.pt --split test
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

from resnet50_custom import resnet50_cifar

def build_model(num_classes=100):
    return resnet50_cifar(num_classes=num_classes)

def get_transforms():
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)
    eval_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train_tf, eval_tf

@torch.no_grad()
def evaluate(model, loader, device, amp_dtype=torch.float16, name="VAL"):
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
    print(f"{name}: loss={loss_sum/total:.4f}  acc={100.0*correct/total:.2f}%  n={total}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, required=True)
    ap.add_argument('--split', type=str, default='test', choices=['val','test'])
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--data-root', type=str, default='./data')
    args = ap.parse_args()

    ckpt = torch.load(args.weights, map_location='cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_dtype = torch.float16 if device.type=='cuda' else torch.float32

    # datasets/loaders
    train_tf, eval_tf = get_transforms()
    if args.split == 'val':
        assert 'val_indices' in ckpt and ckpt['val_indices'] is not None, \
            "checkpoint 裡沒有 val_indices，請用 train_backbone.py 重新訓練產生。"
        base = torchvision.datasets.CIFAR100(root=args.data_root, train=True, download=True, transform=eval_tf)
        subset = Subset(base, ckpt['val_indices'])
        loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, persistent_workers=args.workers>0)
    else:  # test
        test_set = torchvision.datasets.CIFAR100(root=args.data_root, train=False, download=True, transform=eval_tf)
        loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, persistent_workers=args.workers>0)

    # build + load
    model = build_model(100).to(device)
    state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt['model_state']
    model.load_state_dict(state, strict=True)

    evaluate(model, loader, device, amp_dtype=amp_dtype, name=args.split.upper())

if __name__ == "__main__":
    main()
