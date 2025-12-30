# train_semcom_qam64.py
"""python train_semcom_qam64.py --jscc_weights .\ckpts_jscc\jscc_final.pt --n_sym 256 --snr_db 12.0 --epochs 50 """
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision, torchvision.transforms as T

from resnet50_custom import resnet50_cifar
from bottlenet_jscc import ResNet50LastConvJSCC
from semcom_qam64 import FrozenJSCCQAM64System

def get_transforms():
    mean=(0.5071,0.4867,0.4408); std=(0.2675,0.2565,0.2761)
    train_tf = T.Compose([T.RandomCrop(32,4), T.RandomHorizontalFlip(), T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
                          T.ToTensor(), T.Normalize(mean,std)])
    eval_tf  = T.Compose([T.ToTensor(), T.Normalize(mean,std)])
    return train_tf, eval_tf

def build_loaders(data_root, seed, val_ratio, batch, workers):
    train_tf, eval_tf = get_transforms()
    full = torchvision.datasets.CIFAR100(root=data_root, train=True, download=True, transform=train_tf)
    n_total=len(full); n_val=int(n_total*val_ratio); n_train=n_total-n_val
    g=torch.Generator().manual_seed(seed)
    train_set, val_idx_subset = random_split(full, [n_train, n_val], generator=g)
    # val 用 eval_tf & 相同索引
    base_for_val = torchvision.datasets.CIFAR100(root=data_root, train=True, download=False, transform=eval_tf)
    val_set = Subset(base_for_val, val_idx_subset.indices)
    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True, num_workers=workers,
                              pin_memory=True, persistent_workers=workers>0)
    val_loader   = DataLoader(val_set,   batch_size=batch, shuffle=False, num_workers=workers,
                              pin_memory=True, persistent_workers=workers>0)
    return train_loader, val_loader, val_idx_subset.indices

@torch.no_grad()
def evaluate(model, loader, device, use_amp=True, amp_dtype=torch.float16):
    model.eval(); ce=nn.CrossEntropyLoss()
    loss_sum=0; corr=0; tot=0
    device_type='cuda' if device.type=='cuda' else 'cpu'
    for images, labels in loader:
        images,labels=images.to(device,non_blocking=True),labels.to(device,non_blocking=True)
        with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
            logits,_=model(images); loss=ce(logits,labels)
        loss_sum+=loss.item()*labels.size(0); corr+=(logits.argmax(1)==labels).sum().item(); tot+=labels.size(0)
    return loss_sum/tot, 100.0*corr/tot

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--jscc_weights", type=str, required=True, help="train_jscc_source.py 產生的 jscc_best_val.pt 或 jscc_final.pt")
    ap.add_argument("--n_sym", type=int, default=256, help="每張圖使用的 64QAM 符號數（冗餘長度/通道碼長度）")
    ap.add_argument("--snr_db", type=float, default=12.0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--outdir", type=str, default="./ckpts_semcom")
    ap.add_argument("--amp", type=str, default="fp16", choices=["fp16","bf16","off"])
    args=ap.parse_args()

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype={"fp16":torch.float16,"bf16":torch.bfloat16,"off":torch.float32}[args.amp]
    use_amp=(args.amp!="off" and device.type=="cuda")

    # 1) 載入你已訓練好的 JSCC（含 backbone）
    base=resnet50_cifar(num_classes=100)
    jscc_ckpt=torch.load(args.jscc_weights, map_location="cpu")
    state = jscc_ckpt["state_dict"] if isinstance(jscc_ckpt,dict) and "state_dict" in jscc_ckpt else jscc_ckpt
    cfg   = jscc_ckpt.get("args", {}) if isinstance(jscc_ckpt,dict) else {}
    code_dim=int(cfg.get("code_dim",256))
    base_snr=float(cfg.get("snr_db",40.0))

    wrapper=ResNet50LastConvJSCC(base, code_dim=code_dim, snr_db=base_snr)
    _miss,_unexp=wrapper.load_state_dict(state, strict=False)
    # 拆出 backbone 與 JSCC enc/dec
    base_parts={"stem":wrapper.stem,"layer1":wrapper.layer1,"layer2":wrapper.layer2,"layer3":wrapper.layer3,
                "layer4":wrapper.layer4,"avgpool":wrapper.avgpool,"fc":wrapper.fc}
    jscc_enc=wrapper.enc; jscc_dec=wrapper.dec; out_hw=wrapper.out_hw

    # 2) 組合 VAE/Gumbel-Softmax 64QAM 系統（凍結 backbone+JSCC）
    model=FrozenJSCCQAM64System(base_parts, jscc_enc, jscc_dec, out_hw,
                                code_dim=code_dim, n_sym=args.n_sym,
                                snr_db=args.snr_db, hidden=512, tau=1.0).to(device)

    # 3) 資料
    train_loader, val_loader, val_indices = build_loaders(args.data_root, args.seed, args.val_ratio, args.batch, args.workers)

    # 4) 只訓練通道側參數
    params=[p for n,p in model.named_parameters() if p.requires_grad]
    optimizer=optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler=CosineAnnealingLR(optimizer, T_max=args.epochs)
    ce=nn.CrossEntropyLoss()
    scaler=torch.cuda.amp.GradScaler(enabled=use_amp)

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    log_csv=Path(args.outdir)/"train_semcom_qam64_log.csv"
    with open(log_csv,"w",newline="") as f: csv.writer(f).writerow(["epoch","train_loss","train_acc","val_loss","val_acc","lr"])

    best=0.0; best_path=str(Path(args.outdir)/"semcom_qam64_best_val.pt"); final_path=str(Path(args.outdir)/"semcom_qam64_final.pt")

    for ep in range(1, args.epochs+1):
        model.train()
        run_loss=0.0; run_corr=0; tot=0
        for images, labels in train_loader:
            images,labels=images.to(device,non_blocking=True),labels.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
                    logits,_=model(images); loss=ce(logits,labels)
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                logits,_=model(images); loss=ce(logits,labels); loss.backward(); optimizer.step()
            run_loss+=loss.item()*labels.size(0); run_corr+=(logits.argmax(1)==labels).sum().item(); tot+=labels.size(0)

        tr_loss=run_loss/tot; tr_acc=100.0*run_corr/tot
        val_loss,val_acc = evaluate(model, val_loader, device, use_amp=use_amp, amp_dtype=amp_dtype)
        scheduler.step()

        with open(log_csv,"a",newline="") as f:
            csv.writer(f).writerow([ep,f"{tr_loss:.6f}",f"{tr_acc:.3f}",f"{val_loss:.6f}",f"{val_acc:.3f}",f"{scheduler.get_last_lr()[0]:.8f}"])
        print(f"[{ep}/{args.epochs}] train={tr_acc:.2f}%  val={val_acc:.2f}%  best={best:.2f}%")

        if val_acc>best:
            best=val_acc
            torch.save({
                "epoch":ep, "state_dict":model.state_dict(), "val_acc":val_acc,
                "val_indices":val_indices, "args":vars(args),
                "code_dim":code_dim, "out_hw":out_hw
            }, best_path)

    torch.save({
        "epoch":args.epochs, "state_dict":model.state_dict(), "best_val_acc":best,
        "val_indices":val_indices, "args":vars(args),
        "code_dim":code_dim, "out_hw":out_hw
    }, final_path)
    print("Best:",best_path); print("Final:",final_path)

if __name__=="__main__":
    main()
