# test_semcom_qam64.py
import argparse
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision, torchvision.transforms as T

from resnet50_custom import resnet50_cifar
from bottlenet_jscc import ResNet50LastConvJSCC
from semcom_qam64 import FrozenJSCCQAM64System

def get_tf():
    mean=(0.5071,0.4867,0.4408); std=(0.2675,0.2565,0.2761)
    return T.Compose([T.ToTensor(), T.Normalize(mean,std)])

@torch.no_grad()
def eval_once(model, loader, device, amp=True, dtype=torch.float16, name="EVAL"):
    ce=nn.CrossEntropyLoss(); model.eval()
    loss_sum=0; corr=0; tot=0; devt='cuda' if device.type=='cuda' else 'cpu'
    for x,y in loader:
        x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True)
        with torch.amp.autocast(device_type=devt, dtype=dtype, enabled=amp):
            logits,_=model(x); loss=ce(logits,y)
        loss_sum+=loss.item()*y.size(0); corr+=(logits.argmax(1)==y).sum().item(); tot+=y.size(0)
    return loss_sum/tot, 100.0*corr/tot, tot

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True, help="train_semcom_qam64.py 產生的 ckpt（best/final）")
    ap.add_argument("--jscc_weights", type=str, required=True, help="同訓練時所用之 jscc_best_val.pt 或 jscc_final.pt")
    ap.add_argument("--split", type=str, default="test", choices=["val","test"])
    ap.add_argument("--snrs", type=str, default="-6,-3,0,3,6,9,12,15,18,21,24", help="逗號分隔 SNR(dB)")
    ap.add_argument("--batch", type=int, default=256); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--amp", type=str, default="fp16", choices=["fp16","bf16","off"])
    args=ap.parse_args()

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype={"fp16":torch.float16,"bf16":torch.bfloat16,"off":torch.float32}[args.amp]
    use_amp=(args.amp!="off" and device.type=="cuda")

    # 還原 JSCC/Backbone（凍結）
    base=resnet50_cifar(num_classes=100)
    jscc_ckpt=torch.load(args.jscc_weights, map_location="cpu")
    state = jscc_ckpt["state_dict"] if isinstance(jscc_ckpt,dict) and "state_dict" in jscc_ckpt else jscc_ckpt
    cfg   = jscc_ckpt.get("args", {}) if isinstance(jscc_ckpt,dict) else {}
    code_dim=int(cfg.get("code_dim",256)); base_snr=float(cfg.get("snr_db",40.0))
    wrapper=ResNet50LastConvJSCC(base, code_dim=code_dim, snr_db=base_snr)
    wrapper.load_state_dict(state, strict=False)
    base_parts={"stem":wrapper.stem,"layer1":wrapper.layer1,"layer2":wrapper.layer2,"layer3":wrapper.layer3,
                "layer4":wrapper.layer4,"avgpool":wrapper.avgpool,"fc":wrapper.fc}
    jscc_enc=wrapper.enc; jscc_dec=wrapper.dec; out_hw=wrapper.out_hw

    # 還原 semcom 模型（只含通道側可學參數）
    sem_ckpt=torch.load(args.weights, map_location="cpu")
    n_sym=int(sem_ckpt["args"]["n_sym"]); init_snr=float(sem_ckpt["args"]["snr_db"])
    model=FrozenJSCCQAM64System(base_parts, jscc_enc, jscc_dec, out_hw,
                                code_dim=code_dim, n_sym=n_sym, snr_db=init_snr).to(device)
    model.load_state_dict(sem_ckpt["state_dict"], strict=False)

    # 資料
    tf=get_tf()
    if args.split=="val":
        assert "val_indices" in sem_ckpt and sem_ckpt["val_indices"] is not None, "ckpt 無 val_indices"
        base_train=torchvision.datasets.CIFAR100(root=args.data_root, train=True, download=True, transform=tf)
        dataset=Subset(base_train, sem_ckpt["val_indices"]); name="VAL"
    else:
        dataset=torchvision.datasets.CIFAR100(root=args.data_root, train=False, download=True, transform=tf); name="TEST"
    loader=DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                      pin_memory=True, persistent_workers=args.workers>0)

    # SNR 掃描
    snrs=[float(s.strip()) for s in args.snrs.split(",") if s.strip()!=""]
    print(f"Evaluate on {name} with SNR list: {snrs}")
    for sdb in snrs:
        model.set_snr_db(sdb)
        loss,acc,n=eval_once(model, loader, device, amp=use_amp, dtype=amp_dtype, name=name)
        print(f"SNR={sdb:>5.1f} dB | {name}: loss={loss:.4f}  acc={acc:.2f}%  n={n}")

if __name__=="__main__":
    main()
