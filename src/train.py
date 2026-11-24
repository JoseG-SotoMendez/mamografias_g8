"""
Entrenamiento de un modelo (una "corrida"). Usa SOLO train y val; nunca toca prueba.

Ejemplos:
  # Línea base
  python src/train.py --name R0_smallcnn --model smallcnn --epochs 25 --lr 1e-3
  # Transferencia de aprendizaje
  python src/train.py --name R1_resnet18_aug --model resnet18 --epochs 15 --lr 1e-4 --aug
  # Experimento controlado (solo cambia UNA cosa respecto a R1: sin aumentos)
  python src/train.py --name R2_resnet18_noaug --model resnet18 --epochs 15 --lr 1e-4

Salidas en runs/<name>/:
  config.json   todos los parámetros, dispositivo, tiempo, nº de parámetros
  log.csv       pérdida y métricas por época
  best.pt       pesos del mejor modelo según ROC-AUC de validación
  curvas.png    curvas de entrenamiento/validación
  val_preds.csv puntuaciones del mejor modelo en validación
"""
import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import LesionDataset, build_transform, get_device, predict_scores, set_seed
from metrics import binary_metrics
from models import build_model, count_params


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="nombre de la corrida (carpeta en runs/)")
    ap.add_argument("--splits", default="data/splits/lesiones.csv")
    ap.add_argument("--model", choices=["smallcnn", "resnet18", "efficientnet_b0"], default="resnet18")
    ap.add_argument("--no-pretrained", action="store_true", help="no usar pesos ImageNet")
    ap.add_argument("--freeze-backbone", action="store_true", help="entrenar solo la cabeza")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--aug", action="store_true", help="aumentos moderados SOLO en entrenamiento")
    ap.add_argument("--pos-weight", default="none", help="'none', 'auto' (n_neg/n_pos) o un número")
    ap.add_argument("--patience", type=int, default=5, help="parada temprana según ROC-AUC de validación")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-train", type=int, default=0, help="usar solo N lesiones de train (pruebas rápidas)")
    ap.add_argument("--incluir-val", action="store_true",
                    help="modelo FINAL: entrena con train+val durante --epochs fijas (elegidas antes en validación); "
                         "sin parada temprana y sin evaluar en val. Nunca usa prueba.")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    run = Path("runs") / args.name
    run.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.splits)
    if args.incluir_val:
        # Más datos para el modelo final: entrenamiento + validación. Las épocas ya se eligieron en validación
        # con otra corrida, así que aquí no hay nada que decidir con val.
        tr = df[df.split.isin(["train", "val"])].reset_index(drop=True)
    else:
        tr = df[df.split == "train"].reset_index(drop=True)
    va = df[df.split == "val"].reset_index(drop=True)
    if args.max_train:
        tr = tr.sample(min(args.max_train, len(tr)), random_state=args.seed).reset_index(drop=True)
    print(f"train: {len(tr)} lesiones ({tr.label.mean():.1%} malignas) | val: {len(va)} lesiones | device: {device}")

    g = torch.Generator(); g.manual_seed(args.seed)
    dl_tr = DataLoader(LesionDataset(tr, build_transform(args.img_size, True, args.aug)),
                       batch_size=args.batch_size, shuffle=True, num_workers=args.workers, generator=g,
                       pin_memory=device.type == "cuda")
    dl_va = DataLoader(LesionDataset(va, build_transform(args.img_size, False, False)),
                       batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = build_model(args.model, pretrained=not args.no_pretrained, freeze_backbone=args.freeze_backbone).to(device)
    n_total, n_train = count_params(model)

    if args.pos_weight == "none":
        pw = None
    elif args.pos_weight == "auto":
        pw = torch.tensor([(tr.label == 0).sum() / max((tr.label == 1).sum(), 1)], dtype=torch.float32)
    else:
        pw = torch.tensor([float(args.pos_weight)])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw.to(device) if pw is not None else None)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                              weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    use_amp = device.type == "cuda"
    scaler = make_scaler(use_amp)

    config = vars(args).copy()
    config.update({
        "splits_csv": str(Path(args.splits).resolve()),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch": torch.__version__, "python": platform.python_version(),
        "params_total": n_total, "params_entrenables": n_train,
        "pos_weight_valor": float(pw.item()) if pw is not None else None,
        "git_commit": git_commit(),
        "inicio": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    log, best_auc, best_epoch, wait = [], -1.0, -1, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n = 0.0, 0
        te = time.time()
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(x).squeeze(1), y)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            running += loss.item() * len(y); n += len(y)
        sched.step()

        if args.incluir_val:
            # Modelo final: val está dentro del entrenamiento, así que no se mide ni se usa para parar.
            row = {"epoca": epoch, "train_loss": running / n, "lr": optim.param_groups[0]["lr"], "seg": time.time() - te}
            log.append(row)
            print(" | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
            best_epoch, best_auc = epoch, float("nan")
            torch.save(model.state_dict(), run / "best.pt")  # se guarda la última época (número fijado de antemano)
            pd.DataFrame(log).to_csv(run / "log.csv", index=False)
            continue

        y_va, s_va = predict_scores(model, dl_va, device)
        with torch.no_grad():
            val_loss = float(criterion(torch.logit(torch.tensor(s_va).clamp(1e-6, 1 - 1e-6)).to(device),
                                       torch.tensor(y_va, dtype=torch.float32).to(device)).item())
        m = binary_metrics(y_va, s_va, 0.5)
        row = {"epoca": epoch, "train_loss": running / n, "val_loss": val_loss, "val_roc_auc": m["roc_auc"],
               "val_sens@0.5": m["sensibilidad_maligna"], "val_esp@0.5": m["especificidad"],
               "val_f1@0.5": m["f1"], "lr": optim.param_groups[0]["lr"], "seg": time.time() - te}
        log.append(row)
        print(" | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))

        if m["roc_auc"] > best_auc:
            best_auc, best_epoch, wait = m["roc_auc"], epoch, 0
            torch.save(model.state_dict(), run / "best.pt")
            pd.DataFrame({"lesion_uid": va.lesion_uid, "label": y_va, "score": s_va}).to_csv(run / "val_preds.csv", index=False)
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Parada temprana en época {epoch} (mejor: {best_epoch})")
                break
        pd.DataFrame(log).to_csv(run / "log.csv", index=False)

    config.update({"tiempo_entrenamiento_seg": round(time.time() - t0, 1), "mejor_epoca": best_epoch,
                   "mejor_val_roc_auc": None if args.incluir_val else best_auc,
                   "datos_entrenamiento": "train+val" if args.incluir_val else "train", "n_entrenamiento": int(len(tr)), "fin": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "tamano_modelo_MB": round((run / "best.pt").stat().st_size / 1e6, 2)})
    (run / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
    pd.DataFrame(log).to_csv(run / "log.csv", index=False)

    lg = pd.DataFrame(log)
    if args.incluir_val:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(lg.epoca, lg.train_loss, label="train+val")
        ax.set_title(f"{args.name}: pérdida (sin validación)"); ax.set_xlabel("época"); ax.legend()
        fig.tight_layout(); fig.savefig(run / "curvas.png", dpi=120)
        print(f"Listo. Modelo final entrenado {best_epoch} épocas con {len(tr)} lesiones (train+val). Carpeta: {run}")
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(lg.epoca, lg.train_loss, label="train"); ax[0].plot(lg.epoca, lg.val_loss, label="val")
    ax[0].set_title("Pérdida"); ax[0].set_xlabel("época"); ax[0].legend()
    ax[1].plot(lg.epoca, lg.val_roc_auc, label="val ROC-AUC")
    ax[1].axvline(best_epoch, ls="--", c="gray", label="mejor")
    ax[1].set_title("ROC-AUC validación"); ax[1].set_xlabel("época"); ax[1].legend()
    fig.suptitle(args.name); fig.tight_layout(); fig.savefig(run / "curvas.png", dpi=120)
    print(f"Listo. Mejor val ROC-AUC={best_auc:.4f} en época {best_epoch}. Carpeta: {run}")


if __name__ == "__main__":
    main()
