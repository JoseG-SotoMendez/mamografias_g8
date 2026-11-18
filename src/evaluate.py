"""
Evaluación de una corrida.

  1. Calcula puntuaciones en VALIDACIÓN y elige el umbral allí (Youden o sensibilidad objetivo).
  2. Guarda el umbral en runs/<name>/threshold.json (lo usa la app de Gradio).
  3. Reporta métricas en la partición pedida (--split val | test) con ESE umbral.
  4. Tabla de métricas a varios umbrales (efecto de mover el umbral).
  5. Lista y muestra falsos negativos / falsos positivos para el análisis de errores.

IMPORTANTE: --split test se usa UNA vez al final, con el modelo y umbral ya decididos.
Cada uso de prueba queda registrado en runs/<name>/registro_uso_test.txt.

Ejemplos:
  python src/evaluate.py --run runs/R1_resnet18_aug --split val --strategy sens --target-sens 0.90
  python src/evaluate.py --run runs/R1_resnet18_aug --split test --strategy sens --target-sens 0.90
  python src/evaluate.py --run runs/R1_resnet18_aug --split val --threshold 0.3   # umbral fijo
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

from common import LesionDataset, build_transform, get_device, load_image, predict_scores
from metrics import binary_metrics, select_threshold, threshold_table
from models import build_model


def load_run(run: Path, device):
    cfg = json.loads((run / "config.json").read_text())
    model = build_model(cfg["model"], pretrained=False)
    model.load_state_dict(torch.load(run / "best.pt", map_location=device))
    return cfg, model.to(device).eval()


def scores_for(df, cfg, model, device, bs=64):
    dl = DataLoader(LesionDataset(df, build_transform(cfg["img_size"], False, False)), batch_size=bs, shuffle=False)
    t0 = time.time()
    y, s = predict_scores(model, dl, device)
    ms_per_img = (time.time() - t0) / max(len(df), 1) * 1000
    return y, s, ms_per_img


def plot_confusion(m, path, title):
    cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["benigna", "maligna"]); ax.set_yticks([0, 1], ["benigna", "maligna"])
    ax.set_xlabel("predicción"); ax.set_ylabel("real"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_roc(y, s, thr, path, title):
    fpr, tpr, _ = roc_curve(y, s)
    m = binary_metrics(y, s, thr)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, label=f"AUC={m['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", c="gray")
    ax.scatter([1 - m["especificidad"]], [m["sensibilidad_maligna"]], c="red", zorder=3, label=f"umbral={thr:.3f}")
    ax.set_xlabel("1 - especificidad"); ax.set_ylabel("sensibilidad (maligna)"); ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_threshold_curve(table, thr, path):
    t = pd.DataFrame(table)
    fig, ax = plt.subplots(figsize=(6, 4))
    for col, name in [("sensibilidad_maligna", "sensibilidad"), ("especificidad", "especificidad"),
                      ("precision", "precisión"), ("f1", "F1")]:
        ax.plot(t.umbral, t[col], marker="o", label=name)
    ax.axvline(thr, ls="--", c="red", label="umbral elegido (val)")
    ax.set_xlabel("umbral"); ax.set_ylim(0, 1.02); ax.legend(); ax.set_title("Efecto del umbral")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def error_grid(err_df, path, title, n=8):
    err_df = err_df.head(n)
    if err_df.empty:
        return
    cols = min(4, len(err_df)); rows = (len(err_df) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 3))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (_, r) in zip(axes, err_df.iterrows()):
        ax.imshow(load_image(r.image_path).convert("L"), cmap="gray")
        ax.set_title(f"{r.lesion_uid[:22]}\n{r.abnormality_type} | score={r.score:.3f}", fontsize=7)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--strategy", choices=["youden", "sens"], default="sens")
    ap.add_argument("--target-sens", type=float, default=0.90)
    ap.add_argument("--threshold", type=float, default=None, help="umbral fijo (ignora --strategy)")
    args = ap.parse_args()

    run = Path(args.run)
    device = get_device()
    cfg, model = load_run(run, device)
    df = pd.read_csv(cfg["splits_csv"])

    # 1) Umbral SIEMPRE elegido en validación
    va = df[df.split == "val"].reset_index(drop=True)
    y_va, s_va, _ = scores_for(va, cfg, model, device)
    if args.threshold is not None:
        # Umbral fijo = exploración ("¿qué pasa si bajo el umbral?"). NO se guarda en threshold.json
        # para no cambiar en silencio el umbral oficial que usan la app y la evaluación final.
        thr, how = args.threshold, "fijo"
        print(f"Umbral FIJO de exploración: {thr:.4f} (threshold.json no se modifica)")
    else:
        thr = select_threshold(y_va, s_va, args.strategy, args.target_sens)
        how = args.strategy + (f" (objetivo sens={args.target_sens})" if args.strategy == "sens" else "")
        (run / "threshold.json").write_text(json.dumps(
            {"threshold": thr, "metodo": how, "elegido_en": "val", "fecha": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
        print(f"Umbral elegido en validación: {thr:.4f} ({how})")

    # 2) Partición a reportar
    if args.split == "test":
        with open(run / "registro_uso_test.txt", "a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} umbral={thr:.4f} metodo={how}\n")
        ev = df[df.split == "test"].reset_index(drop=True)
        y, s, ms = scores_for(ev, cfg, model, device)
    else:
        ev, y, s = va, y_va, s_va
        _, _, ms = scores_for(va.head(64), cfg, model, device)

    # Con umbral fijo los resultados van a otra carpeta para no pisar la evaluación oficial
    out = run / (f"eval_{args.split}" if args.threshold is None else f"eval_{args.split}_umbral_{thr:.2f}")
    out.mkdir(exist_ok=True)
    m = binary_metrics(y, s, thr)
    m["ms_por_imagen"] = round(ms, 2)
    m["dispositivo"] = str(device)
    (out / "metricas.json").write_text(json.dumps(m, indent=2))
    # El umbral elegido va SIN redondear: redondeado a 4 decimales (0.0664 en vez de 0.06638) la fila
    # contaba 199 TP en vez de 200 y no coincidía con metricas.json.
    table = threshold_table(y, s, sorted(set([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, thr])))
    pd.DataFrame(table).to_csv(out / "tabla_umbrales.csv", index=False)

    plot_confusion(m, out / "matriz_confusion.png", f"{run.name} - {args.split} (umbral {thr:.3f})")
    plot_roc(y, s, thr, out / "roc.png", f"ROC {run.name} - {args.split}")
    plot_threshold_curve(table, thr, out / "umbral_vs_metricas.png")

    ev = ev.copy(); ev["score"] = s; ev["pred"] = (s >= thr).astype(int)
    ev.to_csv(out / "predicciones.csv", index=False)
    fn = ev[(ev.label == 1) & (ev.pred == 0)].sort_values("score")              # malignas con score más bajo
    fp = ev[(ev.label == 0) & (ev.pred == 1)].sort_values("score", ascending=False)
    fn.to_csv(out / "falsos_negativos.csv", index=False)
    fp.to_csv(out / "falsos_positivos.csv", index=False)
    error_grid(fn, out / "falsos_negativos.png", "Falsos negativos (malignas clasificadas como benignas)")
    error_grid(fp, out / "falsos_positivos.png", "Falsos positivos (benignas clasificadas como malignas)")

    print(json.dumps(m, indent=2))
    print(f"FN={len(fn)} FP={len(fp)} | resultados en {out}")


if __name__ == "__main__":
    main()
