"""
Análisis exploratorio (EDA) sobre data/splits/lesiones.csv.

Genera en reports/eda/:
  balance_clases.png      lesiones benignas/malignas por partición
  tipo_vs_clase.png       masa/calcificación vs clase
  tamanos.png             dispersión ancho x alto de los recortes
  ejemplos_benignas.png / ejemplos_malignas.png
  eda_resumen.json        estadísticas numéricas

Uso: python src/eda.py --splits data/splits/lesiones.csv
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def image_size(path):
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:
        return (None, None)


def sample_grid(df, label, out_path, n=12, seed=0):
    sub = df[df.label == label].sample(min(n, (df.label == label).sum()), random_state=seed)
    cols = 6
    rows = max(1, (len(sub) + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for ax in axes:
        ax.axis("off")
    for ax, (_, r) in zip(axes, sub.iterrows()):
        ax.imshow(Image.open(r.image_path).convert("L"), cmap="gray")
        ax.set_title(f"{r.patient_id}\n{r.pathology[:12]}", fontsize=7)
    fig.suptitle("Malignas" if label == 1 else "Benignas")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/splits/lesiones.csv")
    ap.add_argument("--out", default="reports/eda")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.splits)

    # Balance por partición
    ct = pd.crosstab(df.split, df.label).reindex(["train", "val", "test"]).rename(columns={0: "benigna", 1: "maligna"})
    ax = ct.plot(kind="bar", figsize=(6, 4), rot=0)
    ax.set_ylabel("lesiones"); ax.set_title("Balance de clases por partición")
    plt.tight_layout(); plt.savefig(out / "balance_clases.png", dpi=120); plt.close()

    ct2 = pd.crosstab(df.abnormality_type, df.pathology)
    ax = ct2.plot(kind="bar", figsize=(7, 4), rot=0)
    ax.set_ylabel("lesiones"); ax.set_title("Tipo de lesión vs etiqueta original")
    plt.tight_layout(); plt.savefig(out / "tipo_vs_clase.png", dpi=120); plt.close()

    sizes = df.image_path.map(image_size)
    df["w"] = [s[0] for s in sizes]; df["h"] = [s[1] for s in sizes]
    fig, ax = plt.subplots(figsize=(5, 5))
    for lab, name in [(0, "benigna"), (1, "maligna")]:
        s = df[df.label == lab]
        ax.scatter(s.w, s.h, s=6, alpha=0.5, label=name)
    ax.set_xlabel("ancho (px)"); ax.set_ylabel("alto (px)"); ax.legend(); ax.set_title("Tamaño de los recortes")
    plt.tight_layout(); plt.savefig(out / "tamanos.png", dpi=120); plt.close()

    sample_grid(df, 0, out / "ejemplos_benignas.png")
    sample_grid(df, 1, out / "ejemplos_malignas.png")

    lesions_per_patient = df.groupby("patient_id").size()
    summary = {
        "lesiones": int(len(df)),
        "pacientes": int(df.patient_id.nunique()),
        "lesiones_por_paciente": {"media": float(lesions_per_patient.mean()),
                                  "max": int(lesions_per_patient.max())},
        "proporcion_malignas": float(df.label.mean()),
        "tamano_px": {"ancho_mediana": float(df.w.median()), "alto_mediana": float(df.h.median()),
                      "ancho_min": float(df.w.min()), "ancho_max": float(df.w.max())},
        "balance_por_particion": ct.to_dict(orient="index"),
        "imagenes_ilegibles": int(df.w.isna().sum()),
    }
    for col in ["image_view", "left_or_right_breast"]:
        if col in df.columns:
            summary[col] = df[col].value_counts().to_dict()
    (out / "eda_resumen.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"Figuras en {out}")


if __name__ == "__main__":
    main()
