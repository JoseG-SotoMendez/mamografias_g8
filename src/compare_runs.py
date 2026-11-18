"""
Tabla comparativa de todas las corridas (para el informe).

Uso: python src/compare_runs.py --split val      # o --split test al final
Salida: reports/comparacion_<split>.csv y .md
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    args = ap.parse_args()
    rows = []
    for run in sorted(Path(args.runs).iterdir()):
        cfg_f, met_f = run / "config.json", run / f"eval_{args.split}" / "metricas.json"
        if not (cfg_f.exists() and met_f.exists()):
            continue
        cfg, m = json.loads(cfg_f.read_text()), json.loads(met_f.read_text())
        rows.append({
            "corrida": run.name, "modelo": cfg["model"], "aug": cfg["aug"], "lr": cfg["lr"],
            "epocas(mejor)": cfg.get("mejor_epoca"), "pos_weight": cfg.get("pos_weight"),
            "umbral": round(m["umbral"], 3),
            "sensibilidad": round(m["sensibilidad_maligna"], 3), "especificidad": round(m["especificidad"], 3),
            "precision": round(m["precision"], 3), "F1": round(m["f1"], 3), "ROC-AUC": round(m["roc_auc"], 3),
            "FN": m["FN"], "FP": m["FP"],
            "t_entren(min)": round(cfg.get("tiempo_entrenamiento_seg", 0) / 60, 1),
            "ms/img": m.get("ms_por_imagen"), "MB": cfg.get("tamano_modelo_MB"),
        })
    if not rows:
        print("No hay corridas evaluadas en", args.split); return
    df = pd.DataFrame(rows)
    out = Path("reports"); out.mkdir(exist_ok=True)
    df.to_csv(out / f"comparacion_{args.split}.csv", index=False)
    try:
        md = df.to_markdown(index=False)  # requiere 'tabulate'
    except ImportError:
        md = df.to_string(index=False)
    (out / f"comparacion_{args.split}.md").write_text(md)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
