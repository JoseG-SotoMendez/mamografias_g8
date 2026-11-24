"""
Evaluación de un modelo YA ELEGIDO sobre un conjunto externo que nunca vio (p. ej. mini-MIAS).

  - No elige nada: el umbral se toma de un threshold.json fijado antes en validación de CBIS-DDSM
    (--threshold-from). Por defecto, el de la propia corrida.
  - Reporta métricas, matriz de confusión, ROC, tabla de umbrales y FN/FP como evaluate.py.
  - Cada uso queda registrado en runs/<run>/registro_uso_externo.txt.

Uso:
  python src/evaluate_external.py --run runs/R3_resnet18_aug_30ep --csv data/externo/mias/lesiones_mias.csv
  python src/evaluate_external.py --run runs/R4_final_trainval --csv data/externo/mias/lesiones_mias.csv \
      --threshold-from runs/R3_resnet18_aug_30ep/threshold.json
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

from evaluate import error_grid, load_run, plot_confusion, plot_roc, plot_threshold_curve, scores_for
from common import get_device
from metrics import binary_metrics, threshold_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--csv", required=True, help="CSV externo con image_path y label (formato de lesiones.csv)")
    ap.add_argument("--threshold-from", default=None, help="threshold.json elegido en validación (por defecto el de --run)")
    ap.add_argument("--nombre", default="mias", help="sufijo de la carpeta de resultados: eval_externo_<nombre>")
    args = ap.parse_args()

    run = Path(args.run)
    device = get_device()
    cfg, model = load_run(run, device)
    thr_file = Path(args.threshold_from) if args.threshold_from else run / "threshold.json"
    thr_info = json.loads(thr_file.read_text())
    thr = float(thr_info["threshold"])
    print(f"Umbral fijado antes, en validación de CBIS-DDSM: {thr:.4f} (de {thr_file})")

    ev = pd.read_csv(args.csv).reset_index(drop=True)
    y, s, ms = scores_for(ev, cfg, model, device)
    out = run / f"eval_externo_{args.nombre}"
    out.mkdir(exist_ok=True)
    with open(run / "registro_uso_externo.txt", "a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} csv={args.csv} umbral={thr:.4f} de={thr_file}\n")

    m = binary_metrics(y, s, thr)
    m.update({"ms_por_imagen": round(ms, 2), "dispositivo": str(device), "umbral_de": str(thr_file),
              "conjunto": args.csv})
    (out / "metricas.json").write_text(json.dumps(m, indent=2))
    table = threshold_table(y, s, sorted(set([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, thr])))
    pd.DataFrame(table).to_csv(out / "tabla_umbrales.csv", index=False)
    plot_confusion(m, out / "matriz_confusion.png", f"{run.name} - externo {args.nombre} (umbral {thr:.3f})")
    plot_roc(y, s, thr, out / "roc.png", f"ROC {run.name} - externo {args.nombre}")
    plot_threshold_curve(table, thr, out / "umbral_vs_metricas.png")

    ev = ev.copy(); ev["score"] = s; ev["pred"] = (s >= thr).astype(int)
    ev.to_csv(out / "predicciones.csv", index=False)
    fn = ev[(ev.label == 1) & (ev.pred == 0)].sort_values("score")
    fp = ev[(ev.label == 0) & (ev.pred == 1)].sort_values("score", ascending=False)
    fn.to_csv(out / "falsos_negativos.csv", index=False)
    fp.to_csv(out / "falsos_positivos.csv", index=False)
    error_grid(fn, out / "falsos_negativos.png", "Externo: falsos negativos")
    error_grid(fp, out / "falsos_positivos.png", "Externo: falsos positivos")
    print(json.dumps(m, indent=2))
    print(f"FN={len(fn)} FP={len(fp)} | resultados en {out}")


if __name__ == "__main__":
    main()
