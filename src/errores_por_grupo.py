"""
Tasa de error por grupo (tipo de lesión, subtlety, assessment, vista) a partir de predicciones.csv.

Sirve para el análisis de errores: ¿el modelo falla más con calcificaciones que con masas?
¿Los falsos negativos se concentran en lesiones sutiles (subtlety baja)?

Uso: python src/errores_por_grupo.py --pred runs/R1_resnet18_aug/eval_val/predicciones.csv
"""
import argparse

import pandas as pd

def resumen(df, col):
    g = df.groupby(col)
    out = pd.DataFrame({
        "lesiones": g.size(),
        "malignas": g.label.sum(),
        "FN": g.apply(lambda d: int(((d.label == 1) & (d.pred == 0)).sum())),
        "FP": g.apply(lambda d: int(((d.label == 0) & (d.pred == 1)).sum())),
    })
    benignas = out.lesiones - out.malignas
    # % de malignas que se escaparon y % de benignas marcadas como malignas, dentro de cada grupo
    out["%FN_de_malignas"] = (100 * out.FN / out.malignas.where(out.malignas > 0)).round(1)
    out["%FP_de_benignas"] = (100 * out.FP / benignas.where(benignas > 0)).round(1)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="predicciones.csv de evaluate.py")
    args = ap.parse_args()
    df = pd.read_csv(args.pred)
    print(f"{len(df)} lesiones | FN={int(((df.label == 1) & (df.pred == 0)).sum())} "
          f"FP={int(((df.label == 0) & (df.pred == 1)).sum())}")
    for col in ["abnormality_type", "subtlety", "assessment", "image_view"]:
        if col in df.columns:
            print(f"\n=== Por {col} ===")
            print(resumen(df, col).to_string())

if __name__ == "__main__":
    main()
