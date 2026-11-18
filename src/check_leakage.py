"""
Comprobación rápida (útil para el control del día 7):
  - ninguna paciente aparece en dos particiones
  - ninguna imagen idéntica (MD5) aparece en dos particiones
  - distribución de clases por partición

Uso: python src/check_leakage.py --splits data/splits/lesiones.csv [--paciente P_00001]
"""
import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/splits/lesiones.csv")
    ap.add_argument("--paciente", help="muestra en qué partición están las lesiones de esta paciente")
    args = ap.parse_args()
    df = pd.read_csv(args.splits)

    print("Lesiones por partición y clase (0=benigna, 1=maligna):")
    print(pd.crosstab(df.split, df.label, margins=True))
    print("\nPacientes por partición:")
    print(df.groupby("split").patient_id.nunique())

    per_patient = df.groupby("patient_id").split.nunique()
    leaks = per_patient[per_patient > 1]
    print(f"\nPacientes en más de una partición: {len(leaks)}")
    if len(leaks):
        print(leaks.head(20))

    if "md5" in df.columns:
        per_hash = df.groupby("md5").split.nunique()
        print(f"Imágenes idénticas en más de una partición: {int((per_hash > 1).sum())}")

    sets = {s: set(df.loc[df.split == s, "patient_id"]) for s in ["train", "val", "test"]}
    print("\nIntersecciones de pacientes:")
    print("  train ∩ val :", len(sets["train"] & sets["val"]))
    print("  train ∩ test:", len(sets["train"] & sets["test"]))
    print("  val ∩ test  :", len(sets["val"] & sets["test"]))

    if args.paciente:
        print(f"\nLesiones de {args.paciente}:")
        cols = [c for c in ["lesion_uid", "image_view", "pathology", "split"] if c in df.columns]
        print(df.loc[df.patient_id == args.paciente, cols].to_string(index=False))

    print("\nRESULTADO:", "SIN FUGA ✅" if len(leaks) == 0 else "HAY FUGA ❌")


if __name__ == "__main__":
    main()
