"""
Preparación de CBIS-DDSM para clasificación binaria POR LESIÓN (benigna vs maligna).

Qué hace:
  1. Lee los CSV oficiales de descripción de casos (mass/calc, train/test).
  2. Mapea la etiqueta original `pathology` a la etiqueta del proyecto (0 = benigna, 1 = maligna).
  3. Localiza la imagen RECORTADA (cropped ROI) de cada lesión:
       - fuente "kaggle": versión JPEG (usa csv/dicom_info.csv para ubicar los JPG).
       - fuente "tcia":   DICOM originales descargados con NBIA Data Retriever
                          (los convierte a PNG en una carpeta caché).
  4. Revisa duplicados exactos (hash MD5 del archivo).
  5. Divide en entrenamiento / validación / prueba POR PACIENTE (StratifiedGroupKFold),
     comprobando que ninguna paciente quede en dos particiones.

Uso típico:
  python src/prepare_data.py --source kaggle --root /ruta/al/dataset --lesion both --bwc benign

Salidas (carpeta --out, por defecto data/splits):
  lesiones.csv        una fila por lesión con ruta de imagen, etiqueta y partición
  resumen.json        conteos por partición/clase/pacientes y chequeo de fuga
  descartadas.csv     filas que no se pudieron usar y el motivo
  duplicados.csv      grupos de archivos idénticos (si existen)
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Etiqueta original (columna `pathology`) -> etiqueta del proyecto
LABEL_MAP = {"MALIGNANT": 1, "BENIGN": 0, "BENIGN_WITHOUT_CALLBACK": 0}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def load_descriptions(root: Path) -> pd.DataFrame:
    csvs = sorted(p for p in root.rglob("*case_description*.csv"))
    if not csvs:
        sys.exit(f"No se encontraron CSV '*case_description*.csv' dentro de {root}")
    frames = []
    for p in csvs:
        df = normalize_columns(pd.read_csv(p))
        name = p.name.lower()
        df["source_csv"] = p.name
        df["official_split"] = "train" if "train" in name else "test"
        frames.append(df)
        print(f"  leído {p.name}: {len(df)} filas")
    df = pd.concat(frames, ignore_index=True)
    required = {"patient_id", "pathology", "abnormality_type", "cropped_image_file_path"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"Faltan columnas en los CSV: {missing}")
    return df


def series_uid_from_path(p: str):
    parts = str(p).strip().replace("\\", "/").split("/")
    return parts[-2] if len(parts) >= 3 else None


def resolve_kaggle(df: pd.DataFrame, root: Path) -> pd.Series:
    info_files = list(root.rglob("dicom_info.csv"))
    if not info_files:
        sys.exit("Fuente kaggle: no se encontró dicom_info.csv")
    info = pd.read_csv(info_files[0])
    info = info[info["SeriesDescription"].astype(str).str.contains("cropped", case=False)]
    jpeg_dirs = [p for p in root.rglob("jpeg") if p.is_dir()]
    jpeg_root = jpeg_dirs[0] if jpeg_dirs else root
    mapping = {}
    for _, r in info.iterrows():
        rel = str(r["image_path"]).replace("\\", "/")
        idx = rel.find("jpeg/")
        path = jpeg_root / rel[idx + 5:] if idx >= 0 else root / rel
        mapping.setdefault(str(r["SeriesInstanceUID"]), []).append(path)
    out = []
    for uid in df["series_uid"]:
        paths = [p for p in mapping.get(str(uid), []) if p.exists()]
        out.append(str(paths[0]) if paths else None)
    return pd.Series(out, index=df.index)


def resolve_tcia(df: pd.DataFrame, root: Path, cache: Path) -> pd.Series:
    try:
        import pydicom
    except ImportError:
        sys.exit("Fuente tcia: instale pydicom (pip install pydicom)")
    from PIL import Image

    cache.mkdir(parents=True, exist_ok=True)
    needed = set(df["series_uid"].dropna().astype(str))
    index = {}
    print("  indexando DICOM (solo encabezados)...")
    for f in root.rglob("*.dcm"):
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True,
                                 specific_tags=["SeriesInstanceUID", "SeriesDescription"])
        except Exception:
            continue
        uid = str(getattr(ds, "SeriesInstanceUID", ""))
        if uid in needed:
            index.setdefault(uid, []).append((f, str(getattr(ds, "SeriesDescription", ""))))

    resolved = {}
    for uid, files in index.items():
        target = cache / f"{uid}.png"
        if target.exists():
            resolved[uid] = str(target)
            continue
        chosen = None
        # 1) por descripción de la serie
        crops = [f for f, d in files if "crop" in d.lower()]
        candidates = crops if crops else [f for f, _ in files]
        # 2) heurística: la máscara ROI es binaria (2 valores); el recorte no
        for f in candidates:
            arr = pydicom.dcmread(f).pixel_array
            if len(np.unique(arr)) > 2:
                chosen = arr
                break
        if chosen is None:
            continue
        arr = chosen.astype(np.float32)
        arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-6) * 255.0
        Image.fromarray(arr.astype(np.uint8)).save(target)
        resolved[uid] = str(target)
    return df["series_uid"].astype(str).map(resolved)


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def patient_split(df: pd.DataFrame, seed: int, test_frac: float, val_frac: float) -> pd.Series:
    y = df["label"].values
    g = df["patient_id"].values
    n_test = max(2, round(1 / test_frac))
    sgkf = StratifiedGroupKFold(n_splits=n_test, shuffle=True, random_state=seed)
    trval_idx, test_idx = next(sgkf.split(df, y, g))
    sub = df.iloc[trval_idx]
    n_val = max(2, round((1 - test_frac) / val_frac))
    sgkf2 = StratifiedGroupKFold(n_splits=n_val, shuffle=True, random_state=seed)
    tr_rel, val_rel = next(sgkf2.split(sub, sub["label"].values, sub["patient_id"].values))
    split = pd.Series("train", index=df.index)
    split.iloc[test_idx] = "test"
    split.loc[sub.index[val_rel]] = "val"
    return split


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["kaggle", "tcia"], required=True)
    ap.add_argument("--root", required=True, help="carpeta raíz del dataset descargado")
    ap.add_argument("--out", default="data/splits")
    ap.add_argument("--cache", default="data/png_cache", help="solo tcia: dónde guardar PNG convertidos")
    ap.add_argument("--lesion", choices=["mass", "calc", "both"], default="both",
                    help="tipo de lesión a incluir (decisión del equipo; documentarla)")
    ap.add_argument("--bwc", choices=["benign", "exclude"], default="benign",
                    help="qué hacer con BENIGN_WITHOUT_CALLBACK")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--skip-md5", action="store_true", help="omitir revisión de duplicados")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("1) Leyendo CSV de descripción")
    df = load_descriptions(root)
    df["pathology"] = df["pathology"].astype(str).str.strip().str.upper()
    df["abnormality_type"] = df["abnormality_type"].astype(str).str.strip().str.lower()
    discarded = []

    if args.lesion != "both":
        key = "mass" if args.lesion == "mass" else "calcification"
        df = df[df["abnormality_type"] == key].copy()

    if args.bwc == "exclude":
        mask = df["pathology"] == "BENIGN_WITHOUT_CALLBACK"
        d = df[mask].copy(); d["motivo"] = "BENIGN_WITHOUT_CALLBACK excluida"
        discarded.append(d); df = df[~mask].copy()

    df["label"] = df["pathology"].map(LABEL_MAP)
    bad = df["label"].isna()
    if bad.any():
        d = df[bad].copy(); d["motivo"] = "pathology desconocida"
        discarded.append(d); df = df[~bad].copy()
    df["label"] = df["label"].astype(int)

    side_col = "left_or_right_breast" if "left_or_right_breast" in df.columns else None
    view_col = "image_view" if "image_view" in df.columns else None
    df["lesion_uid"] = (
        df["patient_id"].astype(str) + "_"
        + (df[side_col].astype(str) if side_col else "") + "_"
        + (df[view_col].astype(str) if view_col else "") + "_"
        + df["abnormality_type"] + "_"
        + df.get("abnormality_id", pd.Series(range(len(df)), index=df.index)).astype(str)
    )
    df["lesion_uid"] = df["lesion_uid"] + df.groupby("lesion_uid").cumcount().map(lambda i: "" if i == 0 else f"_dup{i}")

    print("2) Localizando imágenes recortadas (ROI)")
    df["series_uid"] = df["cropped_image_file_path"].map(series_uid_from_path)
    if args.source == "kaggle":
        df["image_path"] = resolve_kaggle(df, root)
    else:
        df["image_path"] = resolve_tcia(df, root, Path(args.cache))
    miss = df["image_path"].isna()
    if miss.any():
        d = df[miss].copy(); d["motivo"] = "imagen recortada no encontrada"
        discarded.append(d); df = df[~miss].copy()
    print(f"   lesiones con imagen: {len(df)} | sin imagen: {int(miss.sum())}")

    dup_info = None
    if not args.skip_md5:
        print("3) Revisando duplicados exactos (MD5)")
        df["md5"] = df["image_path"].map(md5)
        dups = df[df.duplicated("md5", keep=False)].sort_values("md5")
        if len(dups):
            dups[["md5", "lesion_uid", "patient_id", "pathology", "image_path"]].to_csv(out / "duplicados.csv", index=False)
            cross_patient = dups.groupby("md5")["patient_id"].nunique().gt(1).sum()
            dup_info = {"filas_duplicadas": int(len(dups)), "grupos": int(dups["md5"].nunique()),
                        "grupos_entre_pacientes_distintas": int(cross_patient)}
            # se conserva la primera aparición de cada archivo idéntico
            first = ~df.duplicated("md5", keep="first")
            d = df[~first].copy(); d["motivo"] = "archivo duplicado exacto"
            discarded.append(d); df = df[first].copy()
        print(f"   duplicados: {dup_info or 'ninguno'}")

    print("4) División por paciente")
    df = df.reset_index(drop=True)
    df["split"] = patient_split(df, args.seed, args.test_frac, args.val_frac)

    # Chequeo de fuga: ninguna paciente en dos particiones
    per_patient = df.groupby("patient_id")["split"].nunique()
    leaks = per_patient[per_patient > 1]
    assert leaks.empty, f"FUGA: pacientes en varias particiones: {list(leaks.index)[:10]}"

    keep = ["lesion_uid", "patient_id", "abnormality_type", "pathology", "label", "split",
            "official_split", "image_path"]
    for c in [side_col, view_col, "assessment", "subtlety", "md5"]:
        if c and c in df.columns:
            keep.append(c)
    df[keep].to_csv(out / "lesiones.csv", index=False)
    if discarded:
        pd.concat(discarded)[["patient_id", "pathology", "abnormality_type", "motivo"]].to_csv(
            out / "descartadas.csv", index=False)

    summary = {
        "parametros": vars(args),
        "lesiones_total": int(len(df)),
        "pacientes_total": int(df["patient_id"].nunique()),
        "por_particion": {
            s: {"lesiones": int((df.split == s).sum()),
                "pacientes": int(df.loc[df.split == s, "patient_id"].nunique()),
                "benignas": int(((df.split == s) & (df.label == 0)).sum()),
                "malignas": int(((df.split == s) & (df.label == 1)).sum())}
            for s in ["train", "val", "test"]
        },
        "pathology_original": df["pathology"].value_counts().to_dict(),
        "tipo_lesion": df["abnormality_type"].value_counts().to_dict(),
        "pacientes_en_varias_particiones": 0,
        "duplicados": dup_info,
        "descartadas": int(sum(len(d) for d in discarded)),
    }
    if view_col:
        summary["vistas"] = df[view_col].value_counts().to_dict()
    (out / "resumen.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary["por_particion"], indent=2))
    print(f"OK -> {out/'lesiones.csv'}  (sin fuga entre pacientes)")


if __name__ == "__main__":
    main()
