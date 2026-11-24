"""
Conjunto EXTERNO de prueba: mini-MIAS (Mammographic Image Analysis Society, Reino Unido).

El modelo nunca vio estas imágenes: otro escáner, otro país, otra época y otra resolución
(200 µm por píxel, mamografías de 1024x1024 en PGM). Sirve para medir si lo aprendido en
CBIS-DDSM se sostiene con datos totalmente nuevos.

Qué hace:
  1. Lee Info.txt (una línea por anomalía: REFNUM, tejido, clase, severidad B/M, X, Y, RADIO).
  2. Descarta mamografías normales (NORM) y anomalías sin coordenadas o sin severidad.
  3. Recorta un cuadrado centrado en la lesión de lado 2 * RADIO * --margen (mínimo --min-lado px),
     como los recortes ROI de CBIS-DDSM. En MIAS el origen de Y está ABAJO a la izquierda.
  4. Guarda los recortes en PNG y escribe lesiones_mias.csv con el mismo formato que lesiones.csv
     (split = "externo"), para evaluarlo con src/evaluate_external.py.

Mapeo de etiquetas: SEVERITY M -> 1 (maligna), B -> 0 (benigna).
Tipo: CALC -> calcification; CIRC, SPIC, MISC, ARCH, ASYM -> mass (lesión no cálcica).
Paciente: las imágenes van en pares izquierda/derecha (mdb001-mdb002 = paciente 1).

Uso: python src/prepare_mias.py --root <carpeta con Info.txt y los .pgm> --out data/externo/mias
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image

TIPO = {"CALC": "calcification", "CIRC": "mass", "SPIC": "mass", "MISC": "mass", "ARCH": "mass", "ASYM": "mass"}


def leer_info(path: Path) -> pd.DataFrame:
    filas = []
    for linea in path.read_text(errors="ignore").splitlines():
        p = linea.split()
        if not p or not p[0].lower().startswith("mdb"):
            continue
        fila = {"refnum": p[0].lower(), "tejido": p[1] if len(p) > 1 else None,
                "clase": p[2] if len(p) > 2 else None, "severidad": None, "x": None, "y": None, "radio": None}
        if len(p) >= 7:
            fila["severidad"] = p[3]
            try:
                fila["x"], fila["y"], fila["radio"] = int(p[4]), int(p[5]), int(p[6])
            except ValueError:
                pass  # coordenadas con nota ("*NOTE 3*") u otro formato: se descarta más abajo
        elif len(p) >= 4:
            fila["severidad"] = p[3]
        filas.append(fila)
    return pd.DataFrame(filas)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="carpeta que contiene Info.txt y las imágenes mdbXXX.pgm")
    ap.add_argument("--out", default="data/externo/mias")
    ap.add_argument("--margen", type=float, default=1.3, help="lado del recorte = 2 * radio * margen")
    ap.add_argument("--min-lado", type=int, default=48, help="lado mínimo del recorte en píxeles")
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    (out / "recortes").mkdir(parents=True, exist_ok=True)
    info_files = sorted(root.rglob("Info.txt")) or sorted(root.rglob("info.txt"))
    if not info_files:
        raise SystemExit(f"No se encontró Info.txt dentro de {root}")
    info = leer_info(info_files[0])
    imagenes = {p.stem.lower(): p for p in root.rglob("*.pgm")}
    print(f"Info.txt: {len(info)} líneas | imágenes .pgm: {len(imagenes)}")

    total_anom = info[info.clase != "NORM"]
    motivos = {
        "normal (sin lesión)": int((info.clase == "NORM").sum()),
        "sin severidad B/M": int(total_anom.severidad.isin(["B", "M"]).eq(False).sum()),
    }
    ok = total_anom[total_anom.severidad.isin(["B", "M"])].copy()
    sin_coord = ok.x.isna() | ok.y.isna() | ok.radio.isna()
    motivos["sin coordenadas"] = int(sin_coord.sum())
    ok = ok[~sin_coord].copy()

    filas, cuenta = [], {}
    for _, r in ok.iterrows():
        src = imagenes.get(r.refnum)
        if src is None:
            motivos["imagen no encontrada"] = motivos.get("imagen no encontrada", 0) + 1
            continue
        img = Image.open(src).convert("L")
        w, h = img.size
        cx, cy = int(r.x), h - int(r.y)  # Y de MIAS se mide desde abajo
        lado = max(args.min_lado, int(round(2 * r.radio * args.margen)))
        x0, y0 = max(0, cx - lado // 2), max(0, cy - lado // 2)
        x1, y1 = min(w, x0 + lado), min(h, y0 + lado)
        recorte = img.crop((x0, y0, x1, y1))
        k = cuenta.get(r.refnum, 0); cuenta[r.refnum] = k + 1
        nombre = f"{r.refnum}_{k}.png"
        recorte.save(out / "recortes" / nombre)
        num = int(r.refnum[3:])
        filas.append({
            "lesion_uid": f"MIAS_{r.refnum}_{k}",
            "patient_id": f"MIAS_P{(num + 1) // 2:03d}",
            "abnormality_type": TIPO.get(r.clase, "mass"),
            "pathology": "MALIGNANT" if r.severidad == "M" else "BENIGN",
            "label": 1 if r.severidad == "M" else 0,
            "split": "externo",
            "image_path": str((out / "recortes" / nombre).resolve()),
            "clase_mias": r.clase, "tejido": r.tejido, "radio_px": int(r.radio),
        })

    df = pd.DataFrame(filas)
    df.to_csv(out / "lesiones_mias.csv", index=False)
    resumen = {
        "lesiones": int(len(df)), "pacientes": int(df.patient_id.nunique()) if len(df) else 0,
        "malignas": int(df.label.sum()) if len(df) else 0, "benignas": int((df.label == 0).sum()) if len(df) else 0,
        "tipo_lesion": df.abnormality_type.value_counts().to_dict() if len(df) else {},
        "descartadas": motivos, "parametros": vars(args),
    }
    (out / "resumen_mias.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(json.dumps(resumen, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
