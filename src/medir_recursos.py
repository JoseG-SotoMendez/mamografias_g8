"""
Mide cuánta memoria y tiempo gastan los modelos, con imágenes reales del conjunto de entrenamiento.

No entrena nada nuevo ni toca validación o prueba: solo corre unos pasos para medir.
- GPU (si hay): pasos de entrenamiento con lotes de 32, AMP y aumentos, igual que train.py.
  Se mide la memoria máxima reservada por PyTorch y el tiempo por paso.
- CPU: predicción de UNA imagen a la vez (lo que hace la app), latencia y memoria del proceso.

Uso: python src/medir_recursos.py --modelos smallcnn resnet18 efficientnet_b0 --out reports/recursos
"""
import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

import pandas as pd
import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import LesionDataset, build_transform, set_seed
from models import build_model, count_params

MB = 1024 ** 2


def rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / MB


def medir_gpu(nombre, df, pasos, batch):
    dev = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build_model(nombre, pretrained=nombre != "smallcnn").to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    crit = nn.BCEWithLogitsLoss()
    dl = DataLoader(LesionDataset(df, build_transform(224, True, True)), batch_size=batch,
                    shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    model.train()
    tiempos, it = [], iter(dl)
    for i in range(pasos + 5):
        x, y = next(it)
        x, y = x.to(dev, non_blocking=True), y.to(dev)
        torch.cuda.synchronize(); t = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda"):
            loss = crit(model(x).squeeze(1), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        torch.cuda.synchronize()
        if i >= 5:  # los primeros pasos calientan cuDNN y no cuentan
            tiempos.append(time.perf_counter() - t)
    entreno = {
        "gpu": torch.cuda.get_device_name(0),
        "memoria_max_asignada_MB": round(torch.cuda.max_memory_allocated() / MB, 1),
        "memoria_max_reservada_MB": round(torch.cuda.max_memory_reserved() / MB, 1),
        "ms_por_paso_gpu": round(1000 * statistics.median(tiempos), 1),
        "imagenes_por_seg_gpu_sin_carga_de_datos": round(batch / statistics.median(tiempos), 1),
    }
    # Predicción en GPU por lotes (como evaluate.py)
    model.eval(); torch.cuda.reset_peak_memory_stats()
    x = torch.randn(batch, 3, 224, 224, device=dev)
    with torch.no_grad(), torch.autocast("cuda"):
        for _ in range(5): model(x)
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(20): model(x)
        torch.cuda.synchronize()
    entreno["imagenes_por_seg_prediccion_gpu"] = round(20 * batch / (time.perf_counter() - t), 1)
    entreno["memoria_max_prediccion_gpu_MB"] = round(torch.cuda.max_memory_allocated() / MB, 1)
    del model, opt
    torch.cuda.empty_cache()
    return entreno


def _cpu_en_proceso_nuevo(nombre, img_path, repeticiones):
    """Corre en un proceso limpio para que la memoria medida sea solo la de este modelo."""
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    from common import load_image
    x = build_transform(224, False, False)(load_image(img_path)).unsqueeze(0)
    base = rss_mb()  # Python + PyTorch cargados, sin modelo
    model = build_model(nombre, pretrained=False).eval()
    tiempos = []
    with torch.no_grad():
        for i in range(repeticiones + 3):
            t = time.perf_counter(); model(x); dt = time.perf_counter() - t
            if i >= 3:
                tiempos.append(dt)
    return {
        "cpu": platform.processor() or platform.machine(),
        "hilos": torch.get_num_threads(),
        "ms_por_imagen_cpu_mediana": round(1000 * statistics.median(tiempos), 1),
        "ms_por_imagen_cpu_max": round(1000 * max(tiempos), 1),
        "memoria_base_python_torch_MB": round(base, 1),
        "memoria_extra_por_modelo_y_prediccion_MB": round(rss_mb() - base, 1),
        "memoria_proceso_total_MB": round(rss_mb(), 1),
    }


def medir_cpu(nombre, img_path, repeticiones):
    import subprocess
    import sys
    r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--_cpu", nombre,
                        str(Path(img_path).resolve()), str(repeticiones)],
                       capture_output=True, text=True, cwd=Path(__file__).resolve().parent)
    linea = [l for l in r.stdout.splitlines() if l.startswith("{")]
    if not linea:
        raise RuntimeError(r.stderr[-2000:])
    return json.loads(linea[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/splits/lesiones.csv")
    ap.add_argument("--modelos", nargs="+", default=["smallcnn", "resnet18"])
    ap.add_argument("--pasos", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imagen", default=None, help="imagen para medir en CPU (por defecto, la primera de train)")
    ap.add_argument("--out", default="reports/recursos")
    args = ap.parse_args()
    set_seed(42)

    hay_gpu = torch.cuda.is_available()
    df = None
    if hay_gpu or args.imagen is None:
        df = pd.read_csv(args.splits)
        df = df[df.split == "train"].reset_index(drop=True)
    img = args.imagen or df.image_path.iloc[0]

    res = {"torch": torch.__version__, "python": platform.python_version(),
           "fecha": time.strftime("%Y-%m-%d %H:%M:%S"), "batch": args.batch, "modelos": {}}
    for nombre in args.modelos:
        m = build_model(nombre, pretrained=False)
        total, _ = count_params(m)
        fila = {"parametros": total, "pesos_MB": round(total * 4 / MB, 1)}
        del m
        if hay_gpu:
            fila.update(medir_gpu(nombre, df, args.pasos, args.batch))
        fila.update(medir_cpu(nombre, img, 30))
        res["modelos"][nombre] = fila
        print(nombre, json.dumps(fila, ensure_ascii=False))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sufijo = "gpu" if hay_gpu else "cpu"
    (out / f"recursos_{sufijo}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print("guardado en", out / f"recursos_{sufijo}.json")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 5 and sys.argv[1] == "--_cpu":
        print(json.dumps(_cpu_en_proceso_nuevo(sys.argv[2], sys.argv[3], int(sys.argv[4]))))
    else:
        main()
