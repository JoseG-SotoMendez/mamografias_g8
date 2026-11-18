"""
Inferencia sobre una sola imagen (la usa app.py y sirve para la demo del día 7).

Uso: python src/inference.py --run runs/R1_resnet18_aug --image ruta/recorte.png
"""
import argparse
import json
import time
from pathlib import Path

import torch

from common import build_transform, get_device, load_image
from models import build_model


class Predictor:
    def __init__(self, run_dir):
        self.run = Path(run_dir)
        self.device = get_device()
        self.cfg = json.loads((self.run / "config.json").read_text())
        self.model = build_model(self.cfg["model"], pretrained=False)
        self.model.load_state_dict(torch.load(self.run / "best.pt", map_location=self.device))
        self.model.to(self.device).eval()
        self.transform = build_transform(self.cfg["img_size"], False, False)
        thr_file = self.run / "threshold.json"
        self.threshold = json.loads(thr_file.read_text())["threshold"] if thr_file.exists() else 0.5

    @torch.no_grad()
    def predict(self, img):
        x = self.transform(img).unsqueeze(0).to(self.device)
        return float(torch.sigmoid(self.model(x)).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()
    p = Predictor(args.run)
    thr = args.threshold if args.threshold is not None else p.threshold
    t0 = time.time()
    score = p.predict(load_image(args.image))
    print(f"puntuación del modelo = {score:.4f} | umbral = {thr:.4f} | "
          f"clase = {'MALIGNA' if score >= thr else 'BENIGNA'} | {1000*(time.time()-t0):.1f} ms")


if __name__ == "__main__":
    main()
