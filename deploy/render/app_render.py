"""
App de Gradio publicada en Render (24/7), con el modelo R1 exportado a ONNX.

Es la misma interfaz de app.py (mismas validaciones, aviso y umbral de validación), pero la predicción la hace
ONNX Runtime en vez de PyTorch: el plan gratis de Render tiene 512 MB de RAM y la app con PyTorch ocupa ~510 MB.
El preprocesamiento es el mismo del entrenamiento (src/common.py): escala de grises a 3 canales,
redimensionado bilineal de Pillow a 224 x 224, /255 y normalización de ImageNet. Las puntuaciones coinciden con
PyTorch (deploy/exportar_onnx.py lo comprueba imagen por imagen).

Uso local:  python deploy/render/app_render.py      (busca modelo_r1.onnx en esta carpeta)
"""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
from PIL import Image

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
sys.path.insert(0, str(RAIZ))

from app import build_app  # noqa: E402  (la interfaz del proyecto)

MEDIA = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DESV = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def cargar_imagen(path):
    """Igual que src/common.py: PNG/JPG o DICOM -> RGB (gris replicado en 3 canales), sin PyTorch."""
    path = str(path)
    if path.lower().endswith(".dcm"):
        import pydicom
        arr = pydicom.dcmread(path).pixel_array.astype(np.float32)
        arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-6) * 255.0
        img = Image.fromarray(arr.astype(np.uint8))
    else:
        img = Image.open(path)
        img.load()
    return img.convert("L").convert("RGB")


class PredictorONNX:
    def __init__(self, modelo, threshold_json):
        opciones = ort.SessionOptions()
        opciones.intra_op_num_threads = 1  # el plan gratis tiene una fracción de CPU
        self.sesion = ort.InferenceSession(str(modelo), opciones, providers=["CPUExecutionProvider"])
        self.entrada = self.sesion.get_inputs()[0].name
        self.threshold = json.loads(Path(threshold_json).read_text())["threshold"]
        self.cfg = {"model": "resnet18 (ONNX)", "img_size": 224}
        self.run = SimpleNamespace(name="R1_resnet18_aug")
        redimensionar = lambda img: img.resize((224, 224), Image.BILINEAR)  # noqa: E731
        self.transform = SimpleNamespace(transforms=[redimensionar])

    def predict(self, img):
        x = np.asarray(img.resize((224, 224), Image.BILINEAR), dtype=np.float32) / 255.0
        x = ((x - MEDIA) / DESV).transpose(2, 0, 1)[None].astype(np.float32)
        logit = float(self.sesion.run(None, {self.entrada: x})[0].ravel()[0])
        return float(1.0 / (1.0 + np.exp(-logit)))


def main():
    modelo = Path(os.environ.get("MODELO_ONNX", AQUI / "modelo_r1.onnx"))
    pred = PredictorONNX(modelo, AQUI / "threshold.json")
    carpeta = AQUI / "ejemplos"
    ejemplos = sorted(str(p) for p in carpeta.glob("*.png")) if carpeta.exists() else []
    demo = build_app(pred, ejemplos, cargar_imagen=cargar_imagen)
    demo.queue(default_concurrency_limit=1, max_size=20).launch(
        server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))


if __name__ == "__main__":
    main()
