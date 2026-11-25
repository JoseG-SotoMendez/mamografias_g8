"""
Punto de entrada del Space de Hugging Face (la app 24/7 del proyecto).

Reutiliza la misma interfaz del repositorio (app_g8.py = copia de app.py) con el modelo R1 y el umbral
elegido en validación. Los ejemplos de la carpeta ejemplos/ aparecen como botones debajo de la app.
"""
import sys
from pathlib import Path

AQUI = Path(__file__).parent
sys.path.insert(0, str(AQUI / "src"))
sys.path.insert(0, str(AQUI))

import torch  # noqa: E402

torch.set_num_threads(2)

try:  # En hardware ZeroGPU, Hugging Face exige al menos una función marcada con @spaces.GPU.
    import spaces  # noqa: E402

    @spaces.GPU
    def _gpu_no_usada():  # La predicción corre en CPU (una ResNet18 tarda ~0,1 s); esto no se llama nunca.
        return None
except ImportError:
    pass

from app_g8 import build_app  # noqa: E402
from inference import Predictor  # noqa: E402

# "zz_no_es_una_imagen.png" es un archivo de texto con extensión .png: sirve para ver el mensaje de error.
ejemplos = sorted(str(p) for p in (AQUI / "ejemplos").glob("*") if p.suffix.lower() in (".png", ".jpg"))
demo = build_app(Predictor(AQUI / "modelo" / "R1_resnet18_aug"), ejemplos)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch()
