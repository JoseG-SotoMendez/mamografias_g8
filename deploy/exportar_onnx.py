"""
Exporta el modelo de una corrida a ONNX (para la versión web) y comprueba que da lo mismo que PyTorch.

Uso: python deploy/exportar_onnx.py --run runs/R1_resnet18_aug --out web/modelo_r1.onnx [--imagenes carpeta]

La comprobación compara, imagen por imagen, la puntuación de PyTorch con la de ONNX Runtime.
Si alguna difiere en más de 1e-4, el script falla: preferimos no publicar a publicar algo distinto.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import load_image  # noqa: E402
from inference import Predictor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="web/modelo_r1.onnx")
    ap.add_argument("--imagenes", default=None, help="carpeta con PNG/JPG para comparar puntuaciones")
    args = ap.parse_args()

    pred = Predictor(args.run)
    model = pred.model.cpu().eval()
    x = torch.randn(1, 3, 224, 224)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, x, str(out), input_names=["imagen"], output_names=["logit"],
                      opset_version=17, dynamo=False)
    print(f"ONNX guardado en {out} ({out.stat().st_size / 1e6:.1f} MB)")

    import onnxruntime as ort
    ses = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    imgs = sorted(p for p in Path(args.imagenes).glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")) if args.imagenes else []
    entradas = [("aleatoria", x)] + [(p.name, pred.transform(load_image(p)).unsqueeze(0)) for p in imgs]
    peor = 0.0
    for nombre, t in entradas:
        with torch.no_grad():
            s_pt = float(torch.sigmoid(model(t)).item())
        s_ox = float(1 / (1 + np.exp(-ses.run(None, {"imagen": t.numpy()})[0].item())))
        peor = max(peor, abs(s_pt - s_ox))
        print(f"{nombre:45s} PyTorch={s_pt:.6f}  ONNX={s_ox:.6f}  dif={abs(s_pt - s_ox):.2e}")
    print(f"Diferencia máxima: {peor:.2e}")
    if peor > 1e-4:
        sys.exit("ERROR: ONNX no reproduce a PyTorch; no publicar.")


if __name__ == "__main__":
    main()
