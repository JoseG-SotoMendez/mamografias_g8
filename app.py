"""
Interfaz Gradio del prototipo.

Uso:
  python app.py --run runs/R1_resnet18_aug          # local
  python app.py --run runs/R1_resnet18_aug --share  # enlace público temporal (Colab)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr  # noqa: E402

from common import load_image  # noqa: E402
from inference import Predictor  # noqa: E402

AVISO = "⚠️ **Prototipo académico; no apto para diagnóstico ni decisiones clínicas.**"
FORMATOS = (".png", ".jpg", ".jpeg", ".dcm")
MAX_MB = 50


def build_app(pred: Predictor):
    def analizar(archivo, umbral):
        if archivo is None:
            return None, "Suba una imagen para analizar."
        path = archivo if isinstance(archivo, str) else archivo.name
        p = Path(path)
        if p.suffix.lower() not in FORMATOS:
            return None, f"❌ Formato no admitido ({p.suffix}). Use: {', '.join(FORMATOS)}."
        if p.stat().st_size > MAX_MB * 1e6:
            return None, f"❌ Archivo demasiado grande (> {MAX_MB} MB)."
        try:
            img = load_image(p)
        except Exception as e:
            return None, f"❌ No se pudo leer la imagen: {type(e).__name__}. ¿Es un archivo de imagen válido?"
        w, h = img.size
        if min(w, h) < 32:
            return None, f"❌ Imagen demasiado pequeña ({w}x{h} px)."

        avisos = []
        if min(w, h) > 2000:
            avisos.append("La imagen parece una **mamografía completa**. El modelo fue entrenado con "
                          "**recortes de lesión (ROI)**; el resultado puede no ser válido.")
        score = pred.predict(img)
        clase = "Patrón asociado a lesión **MALIGNA**" if score >= umbral else "Patrón asociado a lesión **BENIGNA**"
        md = (
            f"### Resultado del modelo\n"
            f"{clase}\n\n"
            f"- Puntuación del modelo: **{score:.3f}** (0 = benigna, 1 = maligna)\n"
            f"- Umbral de decisión: **{umbral:.3f}** (elegido en validación: {pred.threshold:.3f})\n"
            f"- Tamaño de la imagen: {w}x{h} px\n\n"
            f"_La puntuación NO es una probabilidad clínica calibrada._\n\n{AVISO}"
        )
        if avisos:
            md = "\n\n".join(f"⚠️ {a}" for a in avisos) + "\n\n" + md
        vis = pred.transform.transforms[0](img)  # solo el redimensionamiento, para visualizar
        return vis, md

    with gr.Blocks(title="Clasificación de lesiones mamográficas — G8") as demo:
        gr.Markdown("# Clasificación de lesiones mamográficas (benigna / maligna)\n" + AVISO)
        gr.Markdown(
            "**Entrada esperada:** un **recorte de la lesión** (ROI) de una mamografía, como los de CBIS-DDSM, "
            "en PNG, JPG o DICOM (.dcm). No se aceptan mamografías completas ni otro tipo de imágenes médicas "
            "como entrada válida para la tarea."
        )
        with gr.Row():
            with gr.Column():
                archivo = gr.File(label="Imagen (PNG / JPG / DICOM)", file_types=list(FORMATOS))
                umbral = gr.Slider(0.0, 1.0, value=round(pred.threshold, 3), step=0.01,
                                   label="Umbral de decisión (por defecto: el elegido en validación)")
                boton = gr.Button("Analizar", variant="primary")
            with gr.Column():
                vista = gr.Image(label="Imagen preprocesada", type="pil")
                salida = gr.Markdown()
        boton.click(analizar, [archivo, umbral], [vista, salida])
        gr.Markdown(f"Modelo: `{pred.cfg['model']}` · corrida `{pred.run.name}` · tamaño de entrada {pred.cfg['img_size']} px")
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    build_app(Predictor(args.run)).launch(share=args.share)
