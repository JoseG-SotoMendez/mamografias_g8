"""
Descarga el modelo ONNX y las imágenes de ejemplo al construir la app en Render.

No van en el repositorio (ni pesos de 45 MB ni recortes de pacientes): el modelo está publicado en GitHub Pages
y en Hugging Face, y los ejemplos (recortes de la prueba de CBIS-DDSM y de mini-MIAS, licencia CC BY) en el
Space estático de Hugging Face. El script comprueba el MD5 del modelo: si no coincide, la construcción falla.
"""
import hashlib
import urllib.request
from pathlib import Path

AQUI = Path(__file__).resolve().parent
MD5_MODELO = "d0fc2b70f284728a136bf77d40587625"
FUENTES_MODELO = [
    "https://joseg-sotomendez.github.io/mamografias_g8/modelo_r1.onnx",
    "https://jostgreen-mamografias-g8-web.static.hf.space/modelo_r1.onnx",
]
BASE_EJEMPLOS = "https://jostgreen-mamografias-g8-web.static.hf.space/ejemplos/"
EJEMPLOS = [
    "01_cbis_maligna.png", "02_cbis_maligna.png", "03_cbis_maligna.png",
    "04_cbis_benigna.png", "05_cbis_benigna.png", "06_cbis_benigna.png",
    "07_cbis_maligna_falso_negativo.png", "08_cbis_benigna_falso_positivo.png",
    "09_miniMIAS_maligna.png", "11_miniMIAS_benigna.png", "13_mamografia_completa.png",
]


def bajar(url, destino):
    with urllib.request.urlopen(url, timeout=120) as r:
        destino.write_bytes(r.read())


def main():
    modelo = AQUI / "modelo_r1.onnx"
    for url in FUENTES_MODELO:
        try:
            bajar(url, modelo)
            if hashlib.md5(modelo.read_bytes()).hexdigest() == MD5_MODELO:
                print("modelo OK desde", url)
                break
            print("MD5 distinto en", url)
        except Exception as e:
            print("no se pudo bajar", url, e)
    else:
        raise SystemExit("No se pudo obtener el modelo con el MD5 esperado")

    carpeta = AQUI / "ejemplos"
    carpeta.mkdir(exist_ok=True)
    for nombre in EJEMPLOS:
        try:
            bajar(BASE_EJEMPLOS + nombre, carpeta / nombre)
        except Exception as e:  # sin ejemplos la app funciona igual
            print("ejemplo no disponible:", nombre, e)
    (carpeta / "zz_no_es_una_imagen.png").write_text("Esto es texto, no una imagen: la app debe rechazarlo.\n")
    print("ejemplos:", len(list(carpeta.glob("*.png"))))


if __name__ == "__main__":
    main()
