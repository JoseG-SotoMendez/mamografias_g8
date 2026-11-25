"""
Arma la carpeta que se sube al Space de Hugging Face (la app 24/7).

Uso:
  python deploy/preparar_space.py --run <carpeta con best.pt, config.json, threshold.json> \
      --ejemplos <carpeta con PNG de prueba> --out build/space

Después (con la sesión de Hugging Face iniciada por el dueño de la cuenta):
  hf upload <usuario>/mamografias-g8 build/space . --repo-type=space
"""
import argparse
import shutil
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ejemplos", default=None)
    ap.add_argument("--out", default="build/space")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "src").mkdir(parents=True)
    for f in ("common.py", "models.py", "inference.py"):
        shutil.copy(RAIZ / "src" / f, out / "src" / f)
    shutil.copy(RAIZ / "app.py", out / "app_g8.py")
    for f in ("app.py", "README.md", "requirements.txt"):
        shutil.copy(RAIZ / "deploy" / "huggingface" / f, out / f)

    run = Path(args.run)
    dst = out / "modelo" / "R1_resnet18_aug"
    dst.mkdir(parents=True)
    for f in ("best.pt", "config.json", "threshold.json"):
        shutil.copy(run / f, dst / f)

    ej = out / "ejemplos"
    ej.mkdir()
    if args.ejemplos:
        for p in sorted(Path(args.ejemplos).glob("*.png")):
            shutil.copy(p, ej / p.name)
    (ej / "zz_no_es_una_imagen.png").write_text("Esto es texto, no una imagen: la app debe rechazarlo.\n")
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"Space listo en {out} ({total:.1f} MB, {len(list(ej.glob('*')))} ejemplos)")


if __name__ == "__main__":
    main()
