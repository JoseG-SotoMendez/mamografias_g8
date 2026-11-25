# Clasificación de lesiones mamográficas benignas y malignas — Grupo 8

Aprendizaje Computacional – Ingeniería de Sistemas
Integrantes: José Gil Soto Méndez · Andrés Alfonso Luna Moreno · Miguel Francisco Guerra Negrete

> ⚠️ **Prototipo académico; no apto para diagnóstico ni decisiones clínicas.**
> El modelo clasifica **recortes de lesiones** según su patología de referencia en CBIS-DDSM; eso **no** equivale a diagnosticar a una paciente.

## Tarea
- **Unidad de análisis:** lesión (recorte ROI de una vista CC o MLO).
- **Etiqueta:** `pathology` de CBIS-DDSM → 0 = benigna (`BENIGN`, `BENIGN_WITHOUT_CALLBACK`), 1 = maligna (`MALIGNANT`). Ver `docs/tabla_etiquetas.md`.
- **División:** entrenamiento / validación / prueba **por paciente** (70/15/15 aprox.), sin fuga.

## Estructura
```
app.py                    interfaz Gradio
src/prepare_data.py       CSV oficiales → lesiones.csv con partición por paciente
src/eda.py                análisis exploratorio (figuras en reports/eda)
src/check_leakage.py      verifica que ninguna paciente esté en dos particiones
src/train.py              entrena una corrida (solo train/val)
src/evaluate.py           umbral elegido en val; métricas, ROC, matriz, FN/FP
src/inference.py          predicción de una imagen
src/compare_runs.py       tabla comparativa de corridas
src/errores_por_grupo.py  tasa de FN/FP por tipo de lesión, subtlety y assessment
src/prepare_mias.py       recortes de mini-MIAS para la prueba externa
src/evaluate_external.py  evalúa en un conjunto externo con el umbral de validación
src/medir_recursos.py     memoria de GPU/CPU y tiempos de cada arquitectura
src/common.py, models.py, metrics.py
deploy/                   exportación a ONNX y armado del Space de Hugging Face
web/                      versión que corre en el navegador (respaldo en GitHub Pages)
notebooks/Colab_Mamografias.ipynb   flujo completo en Google Colab
docs/                     informe, hipótesis, bitácora, registro de IA, tabla de etiquetas, guía de Colab, guía del día 7
```

## Instalación
```bash
pip install -r requirements.txt
```
Se recomienda Google Colab con GPU (Entorno de ejecución → Cambiar tipo → T4 GPU). Paso a paso en `docs/guia_colab.md`.

## Datos
**CBIS-DDSM** (TCIA): https://www.cancerimagingarchive.net/collection/cbis-ddsm/ — licencia CC BY 3.0 (verificar en la página).
Dos opciones:
1. **Versión JPEG en Kaggle** (más liviana), derivada de la colección de TCIA: `awsaf49/cbis-ddsm-breast-cancer-image-dataset`. → `--source kaggle`
2. **DICOM originales de TCIA** descargados con NBIA Data Retriever (solo “ROI and Cropped Images” + CSV de descripción). → `--source tcia`

Las imágenes **no** se suben al repositorio (`data/` está en `.gitignore`).

## Flujo de ejecución
```bash
# 1. Preparar datos y partición por paciente
python src/prepare_data.py --source kaggle --root /ruta/dataset --lesion both --bwc benign --seed 42
python src/check_leakage.py
python src/eda.py

# 2. Entrenar
python src/train.py --name R0_smallcnn      --model smallcnn --epochs 25 --lr 1e-3 --aug
python src/train.py --name R1_resnet18_aug  --model resnet18 --epochs 15 --lr 1e-4 --aug
python src/train.py --name R2_resnet18_noaug --model resnet18 --epochs 15 --lr 1e-4      # experimento controlado

# 3. Evaluar en VALIDACIÓN (umbral elegido aquí)
python src/evaluate.py --run runs/R0_smallcnn --split val --strategy sens --target-sens 0.90
python src/evaluate.py --run runs/R1_resnet18_aug --split val --strategy sens --target-sens 0.90
python src/evaluate.py --run runs/R2_resnet18_noaug --split val --strategy sens --target-sens 0.90
python src/compare_runs.py --split val

# 4. SOLO AL FINAL: prueba reservada con el modelo y umbral elegidos
python src/evaluate.py --run runs/<ganador> --split test --strategy sens --target-sens 0.90

# 5. Interfaz
python app.py --run runs/<ganador> --share
```

### Indicaciones del día 7: más épocas, más datos y prueba externa
```bash
# Más épocas (validación elige la mejor) y modelo final con train+val
python src/train.py --name R3_resnet18_aug_30ep --model resnet18 --epochs 30 --patience 7 --lr 1e-4 --aug
python src/evaluate.py --run runs/R3_resnet18_aug_30ep --split val --strategy sens --target-sens 0.90
python src/train.py --name R4_final_trainval --model resnet18 --epochs 10 --lr 1e-4 --aug --incluir-val

# Prueba externa con mini-MIAS (Kaggle: kmader/mias-mammography), umbral fijado en validación de CBIS-DDSM
python src/prepare_mias.py --root /ruta/mias --out data/externo/mias
python src/evaluate_external.py --run runs/R1_resnet18_aug --csv data/externo/mias/lesiones_mias.csv
python src/evaluate_external.py --run runs/R4_final_trainval --csv data/externo/mias/lesiones_mias.csv \
       --threshold-from runs/R3_resnet18_aug_30ep/threshold.json
```

### Costo computacional
```bash
cd src && python medir_recursos.py --modelos smallcnn resnet18 efficientnet_b0 --out ../reports/recursos
```
Mide memoria de GPU y tiempo por paso al entrenar (si hay GPU) y latencia y memoria al predecir en CPU. Resultados en `reports/recursos/`.

## App publicada (24/7)
- **Hugging Face Spaces:** https://huggingface.co/spaces/JostGreen/mamografias-g8-web
- **GitHub Pages (respaldo):** https://joseg-sotomendez.github.io/mamografias_g8/

Las dos ejecutan el modelo R1 exportado a ONNX dentro del navegador (mismo preprocesamiento, mismo umbral de validación). La app de Gradio se corre con `python app.py --run runs/R1_resnet18_aug` y está empaquetada para un Space de Gradio en `deploy/` (Hugging Face cobra ese tipo de Space a cuentas nuevas).

Cómo se arma (carpeta `deploy/`):
```bash
python deploy/exportar_onnx.py --run runs/R1_resnet18_aug --out web/modelo_r1.onnx --imagenes <carpeta de prueba>
# Versión web: index.html + modelo_r1.onnx a la rama gh-pages y a un Space estático de Hugging Face
# App de Gradio (opcional, requiere un Space de Gradio):
python deploy/preparar_space.py --run runs/R1_resnet18_aug --ejemplos <carpeta de prueba> --out build/space
hf upload <usuario>/mamografias-g8 build/space . --repo-type=space
```
`exportar_onnx.py` falla si la puntuación de ONNX difiere de la de PyTorch en más de 1e-4. El flujo `.github/workflows/vigilante_app.yml` revisa cada 6 horas que las dos páginas y el modelo respondan (y la app de Gradio, si se define la variable `HF_SPACE`); si algo falla, GitHub avisa por correo.

## Reproducibilidad
- Semilla fija (`--seed 42`) en partición y entrenamiento; `config.json` de cada corrida guarda todos los parámetros, versión de PyTorch, GPU, commit de git y tiempos.
- `data/splits/lesiones.csv` contiene la partición exacta usada (identificadores de lesión y paciente); se puede versionar sin incluir imágenes.
- Cada uso del conjunto de prueba queda registrado en `runs/<name>/registro_uso_test.txt`.

## Uso de IA generativa
Declarado en `docs/registro_ia.md`.
