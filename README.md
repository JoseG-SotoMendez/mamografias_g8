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
src/common.py, models.py, metrics.py
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

## Reproducibilidad
- Semilla fija (`--seed 42`) en partición y entrenamiento; `config.json` de cada corrida guarda todos los parámetros, versión de PyTorch, GPU, commit de git y tiempos.
- `data/splits/lesiones.csv` contiene la partición exacta usada (identificadores de lesión y paciente); se puede versionar sin incluir imágenes.
- Cada uso del conjunto de prueba queda registrado en `runs/<name>/registro_uso_test.txt`.

## Uso de IA generativa
Declarado en `docs/registro_ia.md`.
