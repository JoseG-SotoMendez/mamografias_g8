# Preparación del control del día 7 (8–10 min por integrante, individual)

**Lo que van a pedir** (según el enunciado) y cómo hacerlo con este repositorio.
Cada uno debe practicarlo **solo** y entender cada línea; el profesor pedirá cambios no ensayados.

## 1. Abrir el repositorio y explicar dos decisiones propias
- Mostrar `git log --oneline --date=short --format="%ad %an %s"` (commits de días distintos, de cada integrante).
- Mostrar la entrada correspondiente en `docs/bitacora.md`.

## 2. Leer imágenes y mostrar la distribución por clase
```bash
python src/check_leakage.py                 # clases por partición + pacientes
python src/eda.py                           # figuras en reports/eda/
```
Saber explicar: ¿cuántas lesiones, cuántas pacientes, qué proporción es maligna?

## 3. Demostrar que no se mezclan pacientes
```bash
python src/check_leakage.py --paciente P_00001
```
Saber explicar **por qué** importa (CC y MLO de la misma lesión son casi la misma información → si quedan en train y test, el resultado se infla) y **cómo** lo garantiza `StratifiedGroupKFold` en `src/prepare_data.py` (función `patient_split`).

## 4. Entrenamiento preliminar e interpretar un resultado
- Abrir `runs/R1_.../curvas.png` y `log.csv`: ¿hay sobreajuste? ¿en qué época fue el mejor modelo?
- Inferencia de una imagen:
```bash
python src/inference.py --run runs/R1_resnet18_aug --image <ruta de un recorte>
```

## 5. Modificaciones en vivo que pueden pedir
| Pedido | Cómo |
|---|---|
| Cambiar el umbral | `python src/evaluate.py --run runs/R1_resnet18_aug --split val --threshold 0.3` y comparar la matriz de confusión |
| Elegir umbral para otra sensibilidad | `--strategy sens --target-sens 0.95` |
| Localizar un falso negativo | abrir `runs/<run>/eval_val/falsos_negativos.csv` / `.png`; mostrar la imagen y su puntuación |
| Revisar identificadores | `check_leakage.py --paciente P_xxxxx`; buscar en `data/splits/lesiones.csv` |
| Alterar una transformación | editar `build_transform` en `src/common.py` (p. ej. quitar `RandomVerticalFlip` o cambiar `RandomRotation(10)`) y relanzar un entrenamiento corto: `python src/train.py --name prueba --model resnet18 --epochs 1 --max-train 200 --aug` |
| Cambiar tamaño de imagen | `--img-size 256` en `train.py` |

## 6. Inconveniente real y su solución
Tener al menos uno documentado en la bitácora con fecha (p. ej. rutas que no coincidían, DICOM que era máscara y no recorte, desbalance, memoria GPU).

## Preguntas típicas para repasar
- ¿Qué es la unidad de análisis y por qué es la lesión?
- ¿Qué significa `BENIGN_WITHOUT_CALLBACK` y qué hicieron con ella?
- ¿Por qué el umbral no se elige en prueba?
- ¿Por qué la puntuación del modelo no es una probabilidad clínica?
- ¿Qué es más grave aquí, un FN o un FP? ¿Cómo lo reflejaron?
- ¿Qué hace la transferencia de aprendizaje y por qué se normaliza con media/desviación de ImageNet?
