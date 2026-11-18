# Guía para correr el proyecto en Google Colab

Paso a paso para `notebooks/Colab_Mamografias.ipynb`. Los tiempos son **estimaciones** para una GPU T4 gratuita; anoten los reales en la bitácora.

## 0. Antes de empezar (5 min)

1. Entrar a <https://colab.research.google.com> con su cuenta de Google → **Archivo → Abrir cuaderno → GitHub** → pegar `JoseG-SotoMendez/mamografias_g8` → abrir `notebooks/Colab_Mamografias.ipynb`.
2. **Entorno de ejecución → Cambiar tipo de entorno → T4 GPU → Guardar.** Sin esto, entrenar tarda horas.
3. Tener unos **2 GB libres en Google Drive** para `runs/` (los pesos de ResNet18 pesan ~45 MB por corrida).

## 1. Orden de las celdas

| # | Celda | Qué hace | Tiempo estimado | Qué mirar |
|---|---|---|---|---|
| 1 | `!nvidia-smi` | Confirma la GPU | segundos | Debe decir **Tesla T4**. Si dice "command not found", no activaron la GPU (paso 0.2). |
| 3 | `git clone` | Baja el código e imprime el commit | < 1 min | El hash del commit (anotarlo). |
| 5 | Drive | Monta Drive y restaura corridas previas | 1 min | Autorizar con **su** cuenta en la ventana emergente. |
| 7 | `kagglehub` | Descarga CBIS-DDSM JPEG (≈6 GB según la página de Kaggle; verificar) | 3–10 min | La ruta impresa y que aparezcan `csv` y `jpeg`. |
| 9 | `prepare_data.py` + `check_leakage.py` | Une CSV con imágenes, quita duplicados, parte por paciente | 3–8 min (el MD5 lee todas las imágenes) | `lesiones con imagen`, el JSON por partición y **`RESULTADO: SIN FUGA ✅`**. |
| 10 | Copiar partición a Drive | Guarda `lesiones.csv` y `resumen.json` | segundos | |
| 11 | `eda.py` | Figuras de balance, tipos, tamaños, ejemplos | 1–3 min | Proporción de malignas, tamaños de los recortes. |
| 13 | **Prueba rápida** | 1 época con 200 lesiones | 1–2 min | Que imprima una línea `epoca=1 | ...` y `Listo.` Si esto falla, **no** sigan con R0–R2. |
| 14 | R0 (CNN pequeña) | 25 épocas máx., parada temprana | 10–25 min | `Mejor val ROC-AUC` y época. |
| 15 | R1 (ResNet18 + aumentos) | 15 épocas máx. | 10–25 min | Igual. |
| 16 | R2 (ResNet18 sin aumentos) | 15 épocas máx. | 10–25 min | Igual; comparar con R1 (hipótesis H1). |
| 17 | Curvas | Muestra `curvas.png` de las tres | segundos | ¿Se separan `train` y `val`? (sobreajuste) |
| 19 | Evaluación en **validación** | Elige el umbral (sens ≥ 0,90) y compara | 2–4 min | Tabla de `compare_runs.py`. |
| 20 | Figuras del ganador | Matriz, ROC, umbral vs métricas, FN/FP | segundos | Cambiar `R` por el ganador según validación. |
| 22 | **Prueba (una sola vez)** | Está comentada a propósito | 1 min | Solo cuando el grupo haya elegido modelo y umbral. Descomentar, correr **una vez**, volver a comentar. |
| 24 | Gradio | Interfaz con enlace público temporal | queda corriendo | Abrir el enlace `*.gradio.live`, subir un recorte y tomar capturas con el aviso visible. Para seguir usando el cuaderno, detener la celda (■). |
| 26 | Guardar en Drive | Copia `runs/` y `reports/` | < 1 min | Correrla **después de cada entrenamiento**. |

**Regla de oro:** el conjunto de prueba no se mira para elegir nada. Todo lo que se decide (modelo, umbral, transformación) se decide con validación.

## 2. Qué copiarle de vuelta a José / a Claude

Pegar en el chat (texto, no capturas) o subir la carpeta de Drive:

1. Salida completa de la celda 9 (`prepare_data` + `check_leakage`) y `data/splits/resumen.json`.
2. `reports/eda/eda_resumen.json`.
3. De cada corrida (`runs/R0_smallcnn`, `runs/R1_resnet18_aug`, `runs/R2_resnet18_noaug`): `config.json` y `log.csv`.
4. Salida de la celda 19 y `reports/comparacion_val.md`.
5. Del ganador: `eval_val/metricas.json`, `eval_val/tabla_umbrales.csv`, `threshold.json`, `eval_val/falsos_negativos.csv` y `falsos_positivos.csv` (sin las imágenes).
6. Cuando se corra la prueba: `eval_test/metricas.json` y `registro_uso_test.txt`.
7. Los `.png` de `reports/eda/` (balance, tipos, tamaños) y `runs/<ganador>/eval_*/` van al informe, **no** al repositorio si muestran recortes de pacientes.

Para el repositorio: descargar `data/splits/lesiones.csv` y `resumen.json` desde Drive (`MyDrive/mamografias_g8/splits/`) y ponerlos en `data/splits/` del repo local.

## 3. Errores típicos

| Síntoma | Causa probable | Arreglo |
|---|---|---|
| `nvidia-smi: command not found` o `device: cpu` en el entrenamiento | No hay GPU | Paso 0.2 y **Entorno de ejecución → Reiniciar**. Volver a correr desde la celda 1. |
| `kagglehub` da 401/403 | Kaggle pide autenticación | En Kaggle: Settings → API → *Create New Token*. En Colab: ícono de llave (🔑 Secretos) → agregar `KAGGLE_USERNAME` y `KAGGLE_KEY` y activar "acceso del cuaderno". **Nunca** pegar el token en una celda ni en el chat. |
| `No se encontraron CSV '*case_description*.csv'` | `DATA_ROOT` apunta a otra carpeta | `!ls -R {DATA_ROOT} | head` y pasar `--root` a la carpeta que contiene `csv/`. |
| Muchas `imagen recortada no encontrada` | Cambió la estructura del dataset en Kaggle | Revisar `data/splits/descartadas.csv`; si son más del ~5 %, avisar antes de entrenar (es un problema real para la bitácora). |
| `CUDA out of memory` | Lote muy grande | Añadir `--batch-size 16` al comando de entrenamiento y anotarlo en `config.json`/bitácora. |
| La sesión se desconecta a mitad de un entrenamiento | Límite de Colab gratis | Volver a correr celdas 1→9 (la partición sale igual por la semilla 42) y la 5 restaura lo guardado en Drive. Repetir solo la corrida que se cortó. |
| `FileNotFoundError: runs/.../best.pt` en evaluación o Gradio | Se reinició la sesión y no se restauró Drive | Celda 5. |
| Gradio no da enlace público | Fallo temporal del servicio de túneles | Volver a correr la celda; la interfaz igual queda en la URL local de la salida. |

## 4. Para el control del día 7 (cada integrante)

- Abrir su propia copia del cuaderno **antes de las 11:00**, con GPU, y correr celdas 1→11 (deja datos, partición y EDA listos para la demo en vivo).
- Para la inferencia en vivo necesitan pesos entrenados: José comparte (solo lectura) su carpeta de Drive `mamografias_g8`. Cada uno, en Drive → *Compartido conmigo* → clic derecho sobre la carpeta → **Organizar → Agregar acceso directo → Mi unidad**. Así queda en `MyDrive/mamografias_g8` y la celda 5 restaura `runs/` con los pesos.
- Comandos de la demo: ver `docs/control_dia7.md`.
