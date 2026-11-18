# Hipótesis (escritas ANTES de entrenar)

**Fecha de redacción:** 2026-11-18 (antes de la primera corrida en Colab; la evidencia es la fecha del commit que crea este archivo).
**Redacta:** José Gil Soto Méndez · revisan Andrés Luna y Miguel Guerra.
**Decisiones de partida que asumen estas hipótesis:** masas + calcificaciones (`--lesion both`), `BENIGN_WITHOUT_CALLBACK` como benigna (`--bwc benign`), split por paciente 70/15/15 con semilla 42, imágenes de 224 px.

> Las predicciones numéricas son apuestas, no resultados. Si fallan, se dice que fallaron; para eso las escribimos antes.

---

## H1 — Los aumentos de datos reducen el sobreajuste de ResNet18 (experimento controlado R1 vs R2)

| Campo | Contenido |
|---|---|
| Fecha | 2026-11-18 |
| Autor | José Gil Soto Méndez |
| Hipótesis | Con el mismo modelo (ResNet18 preentrenada en ImageNet), mismos datos, semilla, lr = 1e-4, 15 épocas y batch 32, **activar los aumentos moderados** (volteos, rotación ±10°, brillo/contraste ±10 %) reduce el sobreajuste y mejora la capacidad de separar benignas de malignas en validación. |
| Predicción medible | (a) En la mejor época, la brecha `val_loss − train_loss` será **menor en R1** (con aumentos) que en R2 (sin aumentos). (b) El ROC-AUC de validación de R1 será **al menos 0,02 mayor** que el de R2. (c) R2 alcanzará su mejor época **antes** que R1 (memoriza más rápido). |
| Cómo se pone a prueba | `train.py` con R1 (`--aug`) y R2 (sin `--aug`); lo único que cambia es la bandera de aumentos. Se comparan `log.csv`, `curvas.png` y `config.json` de ambas corridas y `reports/comparacion_val.md`. La CNN pequeña (R0) sirve de piso: esperamos que R1 y R2 la superen en ROC-AUC. |
| Resultado | <<PENDIENTE: ROC-AUC val de R1 y R2, brecha de pérdidas en la mejor época y mejor época de cada una (salen de Colab)>> |
| ¿Se confirma? / Interpretación | <<PENDIENTE: completar con los números reales; si la diferencia es menor que 0,02, decir que con ~500 lesiones de validación no se distingue del ruido>> |

**Por qué lo creemos:** el conjunto de entrenamiento tiene unos pocos miles de recortes, poco para once millones de parámetros. Sin aumentos, la red ve exactamente las mismas imágenes en cada época y termina aprendiéndoselas de memoria, como quien estudia el examen del año pasado y no la materia.

---

## H2 — Elegir el umbral por sensibilidad cuesta especificidad, y en prueba la sensibilidad baja un poco

| Campo | Contenido |
|---|---|
| Fecha | 2026-11-18 |
| Autor | José Gil Soto Méndez |
| Hipótesis | Si el umbral se elige en **validación** como el más alto que logra sensibilidad maligna ≥ 0,90 (en vez del 0,5 por defecto), se recuperan malignas que a 0,5 se escapaban, pero a costa de muchos más falsos positivos. Además, como el umbral se ajustó sobre validación, en la prueba reservada la sensibilidad será **algo menor** que 0,90. |
| Predicción medible | (a) Con el modelo ganador, el umbral elegido quedará **por debajo de 0,5**. (b) En validación, frente a 0,5: la sensibilidad subirá a ≥ 0,90 (por construcción) y la especificidad **caerá al rango 0,35–0,60**. (c) En prueba, con ese mismo umbral, la sensibilidad quedará **entre 0,80 y 0,90**. |
| Cómo se pone a prueba | `evaluate.py --split val --strategy sens --target-sens 0.90` (umbral oficial) contra `evaluate.py --split val --threshold 0.5` (exploración, no toca `threshold.json`). Se comparan `metricas.json` y `tabla_umbrales.csv`. La parte (c) se verifica **una sola vez** con `--split test` al final. |
| Resultado | <<PENDIENTE: umbral elegido, sensibilidad/especificidad a 0,5 y al umbral elegido en validación; sensibilidad en prueba>> |
| ¿Se confirma? / Interpretación | <<PENDIENTE>> |

**Por qué lo creemos:** el umbral es como la sensibilidad de un detector de humo. Si lo pones muy sensible no se te escapa ningún incendio, pero también suena cada vez que alguien hace tostadas. Aquí preferimos las tostadas: un falso negativo es una lesión maligna que el sistema deja pasar como benigna (el peor error, porque retrasa el diagnóstico), mientras que un falso positivo cuesta un estudio o una biopsia más. Y lo de la prueba: cualquier valor que se ajusta mirando unos datos queda un poco "a la medida" de esos datos, así que en datos nuevos rinde un poco peor.
