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
| Resultado (2026-11-18, Colab T4, commit `a73c475`) | ROC-AUC de validación: **R1 = 0,819** (mejor época 12) vs **R2 = 0,782** (mejor época 1); diferencia +0,037. R0 = 0,694. Brecha `val_loss − train_loss` en la mejor época: R1 = 0,617 − 0,264 = **0,353**; R2 = 0,567 − 0,595 = **−0,028**. A igual época (la 6, última de R2): R1 = 0,546 − 0,433 = **0,114**; R2 = 0,906 − 0,028 = **0,878**. R2 llevó la pérdida de entrenamiento a 0,028 en 6 épocas y la parada temprana lo cortó; R1 llegó a 0,24 en 15. Fuente: `runs/R*/log.csv`, `config.json`, `reports/comparacion_val.md`. |
| ¿Se confirma? / Interpretación | **Parcialmente.** (b) Sí: +0,037 de AUC, por encima de lo predicho; con 521 lesiones de validación es una diferencia modesta y no la vendemos como definitiva. (c) Sí: R2 tuvo su mejor época en la 1 y R1 en la 12. (a) **No, tal como la escribimos**: medida en la mejor época, la brecha de R2 sale menor porque su mejor época es la primera, antes de empezar a memorizar. Comparada en la misma época, el efecto esperado aparece claro (0,114 vs 0,878). Lección: la métrica de sobreajuste estaba mal elegida; había que compararla a igual época o en toda la curva, no en la mejor época de cada modelo. |

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
| Resultado (validación, 2026-11-18) | Ganador: R1. Umbral elegido en validación: **0,0664** (exacto 0,06638). A 0,5: sensibilidad **0,604**, especificidad **0,853** (88 FN, 44 FP). Al umbral elegido: sensibilidad **0,901**, especificidad **0,518** (22 FN, 144 FP). Prueba: <<PENDIENTE: evaluación única en prueba>>. Fuente: `runs/R1_resnet18_aug/eval_val/metricas.json`, `tabla_umbrales.csv`, `threshold.json`. |
| ¿Se confirma? / Interpretación | (a) Sí, y de sobra: 0,066 está muy por debajo de 0,5. Eso dice que las puntuaciones de R1 se concentran cerca de 0: no son probabilidades calibradas. (b) Sí: pasar de 0,5 al umbral elegido rescata 66 malignas (de 88 FN a 22) a cambio de 100 falsos positivos más (de 44 a 144); la especificidad cae a 0,518, dentro del rango predicho. (c) <<PENDIENTE: prueba>> |

**Por qué lo creemos:** el umbral es como la sensibilidad de un detector de humo. Si lo pones muy sensible no se te escapa ningún incendio, pero también suena cada vez que alguien hace tostadas. Aquí preferimos las tostadas: un falso negativo es una lesión maligna que el sistema deja pasar como benigna (el peor error, porque retrasa el diagnóstico), mientras que un falso positivo cuesta un estudio o una biopsia más. Y lo de la prueba: cualquier valor que se ajusta mirando unos datos queda un poco "a la medida" de esos datos, así que en datos nuevos rinde un poco peor.
