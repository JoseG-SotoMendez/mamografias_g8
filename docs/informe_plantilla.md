# Clasificación de lesiones mamográficas benignas y malignas mediante aprendizaje profundo
Grupo 8 — José Gil Soto Méndez, Andrés Alfonso Luna Moreno, Miguel Francisco Guerra Negrete
Aprendizaje Computacional – Ingeniería de Sistemas · Noviembre 2026

> Máximo 10 páginas sin anexos. Todo lo marcado [COMPLETAR] sale de **sus** ejecuciones (archivos en `runs/` y `reports/`). No inventar cifras.

**Aviso:** prototipo académico; no apto para diagnóstico ni decisiones clínicas.

## 1. Comprensión del problema (≈1 pág.)
- Usuario previsto y contexto educativo.
- Unidad de análisis: lesión (recorte ROI, vista CC o MLO). Por qué no imagen completa, mama o examen.
- Significado de la etiqueta (ver `docs/tabla_etiquetas.md`): patología de la lesión ≠ diagnóstico de la paciente.
- Costo de errores: falso negativo (maligna clasificada benigna) vs falso positivo. Cuál es peor y cómo eso orienta la elección del umbral.
- Hipótesis H1 y H2 con fecha (de `docs/hipotesis.md`).
- Limitaciones de partida.

## 2. Datos (≈1,5 pág.)
- Fuente, versión, licencia, origen de las etiquetas; [COMPLETAR: Kaggle JPEG o TCIA DICOM].
- Cantidades: pacientes, lesiones, benignas/malignas, masas/calcificaciones, vistas CC/MLO (`data/splits/resumen.json`, `reports/eda/eda_resumen.json`).
- Tabla de mapeo de etiquetas.
- EDA: balance (`balance_clases.png`), tamaños (`tamanos.png`), ejemplos, duplicados, exclusiones (`descartadas.csv`).
- Posibles sesgos: película escaneada (no digital), población y época del DDSM, selección de casos solo con lesión.

## 3. Preparación (≈1 pág.)
- División por paciente con `StratifiedGroupKFold`, semilla, proporciones; evidencia de 0 pacientes compartidas (`check_leakage.py`).
- Lectura DICOM/JPG, conversión a 3 canales, redimensionamiento a [COMPLETAR] px, normalización ImageNet y por qué.
- Aumentos solo en entrenamiento (cuáles y por qué son razonables en mamografía); prueba sin alterar.

## 4. Modelado (≈1,5 pág.)
| Corrida | Modelo | Preentrenado | Aumentos | lr | Épocas (mejor) | Batch | Optimizador | Semilla | Tiempo |
|---|---|---|---|---|---|---|---|---|---|
| R0 | CNN pequeña | no | | | | | AdamW | 42 | |
| R1 | ResNet18 | ImageNet | | | | | AdamW | 42 | |
| R2 | ResNet18 | ImageNet | | | | | AdamW | 42 | |

- Experimento controlado: qué única decisión cambia entre R1 y R2, qué se mantuvo constante, qué se esperaba.
- Hardware (de `config.json`).

## 5. Evaluación (≈2,5 pág.)
- Selección del umbral **en validación** (estrategia y valor; `threshold.json`). Diferencia entre puntuación del modelo y probabilidad calibrada.
- Tabla comparativa en validación (`reports/comparacion_val.md`).
- Resultado final en prueba **una sola vez** con el modelo elegido: matriz de confusión, sensibilidad maligna, especificidad, precisión, F1, ROC-AUC (`eval_test/`).
- Efecto del umbral (`tabla_umbrales.csv`, `umbral_vs_metricas.png`).
- Curvas de entrenamiento y sobreajuste (`curvas.png`).
- Análisis de errores: al menos [N] falsos negativos y falsos positivos reales con imagen, puntuación, tipo de lesión, `subtlety`/`assessment`, e hipótesis de la causa.
- ¿Se confirmaron H1 y H2?

## 6. Interfaz (≈0,5 pág.)
- Captura de Gradio con el aviso visible; formatos admitidos; manejo de imágenes inválidas y de mamografías completas.

## 7. Discusión y límites (≈1 pág.)
- Tamaño muestral, sesgo de selección, generalización a mamografía digital y a población colombiana, límites de interpretación clínica.
- Qué harían con más tiempo.

## 8. Conclusiones

## Anexos
- Bitácora, registro de IA, evidencia del control del día 7, comandos exactos.
