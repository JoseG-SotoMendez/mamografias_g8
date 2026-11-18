# Clasificación de lesiones mamográficas benignas y malignas mediante aprendizaje profundo
Grupo 8 — José Gil Soto Méndez, Andrés Alfonso Luna Moreno, Miguel Francisco Guerra Negrete
Aprendizaje Computacional – Ingeniería de Sistemas · Noviembre 2026

> **Aviso:** prototipo académico; no apto para diagnóstico ni decisiones clínicas.

<!--
REGLAS DEL INFORME (borrar este comentario en la versión final)
- Máximo 10 páginas sin anexos.
- Cada sección indica quién la redacta. Cada uno escribe la suya con sus palabras; antes de editar: git pull; al terminar: commit + push.
- Ninguna cifra se escribe a mano: todo sale de runs/ y reports/. Lo que falte queda como <<PENDIENTE: ...>>.
-->

## 1. Comprensión del problema  *(redacta: José)*

El objetivo es construir un prototipo que, dado el **recorte de una lesión** ya localizada en una mamografía, estime si su patología de referencia es benigna o maligna. El usuario previsto es académico: sirve para estudiar cómo se comporta un clasificador de imágenes médicas y qué decisiones lo vuelven confiable o engañoso. No es una herramienta clínica y no se diseñó para serlo.

**Unidad de análisis: la lesión.** Cada fila de los CSV de CBIS-DDSM describe una lesión vista en una proyección (CC o MLO), con su recorte (ROI) y su patología verificada. Trabajar con la imagen completa obligaría al modelo a encontrar la lesión además de clasificarla, y con los datos que tenemos eso es otro proyecto. Trabajar por mama o por examen mezclaría varias lesiones con etiquetas distintas. La lesión es la unidad donde la etiqueta tiene un significado claro.

**Qué significa la etiqueta.** La columna `pathology` dice si la lesión resultó benigna o maligna según la verificación del DDSM. Eso es la patología de **una lesión ya anotada**, no el diagnóstico de la paciente: una paciente puede tener una lesión benigna en un recorte y cáncer en otro lugar que nadie marcó. El mapeo completo está en `docs/tabla_etiquetas.md`.

**Costo de los errores.** Un falso negativo es una lesión maligna que el sistema llama benigna: en un uso real retrasaría un diagnóstico, y eso puede costar tratamiento y pronóstico. Un falso positivo es una lesión benigna llamada maligna: cuesta un estudio adicional, una biopsia y angustia, pero se corrige. Como no pesan lo mismo, el umbral de decisión no se deja en 0,5 por defecto: se elige en validación como el más alto que alcanza **sensibilidad para malignas ≥ 0,90**, aceptando más falsos positivos a cambio.

**Hipótesis (escritas el 2026-11-18, antes de entrenar; ver `docs/hipotesis.md`).**
- **H1:** con ResNet18 preentrenada, activar los aumentos (R1) frente a no activarlos (R2) reduce la brecha entre pérdida de validación y de entrenamiento y sube el ROC-AUC de validación en al menos 0,02.
- **H2:** el umbral elegido para sensibilidad ≥ 0,90 queda por debajo de 0,5, baja la especificidad de validación al rango 0,35–0,60 y, en prueba, la sensibilidad queda entre 0,80 y 0,90.

**Limitaciones de partida.** Mamografías de película escaneada de los años 90 (no digitales actuales), una población estadounidense, casos seleccionados porque tenían lesiones, y un tamaño de muestra de unos pocos miles de recortes. Todo lo que concluyamos vale para este conjunto, no para una paciente colombiana en 2026.

## 2. Datos  *(redacta: Andrés)*
- Fuente: CBIS-DDSM (TCIA), versión JPEG de Kaggle `awsaf49/cbis-ddsm-breast-cancer-image-dataset`; licencia según la página de TCIA <<PENDIENTE: verificar licencia exacta>>.
- Cantidades: <<PENDIENTE: pacientes, lesiones, benignas/malignas, masas/calcificaciones, vistas CC/MLO — de `data/splits/resumen.json` y `reports/eda/eda_resumen.json`>>.
- Tabla de mapeo de etiquetas (copiar de `docs/tabla_etiquetas.md`).
- EDA: balance (`balance_clases.png`), tipo vs clase (`tipo_vs_clase.png`), tamaños (`tamanos.png`), duplicados y exclusiones (`descartadas.csv`). <<PENDIENTE>>
- Posibles sesgos: película escaneada, época y población del DDSM, selección de casos con lesión. <<PENDIENTE: en palabras de Andrés>>

## 3. Preparación  *(redacta: Andrés)*
- División por paciente con `StratifiedGroupKFold` (semilla 42, ≈70/15/15) y evidencia de 0 pacientes compartidas (`check_leakage.py`). <<PENDIENTE: salida real>>
- Lectura JPEG/DICOM, conversión a 3 canales, redimensionamiento a 224 px, normalización con media y desviación de ImageNet (lo que esperan los pesos preentrenados).
- Aumentos solo en entrenamiento: volteo horizontal y vertical, rotación ±10°, brillo/contraste ±10 %. Validación y prueba sin alterar. <<PENDIENTE: por qué son razonables en mamografía, en palabras de Andrés>>

## 4. Modelado  *(redacta: José)*

Se entrenaron tres corridas con la misma partición, semilla 42, pérdida `BCEWithLogitsLoss`, optimizador AdamW (weight decay 1e-4), tasa de aprendizaje con decaimiento coseno, lotes de 32, imágenes de 224 px y parada temprana con paciencia 5 según el ROC-AUC de validación (se guardan los pesos de la mejor época).

| Corrida | Modelo | Preentrenado | Aumentos | lr | Épocas máx. (mejor) | Batch | Optimizador | Semilla | Tiempo |
|---|---|---|---|---|---|---|---|---|---|
| R0 | CNN pequeña (4 bloques conv, ~0,1 M parámetros) | no | sí | 1e-3 | 25 (<<PENDIENTE>>) | 32 | AdamW | 42 | <<PENDIENTE>> |
| R1 | ResNet18 (~11 M parámetros) | ImageNet | sí | 1e-4 | 15 (<<PENDIENTE>>) | 32 | AdamW | 42 | <<PENDIENTE>> |
| R2 | ResNet18 | ImageNet | **no** | 1e-4 | 15 (<<PENDIENTE>>) | 32 | AdamW | 42 | <<PENDIENTE>> |

**R0 como línea base.** Una red pequeña entrenada desde cero marca el piso: si la transferencia de aprendizaje no la supera con claridad, algo anda mal en el pipeline o los datos no alcanzan.

**Experimento controlado (R1 vs R2).** La única decisión que cambia es la bandera `--aug`. Todo lo demás (modelo, pesos iniciales, datos, semilla, lr, épocas, batch) se mantiene, así que cualquier diferencia se puede atribuir a los aumentos y no a una mezcla de cambios. Lo esperado está en H1.

**Hardware.** <<PENDIENTE: GPU, versión de PyTorch y tiempos, de `config.json`>>

## 5. Evaluación  *(redacta: José, salvo 5.4 que redacta Miguel)*

### 5.1 Umbral
El umbral se elige **solo en validación**: el más alto que logra sensibilidad maligna ≥ 0,90 (`evaluate.py --strategy sens --target-sens 0.90`) y se guarda en `threshold.json`. La salida del modelo es una **puntuación** (sigmoide de un logit), no una probabilidad calibrada: un 0,7 no significa "70 % de probabilidad de cáncer", solo que el modelo la ordena por encima de un 0,6. Por eso el umbral se fija mirando el compromiso sensibilidad/especificidad y no leyendo la puntuación como riesgo. <<PENDIENTE: valor del umbral elegido>>

### 5.2 Comparación en validación
<<PENDIENTE: pegar `reports/comparacion_val.md` y escoger el ganador con su justificación (ROC-AUC y sensibilidad/especificidad al umbral elegido)>>

### 5.3 Resultado final en prueba (una sola vez)
<<PENDIENTE: matriz de confusión, sensibilidad maligna, especificidad, precisión, F1, ROC-AUC del ganador en prueba (`eval_test/metricas.json`) y la línea de `registro_uso_test.txt`>>

Efecto del umbral: <<PENDIENTE: `tabla_umbrales.csv` / `umbral_vs_metricas.png` — qué pasa con FN y FP al moverlo>>. Curvas y sobreajuste: <<PENDIENTE: `curvas.png`>>. ¿Se confirmaron H1 y H2? <<PENDIENTE>>

### 5.4 Análisis de errores  *(redacta: Miguel)*
<<PENDIENTE: al menos 4 falsos negativos y 4 falsos positivos reales con puntuación, tipo de lesión, `subtlety`, `assessment` e hipótesis de la causa; ver `docs/analisis_errores.md`>>

## 6. Interfaz  *(redacta: Miguel)*
<<PENDIENTE: captura de Gradio con el aviso visible; formatos admitidos (PNG, JPG, DICOM); qué pasa con un archivo inválido, una imagen muy pequeña y una mamografía completa>>

## 7. Discusión y límites  *(redactan: Andrés los límites de datos, Miguel los límites clínicos y de la interfaz)*
<<PENDIENTE: tamaño muestral, sesgo de selección, generalización a mamografía digital y a población colombiana, límites de interpretación clínica, qué harían con más tiempo>>

## 8. Conclusiones  *(redacta: José, revisan todos)*
<<PENDIENTE: tras los resultados>>

## Anexos
- `docs/bitacora.md`, `docs/registro_ia.md`, `docs/hipotesis.md`, `docs/tabla_etiquetas.md`, `docs/analisis_errores.md`
- Evidencia del control del día 7 <<PENDIENTE>>
- Comandos exactos: `README.md` y `docs/guia_colab.md`
