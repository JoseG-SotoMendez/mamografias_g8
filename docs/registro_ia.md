# Registro de uso de IA generativa

El enunciado exige registrar: pregunta, respuesta, contraste realizado y decisión final.
Cada integrante registra **su propio** uso de IA en su sección. La regla del grupo: si la IA produjo algo que entregamos, aquí queda escrito, aunque después lo hayamos reescrito.

**Responsabilidad del equipo (según el enunciado):** hipótesis, selección y comprensión de datos, decisiones experimentales, ejecución, interpretación y defensa.

## José Gil Soto Méndez

| Fecha | Herramienta | Pregunta / solicitud | Qué respondió / generó | Contraste realizado | Decisión final |
|---|---|---|---|---|---|
| 2026-11-17 | Claude (Anthropic) | Resumen de los requisitos del proyecto y plan por días | Resumen del enunciado y un cronograma sugerido | Lo comparé con el PDF del enunciado (fechas del control y entregables) | Usamos el cronograma como base; el plan final está en el material de estudio del grupo |
| 2026-11-17 | Claude (Anthropic) | Código base del repositorio (`src/*.py`, `app.py`, notebook) y plantillas de `docs/` | Scripts de preparación con split por paciente, entrenamiento, evaluación con umbral en validación, interfaz Gradio y plantillas | 2026-11-18: se compiló todo y se corrió el pipeline completo con un mini dataset sintético con la estructura del de Kaggle (preparación, `check_leakage`, EDA, entrenamiento, evaluación, inferencia, app). Apareció un error: el umbral fijo sobrescribía el oficial. | Conservamos el código; se corrigió `evaluate.py` (commit `a8e9ccb`). <<PENDIENTE: lo que salga al correr con datos reales en Colab>> |
| 2026-11-18 | Claude Code (Anthropic) | Redactar las hipótesis H1 y H2 a partir de las decisiones del grupo (ambas lesiones, BWC benigna, umbral por sensibilidad ≥ 0,90) | Borrador de `docs/hipotesis.md` con predicciones numéricas y forma de ponerlas a prueba | Se contrastaron con la primera corrida real (2026-11-18): H1 quedó parcialmente confirmada y H2 confirmada en validación; los resultados se escribieron tal como salieron, incluida la parte de H1 que falló. <<PENDIENTE: José revisa la redacción final>> | Se conservaron las predicciones originales (no se ajustaron después de ver los resultados) |
| 2026-11-18 | Claude Code (Anthropic) | Configurar el repositorio con mi cuenta sin tocar la configuración global del PC | Pasos para identidad a nivel repo y remoto con usuario explícito | Verificado con `git config --local --list` y el autor en `git log` | Aplicado |
| 2026-11-18 | Claude Code (Anthropic) | Guía para correr el notebook en Colab y material de estudio para el control del día 7 | `docs/guia_colab.md` y material interno de estudio (glosario, guiones, preguntas) | <<PENDIENTE: seguir la guía en Colab y corregir lo que no coincida>> | <<PENDIENTE>> |
| 2026-11-18 | Claude Code (Anthropic) | Correr el notebook en Colab desde el navegador de la app (yo inicié sesión y autoricé Drive) y pasar los resultados a los docs | Ejecutó las celdas 1→20 (preparación, EDA, R0/R1/R2, evaluación en validación) y redactó los resultados en hipótesis, informe §4–5.2 y bitácora | Encontró y corrigió dos problemas reales (búsqueda lenta de archivos, redondeo del umbral en la tabla). <<PENDIENTE: José verifica las cifras contra los archivos de Drive>> | La prueba reservada NO se ejecutó: queda para cuando el grupo confirme el ganador |
| 2026-11-24 | Claude Code (Anthropic) | Implementar las indicaciones del profesor (más épocas, train+val, prueba externa), correrlas en Colab y armar la presentación | `--incluir-val` en `train.py`, `prepare_mias.py`, `evaluate_external.py`, corridas R3/R4, evaluación en mini-MIAS, informe §5.4 y §8, presentación .pptx y borradores para Andrés y Miguel | Probado primero con datos sintéticos; en Colab revisé que R4 usara 3.048 lesiones y que el umbral viniera de validación. <<PENDIENTE: José revisa la redacción de §5.4 y §8>> | Se reportó tal cual que más épocas y más datos no mejoraron y que el umbral no se transfiere a MIAS |
<!-- filas-jose: las filas nuevas van ENCIMA de esta línea -->

## Andrés Alfonso Luna Moreno

| Fecha | Herramienta | Pregunta / solicitud | Qué respondió / generó | Contraste realizado | Decisión final |
|---|---|---|---|---|---|
| 2026-11-19 | Claude Code (Anthropic), vía José; hoy Claude Code también me apoyó en la realización de la actividad | Redactar la tabla de etiquetas y las filas de bitácora con las decisiones de tipo de lesión y BWC | Texto de `docs/tabla_etiquetas.md` con alternativas y justificación | Comparé la tabla con `LABEL_MAP` en `src/prepare_data.py` (MALIGNANT=1, BENIGN=0, BENIGN_WITHOUT_CALLBACK=0) y con las cifras de `data/splits/resumen.json` (1.870 calcificaciones, 1.696 masas, 681 BWC) | Lo firmo con los cambios que hice al leerlo |
| 2026-11-24 | Claude (Anthropic), vía José: borrador `informe_secciones_andres.md`; hoy Claude Code también me apoyó en la realización de la actividad | Borrador de las secciones 2 (Datos) y 3 (Preparación) del informe con las cifras reales | Texto de fuente, cantidades, EDA, sesgos y etapas de preparación | Verifiqué cada cifra contra `data/splits/resumen.json`, `reports/eda/eda_resumen.json`, la salida de `src/check_leakage.py` y los aumentos en `src/common.py`; la licencia la confirmé en la página de TCIA | Cambios sobre el borrador: agregué la licencia (CC BY 3.0, DOI) y la nota de que cada fila es una lesión en una vista; agregué cuántas malignas hay por tipo y que el 79 % de las BWC son calcificaciones; expliqué por qué la partición da ≈71/15/15 y por qué los aumentos son razonables y suaves |
<!-- filas-andres: las filas nuevas van ENCIMA de esta línea -->

## Miguel Francisco Guerra Negrete

| Fecha | Herramienta | Pregunta / solicitud | Qué respondió / generó | Contraste realizado | Decisión final |
|---|---|---|---|---|---|
<!-- filas-miguel: las filas nuevas van ENCIMA de esta línea -->
