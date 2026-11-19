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
<!-- filas-jose: las filas nuevas van ENCIMA de esta línea -->

## Andrés Alfonso Luna Moreno

| Fecha | Herramienta | Pregunta / solicitud | Qué respondió / generó | Contraste realizado | Decisión final |
|---|---|---|---|---|---|
| 2026-11-19 | Claude Code (Anthropic), vía José; hoy Claude Code también me apoyó en la realización de la actividad | Redactar la tabla de etiquetas y las filas de bitácora con las decisiones de tipo de lesión y BWC | Texto de `docs/tabla_etiquetas.md` con alternativas y justificación | Comparé la tabla con `LABEL_MAP` en `src/prepare_data.py` (MALIGNANT=1, BENIGN=0, BENIGN_WITHOUT_CALLBACK=0) y con las cifras de `data/splits/resumen.json` (1.870 calcificaciones, 1.696 masas, 681 BWC) | Lo firmo con los cambios que hice al leerlo |
<!-- filas-andres: las filas nuevas van ENCIMA de esta línea -->

## Miguel Francisco Guerra Negrete

| Fecha | Herramienta | Pregunta / solicitud | Qué respondió / generó | Contraste realizado | Decisión final |
|---|---|---|---|---|---|
<!-- filas-miguel: las filas nuevas van ENCIMA de esta línea -->
