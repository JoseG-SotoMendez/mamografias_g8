# Evidencia del control del día 7 (24-nov-2026)

El control fue el martes 24 de noviembre. Esta página junta la evidencia verificable: qué había en el repositorio cuando lo presentamos, qué nos pidió el profesor y qué hicimos con eso. Todo se puede comprobar con `git log` y con los archivos citados.

## Avance que llevaba cada uno al control

| Integrante | Commits antes del control | Qué demuestran |
|---|---|---|
| José Gil Soto Méndez | 18-nov (11 commits) y 24-nov (3 commits) | Estructura del repositorio, hipótesis escritas antes de entrenar (`262adc1`), corridas R0–R2 y comparación en validación (`bc04eaa`), prueba reservada usada una vez (`ce77327`), indicaciones del día 7 (`56148a4`, `d427152`) |
| Andrés Alfonso Luna Moreno | 19-nov (2 commits) y 24-nov (3 commits) | Tabla de etiquetas con sus dos decisiones (`82d838c`), partición por paciente versionada (`5fa4602`), secciones de datos del informe y revisión de sus decisiones con los errores (`e109a8a`, `9b3a519`, `d8cbe5d`) |
| Miguel Francisco Guerra Negrete | 24-nov (5 commits) | Método del análisis de errores escrito antes de ver los casos (`28528f2`), `errores_por_grupo.py` (`bc3db5d`), análisis de errores (`64a531a`), interfaz y capturas (`8207d05`), límites clínicos (`f4c33c0`) |

Comando para verlo: `git log --date=format:"%d-%m %H:%M" --format="%ad %an %s"`.

Lo que el enunciado pide para este control y dónde está:
- **Dos decisiones propias con evidencia fechada:** José (umbral por sensibilidad y experimento controlado R1/R2), Andrés (masas + calcificaciones y BWC como benigna), Miguel (método de errores fijado antes y advertencia de mamografía completa). Cada una tiene su fila en `docs/bitacora.md` con fecha, alternativas y evidencia.
- **Lectura de imágenes, distribución por clases y no mezcla de pacientes:** `src/prepare_data.py`, `src/eda.py` y `src/check_leakage.py` (0 pacientes compartidas).
- **Entrenamiento preliminar e inconveniente real con su solución:** curvas de R0–R2; inconvenientes documentados en la bitácora (búsqueda lenta de archivos en Colab, umbral que se sobrescribía, redondeo del umbral en la tabla).
- **Modificación en vivo:** el umbral se cambia con `evaluate.py --threshold <valor>` (no toca el umbral oficial) o con el deslizador de la app; los falsos negativos se localizan en `falsos_negativos.csv`.

## Lo que pidió el profesor y qué hicimos

| Pedido | Qué hicimos | Resultado | Evidencia |
|---|---|---|---|
| Entrenar más ("más épocas") | R3: hasta 30 épocas con paciencia 7; validación elige la mejor | Mejor época 10, AUC 0,812: no mejora a R1 (0,819); después sobreajusta | commit `56148a4`, informe §5.4 |
| Más datos de entrenamiento | R4: entrenamiento + validación (3.048 lesiones, +21 %), con las épocas de R3 | Mismo desempeño en datos externos | `--incluir-val` en `src/train.py` |
| Probar el modelo final con un conjunto nuevo | Prueba externa con mini-MIAS (119 lesiones de 104 pacientes), umbral fijado en CBIS-DDSM | AUC 0,71–0,73; el umbral no se transfiere (cambio de dominio) | `src/prepare_mias.py`, `src/evaluate_external.py`, `d427152` |
| (Verificación del 24-nov) La app no puede depender de Colab: debe estar publicada 24/7 | App de Gradio en Render (modelo R1 en ONNX) y versión en el navegador en Hugging Face Spaces y GitHub Pages; despertador cada 10 min y vigilante cada 6 h | En línea; mismas puntuaciones que PyTorch (README e informe §6) | `render.yaml`, `deploy/`, `web/`, `.github/workflows/` |
