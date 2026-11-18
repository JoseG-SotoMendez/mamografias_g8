# Bitácora del proceso — Grupo 8

Una entrada por cada problema o decisión relevante, escrita **el mismo día** (la fecha de la fila y la del commit deben coincidir).
Formato exigido: fecha · dificultad · alternativas · decisión · evidencia · contribución individual.

Cada integrante escribe **solo en su sección** (así no chocamos al hacer `git pull`). La columna "Revisión posterior" se llena cuando los resultados confirman o tumban la decisión.

## Decisiones de grupo (resumen)

| Decisión | Valor | Quién la documenta y defiende | Dónde |
|---|---|---|---|
| Fuente de datos | CBIS-DDSM, versión JPEG de Kaggle (`awsaf49/cbis-ddsm-breast-cancer-image-dataset`) | José | esta bitácora, `notebooks/Colab_Mamografias.ipynb` |
| Tipo de lesión | Masas + calcificaciones (`--lesion both`) | Andrés | `docs/tabla_etiquetas.md` |
| `BENIGN_WITHOUT_CALLBACK` | Benigna (0) (`--bwc benign`) | Andrés | `docs/tabla_etiquetas.md` |
| División | Por paciente, 70/15/15, semilla 42 (`StratifiedGroupKFold`) | Andrés | `src/prepare_data.py`, `src/check_leakage.py`, `data/splits/lesiones.csv` |
| Experimento controlado | R1 vs R2: solo cambian los aumentos | José | `docs/hipotesis.md` (H1) |
| Umbral | El más alto con sensibilidad ≥ 0,90, elegido en validación | José | `docs/hipotesis.md` (H2), `runs/*/threshold.json` |
| Selección de errores a analizar | <<PENDIENTE: Miguel>> | Miguel | `docs/analisis_errores.md` |
| Validaciones de la interfaz | <<PENDIENTE: Miguel>> | Miguel | `app.py`, capturas |

---

## José Gil Soto Méndez

| Fecha | Dificultad / pregunta | Alternativas consideradas | Decisión | Evidencia | Revisión posterior |
|---|---|---|---|---|---|
| 2026-11-18 | El PC tiene configurada globalmente otra cuenta de Git (`jsotomendez`, correo universitario) y el Administrador de credenciales de Windows guarda esa cuenta para github.com. Los commits iban a salir con el autor equivocado. | Cambiar la configuración global · borrar la credencial guardada · configurar identidad solo en este repo | Identidad `JoseG-SotoMendez` **solo a nivel repo** (`git config user.name/user.email` sin `--global`) y URL del remoto con el usuario explícito, para que el administrador de credenciales no use la otra cuenta. No se borró nada. | `git config --local --list`; autor de los commits en `git log` | |
| 2026-11-18 | ¿De dónde sacamos las imágenes? | DICOM originales de TCIA (≈160 GB la colección completa, requiere NBIA Data Retriever) · solo "ROI and cropped" de TCIA · JPEG de Kaggle derivado de TCIA | JPEG de Kaggle: cabe en Colab, ya trae `dicom_info.csv` para ubicar cada recorte y conserva los CSV oficiales con `pathology`. El código también soporta `--source tcia` si hay que volver al original. | `src/prepare_data.py` (`resolve_kaggle`), celda 3 del notebook | 2026-11-18: en Colab el dataset ya estaba en la caché (`/kaggle/input/...`), sin descarga. De 3.568 lesiones en los CSV, 3.566 tienen recorte; 2 quedaron en `descartadas.csv`; 0 duplicados MD5. |
| 2026-11-18 | Al probar el pipeline con un mini dataset sintético, `evaluate.py --threshold 0.3` sobrescribía `threshold.json` y `eval_val/`: la app y la evaluación final habrían usado ese umbral sin que nadie lo notara. | Dejarlo y "acordarse" de volver a correr · guardar el umbral fijo aparte | Un umbral fijo es exploración: se guarda en `eval_<split>_umbral_<valor>/` y no toca `threshold.json`. | commit `a8e9ccb` | |
| 2026-11-18 | ¿Cómo diseñar el experimento controlado? | Cambiar modelo y aumentos a la vez · cambiar solo el lr · cambiar solo los aumentos | R1 (ResNet18 con aumentos) vs R2 (igual, sin aumentos). Se mantiene modelo, datos, semilla, lr, épocas y batch. | `docs/hipotesis.md` (H1), commit `262adc1` | 2026-11-18: R1 0,819 vs R2 0,782 de AUC en validación; R2 memorizó (pérdida de entrenamiento 0,028 en la época 6). H1 se confirma en (b) y (c) pero no en (a): medir la brecha "en la mejor época" fue mala elección, porque la mejor de R2 es la 1. |
| 2026-11-18 | ¿Con qué criterio se fija el umbral? | 0,5 fijo · Youden · sensibilidad objetivo 0,90 · sensibilidad objetivo 0,95 | Sensibilidad ≥ 0,90 en validación: un falso negativo (maligna marcada como benigna) es peor que un falso positivo (estudio extra). 0,95 se descartó para no disparar los falsos positivos con un dataset pequeño. | `docs/hipotesis.md` (H2), commit `262adc1` | 2026-11-18: con R1 el umbral quedó en 0,066; de 0,5 a 0,066 los FN bajan de 88 a 22 y los FP suben de 44 a 144 (validación). Se mantiene el criterio. |
| 2026-11-18 | Primera corrida real en Colab: `prepare_data.py` tardó ~3 min solo en encontrar los CSV, porque `rglob` recorre las miles de carpetas de `jpeg/` en el disco montado de Kaggle, y lo hace tres veces (CSV, `dicom_info.csv`, carpeta `jpeg`). | Esperar · pasar las rutas a mano por parámetro · buscar primero donde Kaggle pone los archivos | `find_files`: busca primero en `csv/` y `jpeg/` y solo si no encuentra nada recorre todo el árbol. Verificado con el dataset sintético: la partición sale idéntica byte a byte. | commit de este arreglo; salida de la celda 4 del notebook | |
| 2026-11-18 | Primera corrida completa: ¿qué modelo gana y con qué criterio? | Elegir por exactitud · por AUC · por especificidad a sensibilidad fija | R1 (ResNet18 + aumentos): mayor AUC (0,819) y menos FP (144) a la misma sensibilidad (0,90). Elegido solo con validación; la prueba sigue sin usarse. | `reports/comparacion_val.md`, `runs/*/config.json` (commit `a73c475`) | |
| 2026-11-18 | La fila del umbral elegido en `tabla_umbrales.csv` daba 199 TP y `metricas.json` 200 TP para el "mismo" umbral. | Ignorarlo · documentar la diferencia · corregir la causa | La tabla redondeaba el umbral a 4 decimales (0,0664 vs 0,06638). Ahora usa el valor exacto. | commit del arreglo en `src/evaluate.py` | |
<!-- filas-jose: las filas nuevas van ENCIMA de esta línea -->

**Contribución individual de José:** repositorio y su configuración, hipótesis H1/H2, prueba del pipeline con datos sintéticos y corrección de `evaluate.py`, ejecución del notebook (R0/R1/R2), elección del umbral y del modelo ganador.

---

## Andrés Alfonso Luna Moreno

| Fecha | Dificultad / pregunta | Alternativas consideradas | Decisión | Evidencia | Revisión posterior |
|---|---|---|---|---|---|
<!-- filas-andres: las filas nuevas van ENCIMA de esta línea -->

**Contribución individual de Andrés:** <<PENDIENTE: la escribe Andrés>>

---

## Miguel Francisco Guerra Negrete

| Fecha | Dificultad / pregunta | Alternativas consideradas | Decisión | Evidencia | Revisión posterior |
|---|---|---|---|---|---|
<!-- filas-miguel: las filas nuevas van ENCIMA de esta línea -->

**Contribución individual de Miguel:** <<PENDIENTE: la escribe Miguel>>
