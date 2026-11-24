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
| Selección de errores a analizar | En validación: FN de puntuación más baja y FP de puntuación más alta (mínimo 4 y 4), más la tasa de error por tipo de lesión, `subtlety` y `assessment`; regla fijada antes de ver los casos | Miguel | `docs/analisis_errores.md`, `src/errores_por_grupo.py` |
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
| 2026-11-24 | El profesor pidió en la revisión: más épocas, más datos de entrenamiento y probar con imágenes nunca vistas. | Repartir de nuevo los datos (tocaría la prueba ya usada) · aumentos offline · train+val para el final · otro dataset externo (INbreast y VinDr piden permisos que no llegaban a tiempo) | R3: hasta 30 épocas (val elige). R4: train+val con las épocas de R3. Prueba externa con mini-MIAS (119 lesiones), umbral fijado en validación. | commit `56148a4`; `runs/R3_*`, `runs/R4_*`, `eval_externo_mias/` en Drive | 2026-11-24: más épocas no mejoró (0,812 vs 0,819); en MIAS el AUC baja a 0,71 y el umbral no se transfiere. |
| 2026-11-24 | En mini-MIAS el modelo marca como malignas 66 de 68 benignas con el umbral de CBIS-DDSM. | Reajustar el umbral con MIAS (sería evaluar con lo mismo que se ajusta) · reportarlo como está | Se reporta tal cual y se explica como cambio de dominio; la tabla de umbrales de MIAS se usa solo para describir el desplazamiento, no para elegir. | `runs/*/eval_externo_mias/metricas.json`, `tabla_umbrales.csv` | |
<!-- filas-jose: las filas nuevas van ENCIMA de esta línea -->

**Contribución individual de José:** repositorio y su configuración, hipótesis H1/H2, prueba del pipeline con datos sintéticos y corrección de `evaluate.py`, ejecución del notebook (R0/R1/R2), elección del umbral y del modelo ganador.

---

## Andrés Alfonso Luna Moreno

| Fecha | Dificultad / pregunta | Alternativas consideradas | Decisión | Evidencia | Revisión posterior |
|---|---|---|---|---|---|
| 2026-11-19 | ¿Masas, calcificaciones o ambas? | Solo masas · solo calcificaciones · ambas | Ambas (`--lesion both`): casi el doble de datos para ResNet18; `abnormality_type` se conserva para medir errores por tipo. | `docs/tabla_etiquetas.md` (Decisión 1) | 2026-11-24: en prueba (R1, umbral 0,066) las calcificaciones fallan más: sensibilidad 0,83 (17 de 102 malignas sin marcar) frente a 0,95 en masas (5 de 98), y AUC 0,744 frente a 0,837. 17 de los 22 FN son calcificaciones. Se mantiene la decisión (sin ellas se pierde la mitad de los datos) y se reporta como límite en el informe §7.1. |
| 2026-11-19 | ¿Qué hacer con BENIGN_WITHOUT_CALLBACK? | Contarla benigna · excluirla (`--bwc exclude`) | Benigna: es patología benigna; excluirla quitaría datos y dejaría solo las benignas difíciles. | `docs/tabla_etiquetas.md` (Decisión 2), `LABEL_MAP` en `src/prepare_data.py` | 2026-11-24: se confirmó el "en contra": las BWC son benignas fáciles. En prueba el modelo marca como malignas 31 de 114 BWC (especificidad 0,73) frente a 141 de 204 BENIGN (0,31); el AUC malignas vs BWC es 0,892 y malignas vs BENIGN 0,730. Se mantiene la decisión (son patología benigna), pero el AUC global de 0,788 es optimista frente a las benignas difíciles; se reporta en §7.1. Es una medición por subgrupos, no se reentrenó con `--bwc exclude`. |
| 2026-11-19 | ¿La partición por paciente quedó sin fuga y balanceada? | Split por lesión (rápido, pero mete la misma paciente en train y test) · por paciente | Por paciente con `StratifiedGroupKFold`, semilla 42. 3566 lesiones, 1566 pacientes; malignas train/val/test: 1034 / 222 / 200. check_leakage: SIN FUGA, 0 pacientes compartidas, 0 imágenes repetidas entre particiones. | `data/splits/lesiones.csv`, `data/splits/resumen.json`, salida de `src/check_leakage.py` | |
| 2026-11-24 | Para revisar D1 y D2 hacía falta la tasa de error por tipo de lesión, y el análisis de errores de Miguel aún no estaba en el repo. | Esperar a Miguel · usar solo mini-MIAS (27 calcificaciones, muy pocas) · contar los errores por grupo en las predicciones de prueba de R1 que ya existían | Contar FN y FP por `abnormality_type` y por `pathology` en `predicciones.csv` de la prueba de R1 (umbral 0,066). La prueba ya se había evaluado una vez: no se reentrena ni se cambia el umbral, solo se describe por subgrupos. Los totales coinciden con `metricas.json` (FN 22, FP 172, AUC 0,788). | `runs/R1_resnet18_aug/eval_test/predicciones.csv` (Drive del grupo); informe §7.1 | |
<!-- filas-andres: las filas nuevas van ENCIMA de esta línea -->

**Contribución individual de Andrés:** Decidí y documenté el tipo de lesión (masas y calcificaciones) y el tratamiento de `BENIGN_WITHOUT_CALLBACK`; versioné y verifiqué la partición por paciente sin fuga; redacté las secciones de datos y preparación del informe y los límites de los datos; y revisé mis dos decisiones con los errores del modelo por tipo de lesión.

---

## Miguel Francisco Guerra Negrete

| Fecha | Dificultad / pregunta | Alternativas consideradas | Decisión | Evidencia | Revisión posterior |
|---|---|---|---|---|---|
| 2026-11-24 | ¿Cómo elegir qué errores analizar sin hacer trampa? | Escoger a mano los más vistosos · al azar · los de puntuación más extrema, más la tasa por grupo | Método fijado antes de revisar los casos de error: en validación, FN de puntuación más baja y FP de puntuación más alta (mínimo 4 y 4) y tasa de error por tipo, subtlety y assessment. | `docs/analisis_errores.md` | |
| 2026-11-24 | ¿Qué hace la interfaz si suben una mamografía completa en vez de un recorte? | Rechazarla · intentar recortar la lesión · aceptarla con advertencia | Aceptar con advertencia visible: el modelo solo vio recortes, así que el resultado no vale, pero rechazar por tamaño puede bloquear recortes grandes legítimos. El corte de tamaño se fijará con el tamaño máximo real de los recortes (EDA). | `app.py` (`analizar`), `reports/eda/eda_resumen.json` | <<PENDIENTE: ajustar el corte con el EDA>> |
<!-- filas-miguel: las filas nuevas van ENCIMA de esta línea -->

**Contribución individual de Miguel:** <<PENDIENTE: la escribe Miguel>>
