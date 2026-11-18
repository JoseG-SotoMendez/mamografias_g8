# Tabla de mapeo de etiquetas (obligatoria)

Fuente: columna `pathology` de los CSV `mass_case_description_*` y `calc_case_description_*` de CBIS-DDSM.
La patología proviene de la verificación del DDSM original (no es una lectura BI-RADS).

| Etiqueta original (`pathology`) | Significado | Etiqueta del proyecto | Valor |
|---|---|---|---|
| `MALIGNANT` | Lesión con patología maligna | maligna | 1 |
| `BENIGN` | Lesión con patología benigna | benigna | 0 |
| `BENIGN_WITHOUT_CALLBACK` | Benigna; no requirió seguimiento adicional | benigna *(decisión del equipo)* | 0 |

**Decisión sobre `BENIGN_WITHOUT_CALLBACK`:** [COMPLETAR: la incluimos como benigna / la excluimos, y por qué.
Se cambia con `--bwc exclude` en `prepare_data.py`.]

**Otras columnas que NO se usan como etiqueta:**
- `assessment` (categoría BI-RADS asignada): es sospecha radiológica, no patología confirmada. Se puede usar en el análisis de errores, no como objetivo.
- `subtlety`: dificultad de visualización (1–5); útil para explicar errores.

**Tipo de lesión incluido:** [COMPLETAR: masas, calcificaciones o ambas (`--lesion`), y por qué.]

**Advertencia:** la ausencia de una lesión anotada no significa ausencia de cáncer. Este proyecto solo clasifica lesiones ya anotadas.
