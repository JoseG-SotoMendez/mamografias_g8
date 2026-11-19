# Tabla de mapeo de etiquetas (obligatoria)

Fuente: columna `pathology` de los CSV `mass_case_description_*` y `calc_case_description_*` de CBIS-DDSM.
La patología viene de la verificación del DDSM original; no es una lectura BI-RADS.

| Etiqueta original (`pathology`) | Significado | Etiqueta del proyecto | Valor |
|---|---|---|---|
| `MALIGNANT` | Lesión con patología maligna confirmada | maligna | 1 |
| `BENIGN` | Lesión con patología benigna confirmada | benigna | 0 |
| `BENIGN_WITHOUT_CALLBACK` | Benigna; el radiólogo no consideró necesario volver a llamar a la paciente | benigna *(decisión del equipo)* | 0 |

El mapeo está en el código, en `LABEL_MAP` de `src/prepare_data.py`.

## Decisión 1 — Tipo de lesión: masas **y** calcificaciones (`--lesion both`)

| Alternativa | A favor | En contra |
|---|---|---|
| Solo masas | Imágenes más parecidas entre sí; el modelo aprende un solo tipo de patrón | Cerca de la mitad de los datos; con ~11 M parámetros en ResNet18 eso es pedir sobreajuste |
| Solo calcificaciones | Igual cantidad aproximada que masas | Son puntos finos que se pierden al reducir el recorte a 224 px |
| **Ambas (elegida)** | Casi el doble de lesiones para entrenar y evaluar | Mezcla dos problemas visuales distintos |

**Por qué ambas:** con pocos datos, la cantidad pesa más que la homogeneidad. Y no perdemos la diferencia: la columna `abnormality_type` queda en `lesiones.csv`, así que en el análisis de errores medimos por separado cuánto falla el modelo con masas y con calcificaciones. Si las calcificaciones fallan mucho más, eso se reporta como límite; no se esconde en un promedio.

## Decisión 2 — `BENIGN_WITHOUT_CALLBACK` cuenta como **benigna** (`--bwc benign`)

| Alternativa | A favor | En contra |
|---|---|---|
| **Contarla como benigna (elegida)** | Es patología benigna: el problema pregunta benigna vs maligna, y esta lo es. Conserva todas las lesiones | Son lesiones "obviamente benignas", así que pueden hacer la clase benigna un poco más fácil |
| Excluirla (`--bwc exclude`) | Solo quedan benignas que sí generaron seguimiento, más parecidas a las malignas | Menos datos y un sesgo de selección distinto: quitamos a propósito las benignas fáciles |

**Por qué benigna:** "sin re-llamado" describe lo que hizo el radiólogo, no cambia la patología. Excluirlas sería como evaluar a un detector de mentiras solo con mentirosos difíciles. La alternativa queda a un parámetro de distancia (`--bwc exclude`) por si el grupo quiere medir su efecto. Cuántas son: 681 de 3.566 lesiones (19 %), según `pathology_original` en `data/splits/resumen.json`.

## Columnas que NO se usan como etiqueta
- `assessment` (categoría BI-RADS): es sospecha radiológica, no patología confirmada. Se usa para analizar errores, no como objetivo.
- `subtlety` (1 = muy sutil, 5 = obvia): sirve para explicar errores.

**Advertencia:** que una lesión no esté anotada no significa que no haya cáncer. Este proyecto solo clasifica lesiones ya anotadas.