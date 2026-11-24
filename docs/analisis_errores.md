# Análisis de errores

Responsable: Miguel Guerra. El método se escribió **antes** de revisar los casos de error uno por uno (las métricas generales ya existían desde el 18-nov), para no escoger después los casos que mejor quedan en el informe.

## Método (decidido antes de revisar los casos de error)

1. **Dónde se miran los errores.** En **validación** (`runs/<ganador>/eval_val/`). La prueba se evalúa una sola vez al final; sus errores se reportan, pero no se usan para cambiar nada.
2. **Qué errores se muestran.**
   - **Falsos negativos (FN):** malignas que el modelo llamó benignas. Se toman en el orden de `falsos_negativos.csv`, que ya viene de la puntuación más baja a la más alta: los primeros son los errores en los que el modelo estaba "más seguro" de estar bien. Mínimo 4, incluyendo masas y calcificaciones si hay de ambas.
   - **Falsos positivos (FP):** benignas que el modelo llamó malignas, de la puntuación más alta a la más baja (`falsos_positivos.csv`). Mínimo 4.
3. **Qué se anota de cada error:** `lesion_uid`, puntuación, tipo de lesión, vista, `subtlety` (1 = muy sutil, 5 = obvia), `assessment` (BI-RADS que dio el radiólogo) y una hipótesis de por qué falló.
4. **Tasa de error por grupo:** `python src/errores_por_grupo.py --pred runs/<ganador>/eval_val/predicciones.csv`. Responde si el modelo falla más con calcificaciones que con masas o con lesiones sutiles. Así un promedio bueno no esconde un grupo en el que el modelo es malo.
5. **Las imágenes de los errores** (`falsos_negativos.png`, `falsos_positivos.png`) van al informe, **no** al repositorio, porque son recortes de pacientes.

**Hipótesis de causas que vamos a contrastar (no son resultados):** lesiones con `subtlety` baja; calcificaciones finas que se pierden al reducir el recorte a 224 px; benignas con `assessment` 4–5 (también le parecieron sospechosas al radiólogo); recortes muy grandes o muy pequeños deformados al redimensionar.

## Falsos negativos (validación)
<<PENDIENTE: tabla con los FN reales>>

## Falsos positivos (validación)
<<PENDIENTE: tabla con los FP reales>>

## Tasa de error por grupo
<<PENDIENTE: salida de src/errores_por_grupo.py>>

## Errores en prueba (solo se reportan)
<<PENDIENTE: tras la evaluación final>>
