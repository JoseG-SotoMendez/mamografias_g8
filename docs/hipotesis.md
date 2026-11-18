# Hipótesis (escribirlas ANTES de experimentar, con fecha)

El enunciado exige **dos hipótesis propias**, una de ellas sobre **sensibilidad o umbral de decisión**.
Esta parte la deben redactar ustedes: es una de las responsabilidades que el curso no permite delegar a la IA.

Formato sugerido para cada una:

| Campo | Contenido |
|---|---|
| Fecha | |
| Autor(es) | |
| Hipótesis | Afirmación concreta y verificable |
| Predicción medible | Qué métrica, en qué partición, qué dirección/magnitud esperan |
| Cómo se pone a prueba | Qué corridas/comandos se comparan, qué se mantiene constante |
| Resultado | (completar después) |
| ¿Se confirma? / Interpretación | (completar después) |

## H1 — sobre el modelo o los datos
[COMPLETAR]

## H2 — sobre sensibilidad / umbral
[COMPLETAR]

Pistas para pensarlas (no son hipótesis hechas):
- ¿Qué esperan que pase con la sensibilidad para malignas si eligen el umbral para sensibilidad ≥ 0,90 en vez de 0,5? ¿Cuánto creen que caerá la especificidad?
- ¿Esperan que la transferencia de aprendizaje supere a la CNN pequeña en ROC-AUC? ¿Por cuánto, considerando el tamaño del dataset?
- ¿Esperan diferencias entre masas y calcificaciones?
- ¿Qué efecto esperan de los aumentos de datos sobre el sobreajuste (brecha entre pérdida de train y val)?
