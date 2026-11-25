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

R1 (`runs/R1_resnet18_aug/eval_val/`), umbral 0,066: **22 FN de 222 malignas** (17 pacientes distintas: en 5 la misma lesión se escapó en las dos vistas, CC y MLO). Siguiendo la regla, se toman los 4 primeros de `falsos_negativos.csv` (puntuación más baja). El 4.º ya es una masa, así que quedan cubiertos los dos tipos.

| # | `lesion_uid` | Puntuación | Tipo | Vista | `subtlety` | `assessment` | Hipótesis de la causa |
|---|---|---|---|---|---|---|---|
| 1 | P_00479_LEFT_CC_calcification_1 | 0,0002 | calcificación | CC | 4 | 5 | Grupo de calcificaciones pequeñas sobre un fondo con muchas estructuras lineales; al reducir a 224 px los puntos finos se confunden con el tejido. El radiólogo la vio muy sospechosa (BI-RADS 5): el error es del modelo, no del caso. |
| 2 | P_01400_LEFT_MLO_calcification_1 | 0,0004 | calcificación | MLO | 3 | 4 | Pocas calcificaciones gruesas y redondeadas, de aspecto parecido a las benignas. En la otra vista (CC) también se escapó (FN n.º 6): el modelo falla con este patrón, no con una vista. |
| 3 | P_01443_RIGHT_MLO_calcification_1 | 0,0004 | calcificación | MLO | 4 | 4 | Dos o tres calcificaciones aisladas en un recorte casi uniforme: después de redimensionar queda muy poca señal para el modelo. |
| 4 | P_00644_LEFT_CC_mass_1 | 0,0076 | masa | CC | 5 | 4 | Masa grande de bordes bastante definidos que ocupa casi todo el recorte; a primera vista parece benigna. El modelo no ve el tejido alrededor de la lesión, que es lo que suele delatar el borde. |

Otros FN de la lista (n.º 5 a 8): tres calcificaciones más (P_01339 en CC y MLO, P_01400 en CC) y una masa (P_00420_RIGHT_MLO_mass_2), todas con `assessment` 4.

## Falsos positivos (validación)

**144 FP de 299 benignas.** Los 4 primeros de `falsos_positivos.csv` (puntuación más alta):

| # | `lesion_uid` | Puntuación | Tipo | Vista | `subtlety` | `assessment` | `pathology` | Hipótesis de la causa |
|---|---|---|---|---|---|---|---|---|
| 1 | P_00250_LEFT_MLO_calcification_1 | 0,992 | calcificación | MLO | 2 | 4 | BENIGN | Recorte muy angosto y alargado, con el borde de la imagen visible: al llevarlo a 224×224 se deforma. Además, el radiólogo también lo vio sospechoso (BI-RADS 4). |
| 2 | P_01769_LEFT_CC_calcification_1 | 0,959 | calcificación | CC | 3 | 2 | BENIGN_WITHOUT_CALLBACK | Calcificación grande y gruesa, típica de benigna. El modelo parece reaccionar a "zona muy blanca y grande" y no a la forma. |
| 3 | P_00120_LEFT_MLO_mass_1 | 0,956 | masa | MLO | 5 | 4 | BENIGN | Masa grande y muy densa que satura el recorte; brillo alto como el de muchas malignas. BI-RADS 4: también le pareció sospechosa al radiólogo. |
| 4 | P_00572_RIGHT_CC_mass_1 | 0,956 | masa | CC | 1 | 4 | BENIGN | Lesión muy sutil en tejido denso; el patrón irregular del tejido se parece al de una masa espiculada. BI-RADS 4. |

Otros FP de la lista (n.º 5 a 8): tres masas (P_00911, P_00847, P_00630) y una calcificación (P_00355). De los 8 primeros, 7 son `BENIGN` (benignas que sí generaron seguimiento).

## Tasa de error por grupo

Salida de `python src/errores_por_grupo.py --pred runs/R1_resnet18_aug/eval_val/predicciones.csv`:

```
521 lesiones | FN=22 FP=144

=== Por abnormality_type ===
                  lesiones  malignas  FN  FP  %FN_de_malignas  %FP_de_benignas
calcification          273       107  14  69             13.1             41.6
mass                   248       115   8  75              7.0             56.4

=== Por subtlety ===
          lesiones  malignas  FN  FP  %FN_de_malignas  %FP_de_benignas
1               27        16   0   9              0.0             81.8
2               62        34   4  19             11.8             67.9
3              159        67   9  42             13.4             45.7
4              122        47   3  31              6.4             41.3
5              151        58   6  43             10.3             46.2

=== Por assessment ===
            lesiones  malignas  FN  FP  %FN_de_malignas  %FP_de_benignas
0                 23         1   0  15              0.0             68.2
2                 99         0   0  28              NaN             28.3
3                 67        24   3  18             12.5             41.9
4                257       125  16  80             12.8             60.6
5                 75        72   3   3              4.2            100.0

=== Por image_view ===
            lesiones  malignas  FN  FP  %FN_de_malignas  %FP_de_benignas
CC               246       105  10  66              9.5             46.8
MLO              275       117  12  78             10.3             49.4
```

Por `pathology` de las benignas (conteo aparte sobre `falsos_positivos.csv`): 124 FP de 213 `BENIGN` (58 %) y 20 de 86 `BENIGN_WITHOUT_CALLBACK` (23 %).

**Qué dicen los datos frente a las hipótesis del método:**
- **Calcificaciones: se confirma.** Se escapa el 13,1 % de las calcificaciones malignas frente al 7,0 % de las masas; 14 de los 22 FN y los 3 primeros de la lista son calcificaciones. Coincide con la prueba (17 de 22 FN, §7.1 del informe).
- **`subtlety` baja: no se confirma para los FN.** Los FN no se concentran en las lesiones sutiles (0 FN con `subtlety` 1, 11,8 % con 2, 10,3 % con 5). Sí se ve en los FP: con `subtlety` 1 y 2 el modelo marca como malignas el 82 % y el 68 % de las benignas.
- **Benignas con `assessment` 4–5: se confirma.** Con `assessment` 4 sale como FP el 60,6 % de las benignas, frente al 28,3 % con `assessment` 2; y las BWC casi no engañan al modelo (23 %). Muchos FP son casos que también le parecieron sospechosos al radiólogo.
- **Recortes deformados al redimensionar: posible en algunos casos** (FP n.º 1), pero no se puede medir con `predicciones.csv`, que no trae el tamaño del recorte.
- La vista (CC o MLO) no cambia nada: 9,5 % vs 10,3 % de FN.

## Errores en prueba (solo se reportan)

R1 en la prueba reservada (evaluada una sola vez, umbral 0,066 de validación): **22 FN de 200 malignas y 172 FP de 318 benignas** (`runs/R1_resnet18_aug/eval_test/metricas.json`). No se usaron para cambiar nada: ni el modelo, ni el umbral, ni el método. Los conteos por grupo salen de `predicciones.csv` de la prueba (los hizo Andrés para revisar sus decisiones de datos, informe §5.5 y bitácora del 24-nov):

| Grupo (prueba) | Lesiones | Errores | Métrica | ROC-AUC del grupo |
|---|---|---|---|---|
| Calcificaciones malignas | 102 | 17 FN | sensibilidad 0,83 | 0,744 (todas las calcificaciones) |
| Masas malignas | 98 | 5 FN | sensibilidad 0,95 | 0,837 (todas las masas) |
| Benignas `BENIGN` | 204 | 141 FP | especificidad 0,31 | 0,730 (malignas vs `BENIGN`) |
| Benignas `BENIGN_WITHOUT_CALLBACK` | 114 | 31 FP | especificidad 0,73 | 0,892 (malignas vs BWC) |

**Lectura.** La prueba confirma lo que vimos en validación con el método fijado antes: el punto débil son las calcificaciones (17 de los 22 FN) y las falsas alarmas se concentran en las benignas que sí generaron seguimiento, no en las BWC. Como en validación, no es un problema de lesiones sutiles sino, muy probablemente, de resolución: el detalle fino de las calcificaciones se pierde al reducir el recorte a 224 px.
