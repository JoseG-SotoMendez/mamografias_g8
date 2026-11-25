# Clasificación de lesiones mamográficas benignas y malignas mediante aprendizaje profundo

**Grupo 8:** José Gil Soto Méndez, Andrés Alfonso Luna Moreno y Miguel Francisco Guerra Negrete
**Curso:** Aprendizaje Computacional, Ingeniería de Sistemas · Noviembre de 2026

> **Aviso:** esto es un prototipo académico; no apto para diagnóstico ni decisiones clínicas.

## 1. El problema que quisimos resolver

Construimos un prototipo que recibe el **recorte de una lesión** ya marcada en una mamografía y dice si su patología de referencia es benigna o maligna. El usuario es académico: el prototipo sirve para estudiar cómo se comporta un clasificador de imágenes médicas y qué decisiones lo vuelven confiable o engañoso. No es una herramienta clínica.

**Unidad de análisis: la lesión.** Cada fila de los CSV de CBIS-DDSM es una lesión vista en una proyección (CC o MLO), con su recorte (ROI) y su patología verificada. Trabajar con la mamografía completa obligaría al modelo a encontrar la lesión antes de clasificarla, y eso es otro proyecto. Trabajar por mama o por examen mezclaría lesiones con etiquetas distintas. La lesión es donde la etiqueta tiene un significado claro.

**Qué significa la etiqueta.** La columna `pathology` dice si la lesión resultó benigna o maligna según la verificación del DDSM. Es la patología de **una lesión ya anotada**, no el diagnóstico de la paciente: alguien puede tener una lesión benigna en un recorte y un cáncer en otro sitio que nadie marcó. Tampoco usamos BI-RADS (`assessment`) como etiqueta, porque eso es sospecha radiológica y no cáncer confirmado.

**Los dos errores no pesan igual.** Un falso negativo es una lesión maligna que el sistema llama benigna: en la vida real retrasaría un diagnóstico. Un falso positivo es una benigna llamada maligna: cuesta un estudio más, una biopsia y un susto, pero se corrige. Pensamos el umbral como la sensibilidad de un detector de humo: muy sensible suena con las tostadas, pero no deja pasar un incendio. Nosotros preferimos las tostadas. Por eso no dejamos el umbral en 0,5: lo elegimos **en validación** como el más alto que atrapa al menos el **90 % de las malignas**.

**Hipótesis (escritas el 18-nov, antes de entrenar; `docs/hipotesis.md`).**

- **H1:** con ResNet18 preentrenada, activar los aumentos de datos (R1) frente a no activarlos (R2) reduce el sobreajuste y sube el ROC-AUC de validación en al menos 0,02.
- **H2:** el umbral elegido para sensibilidad ≥ 0,90 queda por debajo de 0,5, baja la especificidad de validación al rango 0,35–0,60 y, en la prueba, la sensibilidad queda entre 0,80 y 0,90.

## 2. Los datos

**Fuente.** CBIS-DDSM (*Curated Breast Imaging Subset of DDSM*), de The Cancer Imaging Archive, con licencia **CC BY 3.0** (DOI 10.7937/K9/TCIA.2016.7O02S9CY). Usamos la copia en JPEG de Kaggle (`awsaf49/cbis-ddsm-breast-cancer-image-dataset`), que conserva los CSV oficiales con la patología de cada lesión y el archivo `dicom_info.csv` para ubicar cada recorte. Los DICOM originales pesan unos 160 GB; la versión JPEG cabe en Colab. Son mamografías de **película escaneada**, de hospitales de Estados Unidos y de los años noventa.

**Cantidades.** Los cuatro CSV (masas y calcificaciones, entrenamiento y prueba oficiales) describen 3.568 lesiones. Dos no tenían recorte y se descartaron, y no hubo archivos repetidos (comparamos el MD5 de cada imagen). Quedaron **3.566 lesiones de 1.566 pacientes** (2,3 por paciente en promedio y hasta 24 en una misma paciente): 1.870 calcificaciones y 1.696 masas, 1.894 en vista MLO y 1.672 en CC. Cada fila es una lesión **en una vista**, así que la misma lesión vista en CC y en MLO cuenta dos veces.

| Etiqueta original (`pathology`) | Etiqueta del proyecto | Lesiones |
|---|---|---|
| `MALIGNANT` | maligna (1) | 1.456 |
| `BENIGN` | benigna (0) | 1.429 |
| `BENIGN_WITHOUT_CALLBACK` (BWC) | benigna (0) | 681 |

**Dos decisiones de datos** (`docs/tabla_etiquetas.md`). Usamos **masas y calcificaciones juntas**, porque casi duplica los datos y guardamos el tipo de lesión para medir los errores por separado. Contamos las **BWC como benignas**, porque lo son: "sin re-llamado" describe lo que hizo el radiólogo, no cambia la patología. Excluirlas habría dejado solo las benignas difíciles.

**Lo que vimos en el análisis exploratorio.** El 40,8 % de las lesiones es maligno: hay algo de desbalance (6 benignas por cada 4 malignas), pero no tanto como para remuestrear. Los dos tipos no se reparten igual: las masas son 46 % malignas y las calcificaciones 36 %, y el 79 % de las BWC son calcificaciones. Los recortes van de 68 a 2.893 píxeles de ancho (mediana 340 × 337), y como todos se llevan a 224 × 224, los grandes se reducen más de diez veces. Eso le pega sobre todo a las calcificaciones, que son puntos finos. No hubo imágenes ilegibles.

![Figura 1. Ejemplos de recortes de entrenamiento: benignas arriba y malignas abajo (CBIS-DDSM, CC BY 3.0).](figuras/ejemplos_eda.png)

![Figura 2. Balance de clases por partición (izquierda) y lesiones por tipo y patología original (derecha).](figuras/eda_balance_tipo.png)

**Sesgos posibles.** Película escaneada en vez de mamografía digital; una población y una época concretas; casos que entraron al conjunto porque tenían una lesión (el modelo nunca vio una mamografía normal); calidad de escaneo variable; y la mezcla desigual de tipos y de BWC, que puede hacer que el modelo aprenda atajos del tipo de lesión en vez de la patología.

## 3. Preparación: cinco etapas

1. **Lectura y etiquetas.** Unimos los cuatro CSV y traducimos `pathology` a 0/1 con `LABEL_MAP` (`src/prepare_data.py`).
2. **Limpieza.** Ubicamos cada recorte con `dicom_info.csv` usando la serie *cropped* (el recorte, no la máscara) y descartamos lesiones sin imagen y archivos idénticos.
3. **Partición por paciente.** `StratifiedGroupKFold` con semilla 42 agrupa por `patient_id` y estratifica por etiqueta. Resultado: **2.527 lesiones de entrenamiento (1.115 pacientes), 521 de validación (225) y 518 de prueba (226)**, con 40,9 %, 42,6 % y 38,6 % de malignas (≈71/15/15). `src/check_leakage.py` confirma **0 pacientes compartidas** entre particiones y 0 imágenes idénticas cruzadas.
4. **Transformación.** Escala de grises copiada a 3 canales, redimensionado a 224 × 224 y normalización con la media y la desviación de ImageNet, que es lo que esperan los pesos preentrenados. Leemos JPEG, PNG o DICOM; en DICOM la intensidad se lleva a 0–255 con su mínimo y su máximo.
5. **Aumentos, solo en entrenamiento.** Volteo horizontal y vertical, rotación de hasta ±10° y brillo/contraste ±10 %. No cambian la patología, y los dejamos suaves a propósito: un giro grande o mucho contraste podrían borrar las calcificaciones. Validación y prueba nunca se alteran.

¿Por qué tanto cuidado con la partición? Porque la vista CC y la MLO de una lesión son casi la misma imagen. Si una queda en entrenamiento y la otra en prueba, es como dejar que el estudiante vea el examen antes de presentarlo: la nota sale alta, pero no mide nada.

## 4. Modelado

**Cuántos modelos entrenamos.** Hicimos **cinco corridas con dos arquitecturas** y una corrida piloto de una época con 200 imágenes, solo para comprobar que el código funcionaba en Colab (no se usó para nada más). Todas comparten la partición, la semilla 42, la pérdida `BCEWithLogitsLoss`, el optimizador AdamW (*weight decay* 1e-4), una tasa de aprendizaje con decaimiento coseno, lotes de 32, imágenes de 224 px, precisión mixta (AMP) y parada temprana por el ROC-AUC de validación, guardando los pesos de la mejor época. Corrimos todo en Google Colab con una GPU Tesla T4 (15 GB), PyTorch 2.14 y Python 3.13.

| Corrida | Modelo | Qué la distingue | Datos | Épocas máx. (mejor; paró en) | lr | Tiempo |
|---|---|---|---|---|---|---|
| R0 | CNN pequeña (4 bloques, 98.049 parámetros) | línea base desde cero | 2.527 | 25 (5; 10) | 1e-3 | 3,7 min |
| R1 | ResNet18 preentrenada (11.177.025 parámetros) | con aumentos | 2.527 | 15 (12; 15) | 1e-4 | 5,8 min |
| R2 | ResNet18 preentrenada | **sin** aumentos (experimento controlado) | 2.527 | 15 (1; 6) | 1e-4 | 1,9 min |
| R3 | ResNet18 preentrenada | más épocas: hasta 30, paciencia 7 | 2.527 | 30 (10; 17) | 1e-4 | 6,8 min |
| R4 | ResNet18 preentrenada | más datos: entrenamiento + validación | 3.048 | 10 fijas (de R3) | 1e-4 | 3,8 min |

**R0, el piso.** Una red pequeña entrenada desde cero nos dice cuánto aporta la transferencia de aprendizaje. Si ResNet18 no la superaba con claridad, algo andaba mal en el pipeline.

**El experimento controlado (R1 contra R2).** Lo único que cambia es la bandera `--aug`. Modelo, pesos iniciales, datos, semilla, lr, épocas y lote son iguales, así que cualquier diferencia se puede atribuir a los aumentos y no a una mezcla de cambios.

**Lo que vimos al entrenar (estabilidad).** R0 fue la más inestable. Con lr = 1e-3 su AUC de validación se movió entre 0,67 y 0,69 y en la época 8 se desplomó a 0,47, cuando el modelo empezó a llamar benignas a todas las lesiones; la parada temprana conservó la época 5 (0,694). R2 memorizó rapidísimo: la pérdida de entrenamiento bajó de 0,595 a 0,028 en seis épocas mientras la de validación subía de 0,567 a 0,906, y su mejor AUC fue el de la primera época. R1 fue la más estable: tuvo un bache en la época 2 (0,72), luego subió de forma sostenida y en las últimas cuatro épocas se quedó entre 0,812 y 0,819, cuando el coseno ya había bajado la tasa de aprendizaje. Un detalle que nos sirvió para confiar en la semilla: R1 y R3 dan exactamente la misma pérdida en la primera época (0,6138) y solo se separan cuando su tasa de aprendizaje empieza a bajar a ritmos distintos.

![Figura 3. Pérdida de entrenamiento y de validación por época: R1 con aumentos (izquierda) y R2 sin aumentos (derecha).](figuras/curvas_r1_r2.png)

### 4.1 Costo computacional: memoria, GPU y tiempos

Queríamos saber si el modelo es pesado de entrenar y de poner en producción. Medimos con `src/medir_recursos.py` (unos pasos de entrenamiento con imágenes reales, sin tocar validación ni prueba). En GPU usamos una T4 de Colab; en CPU, el portátil de uno de nosotros (AMD Ryzen 7 7435HS, 4 hilos).

| Modelo | Parámetros | Pesos | Memoria GPU al entrenar (lote 32, AMP) | Paso de entrenamiento en GPU | Predicción en CPU (1 imagen) | Memoria extra en CPU |
|---|---|---|---|---|---|---|
| CNN pequeña (R0) | 98 mil | 0,4 MB | 354 MB | 36 ms | 12 ms | 10 MB |
| ResNet18 (R1–R4) | 11,2 millones | 42,6 MB | 572 MB | 51 ms | 53 ms | 57 MB |
| EfficientNet-B0 (solo medida, no entrenada) | 4,0 millones | 15,3 MB | 1.458 MB | 92 ms | 45 ms | 39 MB |

Lo que sacamos de esto:

- **La GPU no fue el cuello de botella.** Entrenar ResNet18 con lotes de 32 ocupó como máximo 572 MB de los 15 GB de la T4 (menos del 4 %), así que nos sobró memoria de lejos; podríamos haber usado lotes mucho más grandes o imágenes de más resolución. Lo curioso es EfficientNet-B0: tiene casi tres veces menos parámetros que ResNet18, pero gasta 2,5 veces más memoria al entrenar y cada paso tarda casi el doble, porque guarda muchas más activaciones intermedias. El número de parámetros no lo dice todo. En GPU, ResNet18 procesa unas 630 imágenes por segundo al entrenar, así que una época de 2.527 imágenes necesita unos 4 s de cálculo. Pero cada época real tardó unos 22 s, y casi lo mismo con la red pequeña que con ResNet18 (`log.csv`), aunque una tiene 100 veces más parámetros que la otra. O sea que cerca del 80 % del tiempo se iba en **leer, decodificar y aumentar los JPEG** con solo 2 procesos de carga, no en calcular: la GPU pasaba la mayor parte del tiempo esperando. Entrenar R1 completa tomó menos de seis minutos.
- **En producción alcanza con CPU.** Una predicción de ResNet18 tarda unos 53 ms en el portátil. Python con PyTorch ya ocupa unos 320 MB antes de cargar el modelo, y el modelo con una predicción suma unos 57 MB más. La app de Gradio completa, con el modelo cargado, ocupó unos 510 MB de RAM. Por eso no hizo falta pagar un servidor con GPU (sección 6).
- **EfficientNet-B0** es más liviana (4 millones de parámetros) y en CPU va un poco más rápido, pero no la entrenamos: el enunciado pedía una línea base y una red con transferencia, y con el tiempo que teníamos preferimos hacer bien el experimento controlado con una sola arquitectura. Queda como siguiente paso natural.

## 5. Evaluación

### 5.1 El umbral se elige en validación

La salida del modelo es una **puntuación** (la sigmoide de un logit), no una probabilidad calibrada: un 0,7 no quiere decir "70 % de probabilidad de cáncer", solo que el modelo pone esa lesión por encima de una con 0,6. El umbral se fija **solo con validación** (`evaluate.py --strategy sens --target-sens 0.90`) y se guarda en `threshold.json`. Para R1 quedó en **0,066**. Que el corte esté tan lejos de 0,5 confirma lo anterior: las puntuaciones de R1 se amontonan cerca de cero.

### 5.2 ¿Cuál modelo fue el mejor?

Validación: 521 lesiones (222 malignas y 299 benignas) de 225 pacientes. Cada corrida se evalúa con su propio umbral para sensibilidad ≥ 0,90, así que todas atrapan más o menos las mismas malignas y la diferencia está en cuántas benignas marcan de más.

| Corrida | ROC-AUC | Umbral | Sensibilidad | Especificidad | Precisión | F1 | FN | FP |
|---|---|---|---|---|---|---|---|---|
| R0 CNN pequeña | 0,694 | 0,378 | 0,905 | 0,361 | 0,513 | 0,655 | 21 | 191 |
| **R1 ResNet18 + aumentos** | **0,819** | 0,066 | 0,901 | **0,518** | **0,581** | **0,707** | 22 | **144** |
| R2 ResNet18 sin aumentos | 0,782 | 0,156 | 0,901 | 0,418 | 0,535 | 0,671 | 22 | 174 |
| R3 ResNet18, hasta 30 épocas | 0,812 | 0,068 | 0,901 | 0,498 | 0,571 | 0,699 | 22 | 150 |

**Ganó R1.** Tiene el mayor ROC-AUC y, con la misma sensibilidad que las demás, es la que menos falsas alarmas da (144 benignas marcadas, frente a 150 de R3, 174 de R2 y 191 de R0). La transferencia de aprendizaje le ganó a la red desde cero por 0,125 de AUC, y los aumentos sumaron 0,037 más. Entre las cuatro ResNet18, R1 y R3 quedaron prácticamente empatadas (0,819 contra 0,812). Con 521 lesiones de validación y una sola semilla, una diferencia de 0,007 cabe dentro del ruido, así que la lectura honesta es que **entrenar el doble no ayudó**, no que R1 sea mucho mejor. R4 no se puede comparar en validación, porque esas imágenes están dentro de su entrenamiento. Elegimos R1 sin mirar la prueba.

### 5.3 La prueba reservada, una sola vez

Con el modelo y el umbral ya fijos, evaluamos la prueba **una sola vez**, el 18-nov a las 11:53 (queda constancia en `registro_uso_test.txt`). Son 518 lesiones (200 malignas y 318 benignas) de 226 pacientes que el modelo nunca vio.

| | Predice benigna | Predice maligna |
|---|---|---|
| **Es benigna** (318) | VN = 146 | FP = 172 |
| **Es maligna** (200) | FN = 22 | VP = 178 |

| Sensibilidad maligna | Especificidad | Precisión | F1 | ROC-AUC | Exactitud |
|---|---|---|---|---|---|
| **0,890** | 0,459 | 0,509 | 0,647 | 0,788 | 0,625 |

En palabras: de cada 100 lesiones malignas, el modelo deja pasar 11 como benignas; a cambio, marca como sospechosas a más de la mitad de las benignas (172 de 318). Todo bajó un poco respecto a validación (AUC 0,819 → 0,788; sensibilidad 0,901 → 0,890; especificidad 0,518 → 0,459), que es lo normal cuando modelo y umbral se eligieron mirando validación.

![Figura 4. Matriz de confusión y curva ROC de R1 en la prueba reservada.](figuras/test_matriz_roc.png)

**¿Qué pasa si se mueve el umbral?** (validación, R1, `tabla_umbrales.csv`)

| Umbral | Sensibilidad | Especificidad | FN (de 222) | FP (de 299) |
|---|---|---|---|---|
| 0,5 | 0,604 | 0,853 | 88 | 44 |
| 0,2 | 0,802 | 0,696 | 44 | 91 |
| **0,066 (elegido)** | **0,901** | **0,518** | **22** | **144** |

Pasar de 0,2 a 0,066 atrapa 22 cánceres más a cambio de 53 falsas alarmas más, unas 2,4 por cada cáncer rescatado. Es el precio que aceptamos para que un falso negativo sea raro.

**¿Se cumplieron las hipótesis?** **H1, en parte.** Los aumentos subieron el AUC (+0,037) y retrasaron la mejor época (12 contra 1), pero la forma en que habíamos decidido medir el sobreajuste (la brecha de pérdidas en la mejor época de cada modelo) no lo mostró, porque la mejor época de R2 fue la primera, antes de empezar a memorizar. Comparadas en la misma época (la 6), la brecha es 0,11 en R1 contra 0,88 en R2: el efecto está, pero la métrica que escogimos fue mala y lo dejamos escrito. **H2, sí:** el umbral quedó muy por debajo de 0,5 (0,066), la especificidad de validación bajó a 0,518 y la sensibilidad en prueba fue 0,890, dentro del rango 0,80–0,90 que habíamos predicho.

### 5.4 Lo que pidió el profesor en el día 7: más épocas, más datos e imágenes nuevas

En la revisión del día 7 el profesor nos pidió entrenar más, con más datos, y probar el modelo final con imágenes que nunca hubiera visto. Lo hicimos el 24-nov (commit `56148a4`):

- **Más épocas (R3).** La mejor época fue la 10 (AUC 0,812) y después el modelo empezó a memorizar: la pérdida de validación subió de 0,59 a 0,87 mientras la de entrenamiento seguía bajando. R1 ya estaba cerca de su techo con 15 épocas.
- **Más datos (R4).** Entrenamos con entrenamiento + validación (3.048 lesiones, +21 %) durante las 10 épocas que eligió R3, y tomamos el umbral de R3. No repartimos de nuevo los datos, porque habría tocado la prueba ya usada.
- **Imágenes nuevas: mini-MIAS.** Es otro conjunto (Reino Unido), con otro escáner, otra población y otra resolución. De sus 330 anotaciones quitamos 207 mamografías normales y 4 anomalías sin coordenadas, y quedaron **119 lesiones de 104 pacientes** (51 malignas y 68 benignas). Recortamos cada lesión con el centro y el radio de `Info.txt` (`src/prepare_mias.py`) y usamos el umbral de CBIS-DDSM sin reajustar nada.

| Modelo | ROC-AUC en mini-MIAS | Sensibilidad | Especificidad | FN | FP |
|---|---|---|---|---|---|
| R1 | 0,710 | 1,000 | 0,029 | 0 | 66 |
| R3 | 0,725 | 1,000 | 0,015 | 0 | 67 |
| R4 (train + val) | 0,714 | 1,000 | 0,015 | 0 | 67 |

El modelo conserva parte de su capacidad de **ordenar** las lesiones (AUC 0,71–0,73, frente a 0,788 en la prueba de CBIS-DDSM), pero con el nuevo escáner sus puntuaciones se corren hacia arriba y, con el umbral de CBIS-DDSM, marca casi todo como maligno. Es como una balanza que pesa bien pero quedó mal calibrada en otra cocina: el orden de los pesos se mantiene, el cero no. Ni más épocas ni más datos del mismo dominio lo arreglan, porque el problema es el **cambio de dominio** y no la cantidad. Para usar el modelo con otro equipo habría que recalibrar el umbral con datos de ese equipo, separados de los que se usen para evaluar. Para nosotros es el hallazgo más importante del proyecto.

### 5.5 Análisis de errores

**Método, fijado antes de mirar los casos** (`docs/analisis_errores.md`): en validación tomamos los falsos negativos de puntuación más baja y los falsos positivos de puntuación más alta (mínimo 4 de cada uno) y medimos la tasa de error por tipo de lesión, `subtlety` y `assessment` con `src/errores_por_grupo.py`. Así no escogemos a mano los casos "bonitos".

| Grupo (R1, validación, umbral 0,066) | Lesiones | FN | FP | % de malignas que se escapan | % de benignas marcadas malignas |
|---|---|---|---|---|---|
| Calcificaciones | 273 | 14 | 69 | 13,1 | 41,6 |
| Masas | 248 | 8 | 75 | 7,0 | 56,4 |
| `assessment` 2 (BI-RADS benigno) | 99 | 0 | 28 | — | 28,3 |
| `assessment` 4 (sospechoso) | 257 | 16 | 80 | 12,8 | 60,6 |

En la prueba se repite el patrón: **17 de los 22 cánceres que se escapan son calcificaciones** (sensibilidad 0,83 en calcificaciones frente a 0,95 en masas). Y las benignas BWC casi no engañan al modelo (marca 31 de 114), mientras que las `BENIGN` que sí generaron seguimiento sí lo hacen (141 de 204). Con estas últimas el AUC baja a 0,73, así que el 0,788 global es algo optimista.

**Qué tienen en común los errores.** Los cánceres que se escapan no son los sutiles: varios tenían `subtlety` 4 y BI-RADS 4–5, o sea que el radiólogo los vio claros. Lo más probable es que los puntos finos de las calcificaciones se pierdan al reducir el recorte a 224 px. Los falsos positivos se concentran en benignas que también le parecieron sospechosas al radiólogo (con `assessment` 4 se marca el 61 %), así que muchas de las "falsas alarmas" son casos difíciles de verdad. Para reducirlos probaríamos más resolución o recortes en mosaico para las calcificaciones.

| Error | Caso | Puntuación | Tipo | Hipótesis |
|---|---|---|---|---|
| FN | P_00479_LEFT_CC_calcification_1 | 0,0002 | calcificación | puntos finos que se pierden al reducir a 224 px |
| FN | P_00644_LEFT_CC_mass_1 | 0,0076 | masa | masa grande de borde definido; el recorte no muestra el tejido alrededor |
| FP | P_00250_LEFT_MLO_calcification_1 | 0,992 | calcificación | recorte angosto, deformado al llevarlo a 224 × 224 |
| FP | P_00120_LEFT_MLO_mass_1 | 0,956 | masa | masa muy densa que satura el recorte |

## 6. Interfaz y despliegue

**La app (Gradio, `app.py`).** Recibe un recorte en PNG, JPG o DICOM y devuelve la imagen preprocesada (224 × 224), la puntuación del modelo, el umbral (por defecto el de validación, 0,066, con un deslizador de paso 0,001 para explorar) y la clase. El aviso «Prototipo académico; no apto para diagnóstico ni decisiones clínicas» está siempre visible, junto con la nota de que la puntuación no es una probabilidad. Rechaza con un mensaje claro los formatos no admitidos, los archivos que no son imagen, los de más de 50 MB y las imágenes de menos de 32 px. Si el lado corto de la imagen pasa de 2.893 px (el recorte más grande que vimos en el EDA), responde pero advierte que parece una **mamografía completa** y que el resultado puede no valer. Decidimos advertir en vez de rechazar para no bloquear recortes grandes legítimos.

**Disponible las 24 horas.** Un enlace de Colab muere cuando se cierra la sesión, así que publicamos el modelo en dos sitios que no dependen de ningún computador nuestro:

1. **Hugging Face Spaces:** https://huggingface.co/spaces/JostGreen/mamografias-g8-web
2. **GitHub Pages (copia de respaldo):** https://joseg-sotomendez.github.io/mamografias_g8/

Las dos ejecutan el mismo modelo R1 exportado a ONNX **dentro del navegador**, con el mismo preprocesamiento (escala de grises, redimensionado bilineal igual al de Pillow y normalización de ImageNet), el mismo umbral de validación y las mismas validaciones de la app. Como son páginas estáticas, no tienen un servidor que se duerma o se caiga, y la imagen nunca sale del equipo de quien la usa. Antes de publicarlas comprobamos que dan la misma puntuación que PyTorch: la diferencia máxima entre PyTorch y ONNX fue de 1e-6, y en el navegador las imágenes de prueba dieron las mismas tres cifras decimales que en la app de Gradio (por ejemplo, 0,008 para una benigna, en unos 250 ms). La app de Gradio sigue en el repositorio (`python app.py --run runs/R1_resnet18_aug`) y quedó empaquetada para un Space de Gradio (`deploy/`), pero Hugging Face cobra ese tipo de Space a las cuentas nuevas, así que la publicamos de esta otra forma.

Además, un trabajo de GitHub Actions revisa cada 6 horas que las dos páginas y el archivo del modelo respondan; si alguna falla, nos llega un correo.

![Figura 5. La app de Gradio con un recorte maligno (izquierda) y con una mamografía completa, donde muestra la advertencia (derecha).](figuras/capturas_app.png)

## 7. Límites

- **Tamaño de muestra.** Son 2.527 recortes de entrenamiento para una red de 11 millones de parámetros, y 521 de validación: diferencias de 0,01–0,02 de AUC pueden ser ruido. Además usamos una sola semilla, así que no medimos cuánto cambian los resultados de una corrida a otra.
- **Sesgo de selección.** Solo hay lesiones ya anotadas: el modelo nunca vio una mamografía normal ni tiene que encontrar la lesión. Que una lesión no esté anotada no significa que no haya cáncer.
- **Generalización.** Son placas escaneadas de Estados Unidos de los noventa. En mini-MIAS el AUC bajó a 0,71 y el umbral dejó de servir. Nada de esto vale para mamografía digital ni para pacientes colombianas sin validarlo con datos locales.
- **Interpretación clínica.** La puntuación no está calibrada y no es un riesgo. Con el umbral elegido se escapan 11 de cada 100 malignas, sobre todo calcificaciones: un «benigna» del modelo no descarta nada. Las métricas son por recorte, no por paciente.
- **Interfaz.** El deslizador deja mover el umbral sin protección; la app no explica en qué parte de la imagen se fijó el modelo (no hicimos Grad-CAM) y no sabe decir «esto no es una mamografía»: solo revisa formato y tamaño.

## 8. Conclusiones

1. Partir por paciente fue la decisión más importante: con 0 pacientes compartidas, las cifras de validación y prueba no están infladas por fuga.
2. La transferencia de aprendizaje funciona con pocos datos: ResNet18 preentrenada le ganó a la red desde cero por 0,125 de AUC, y los aumentos le sumaron 0,037 y frenaron la memorización.
3. Con el umbral elegido en validación, el modelo atrapa 89 de cada 100 malignas en la prueba, pero marca como sospechosas a más de la mitad de las benignas: es un filtro muy sensible, no un clasificador fino.
4. Ni más épocas ni más datos del mismo dominio mejoraron el modelo. Las imágenes de otro escáner mostraron el límite real: el modelo ordena algo, pero su umbral no se transfiere.
5. El modelo es liviano: entrena en menos de seis minutos en una GPU de Colab, predice en unos 50 ms en un portátil y corre incluso dentro del navegador, así que lo pudimos publicar 24/7 sin pagar un servidor con GPU.
6. Con más tiempo probaríamos más resolución para las calcificaciones, Grad-CAM, EfficientNet-B0, varias semillas y una calibración de la puntuación en un conjunto aparte.

**Referencias.** Lee R. S. et al. (2017), *A curated mammography data set for use in computer-aided detection and diagnosis research*, Scientific Data 4:170177 · CBIS-DDSM, The Cancer Imaging Archive, DOI 10.7937/K9/TCIA.2016.7O02S9CY · Suckling J. et al. (1994), *The Mammographic Image Analysis Society digital mammogram database* (mini-MIAS) · He K. et al. (2016), *Deep Residual Learning for Image Recognition*, CVPR.

**Anexos** (documento aparte): tabla de etiquetas, hipótesis, bitácora, registro de uso de IA, análisis de errores completo y evidencia del control del día 7. Código, partición y comandos: https://github.com/JoseG-SotoMendez/mamografias_g8
