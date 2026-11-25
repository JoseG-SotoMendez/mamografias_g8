---
title: Lesiones mamográficas G8
emoji: 🩻
colorFrom: pink
colorTo: gray
sdk: gradio
sdk_version: 6.32.0
python_version: "3.11"
app_file: app.py
pinned: false
license: cc-by-3.0
short_description: Lesiones benigna/maligna con ResNet18 (académico)
---

# Clasificación de lesiones mamográficas — Grupo 8

**⚠️ Prototipo académico; no apto para diagnóstico ni decisiones clínicas.**

Proyecto de Aprendizaje Computacional (Ingeniería de Sistemas). Clasifica **recortes de lesión (ROI)** de mamografía
como benignos o malignos con una ResNet18 preentrenada en ImageNet y ajustada con CBIS-DDSM (corrida R1).
El umbral por defecto (0,066) se eligió en validación para atrapar al menos el 90 % de las malignas.

- Entrada: recorte de lesión en PNG, JPG o DICOM. No sirve para mamografías completas.
- Salida: puntuación del modelo (no es una probabilidad calibrada), umbral y clase.
- Resultado en la prueba reservada (518 lesiones, una sola vez): sensibilidad 0,890, especificidad 0,459, ROC-AUC 0,788.

Código, partición, informe y bitácora: https://github.com/JoseG-SotoMendez/mamografias_g8

Datos: CBIS-DDSM (The Cancer Imaging Archive, licencia CC BY 3.0, DOI 10.7937/K9/TCIA.2016.7O02S9CY).
