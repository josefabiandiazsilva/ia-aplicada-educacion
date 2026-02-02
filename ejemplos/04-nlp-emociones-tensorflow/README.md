# Demo educativa: Clasificador simple de emociones (NLP) con TensorFlow/Keras

Este ejemplo implementa un **clasificador básico de emociones** a partir de texto,
con tres clases: **positivo**, **negativo** y **neutral**.

El objetivo es **mostrar el flujo completo** de un modelo NLP sencillo usando
TensorFlow/Keras, desde la preparación del texto hasta la predicción.

> **Nota importante**
> - El dataset es **intencionalmente pequeño** y solo sirve para demostrar el pipeline.
> - **No es un modelo listo para producción** ni tiene validez estadística.
> - Con datos reales se requiere un dataset amplio, validación adecuada y métricas robustas.

---

## ¿Qué aprenderás con este ejemplo?

- Cómo convertir texto en números mediante **tokenización**
- Cómo aplicar **padding** para igualar longitudes (secuencias)
- Cómo usar una capa **Embedding** para representar palabras
- Cómo entrenar un modelo simple con **softmax** y `sparse_categorical_crossentropy`
- Cómo ejecutar una predicción y leer el vector **softmax** (probabilidades)

---

## Dataset de ejemplo

Se usan frases cortas con su etiqueta asociada:

- `0` → negativo  
- `1` → positivo  
- `2` → neutral  

Este dataset es solo demostrativo. En un caso real, se debe:
- balancear clases,
- ampliar variedad lingüística,
- incluir expresiones regionales,
- separar entrenamiento/validación/prueba.

---

## Arquitectura del modelo

- **Embedding**: `VOCAB_SIZE=1000`, `output_dim=16`
- **Flatten**
- **Dense (ReLU)**: 32 neuronas
- **Dense (Softmax)**: 3 neuronas (3 clases)

Parámetros clave:
- `MAXLEN = 5` (longitud máxima de secuencia)
- Optimizador: `adam`
- Pérdida: `sparse_categorical_crossentropy`

---

## Requisitos

Crea o usa este archivo en la misma carpeta:

**`requirements.txt`**
```txt
tensorflow
numpy

