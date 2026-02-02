# Red neuronal densa desde cero (DNN)

Este ejemplo implementa una red neuronal densa simple utilizando únicamente
NumPy, con el objetivo de ilustrar el funcionamiento interno de una red
multicapa sin depender de frameworks de alto nivel.

## Arquitectura
- Capa de entrada: 3 valores
- Capa oculta 1: 4 neuronas (ReLU)
- Capa oculta 2: 3 neuronas (ReLU)
- Capa de salida: 1 neurona (ReLU)

## Componentes
- Pesos definidos manualmente
- Sesgos por capa
- Propagación hacia adelante (forward pass)
- Función de activación ReLU

## Objetivo pedagógico
Facilitar la comprensión de cómo se procesan los datos dentro de una red
neuronal profunda, destacando la importancia de la estructura, los pesos
y las funciones de activación.

Este ejemplo no incluye entrenamiento ni retropropagación, ya que su
propósito es exclusivamente conceptual y educativo.

## Ejecución
```bash
python dnn_desde_cero.py
