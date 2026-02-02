"""
Implementación educativa de una red neuronal convolucional (CNN) básica
usando NumPy.

Este ejemplo ilustra paso a paso:
- Una imagen representada como una matriz
- La aplicación de un filtro (kernel)
- La operación de convolución 2D
- La generación de un feature map
- El proceso de flatten
- Una neurona final para producir la salida

No utiliza frameworks de deep learning y no incluye entrenamiento.
Su objetivo es exclusivamente conceptual y pedagógico.
"""

import numpy as np

# =========================================================
# 1. IMAGEN DE ENTRADA (matriz 4x4)
# =========================================================
# Imagen pequeña donde cada valor representa la intensidad de un píxel
imagen = np.array([
    [1, 2, 3, 0],
    [4, 5, 6, 1],
    [7, 8, 9, 2],
    [0, 1, 2, 3]
])

print("Imagen de entrada (4x4):")
print(imagen)
print()

# =========================================================
# 2. FILTRO (KERNEL)
# =========================================================
# Filtro 2x2 para detección simple de patrones (bordes)
filtro = np.array([
    [1, 0],
    [0, -1]
])

print("Filtro (Kernel 2x2):")
print(filtro)
print()

# =========================================================
# 3. FUNCIÓN DE CONVOLUCIÓN 2D
# =========================================================
def conv2d(img, kernel):
    """
    Realiza la operación de convolución 2D de forma manual:
    - Recorre la imagen por regiones
    - Multiplica región * kernel
    - Suma los resultados
    - Construye el feature map
    """
    h, w = img.shape
    kh, kw = kernel.shape

    salida = []

    for i in range(h - kh + 1):
        fila = []
        for j in range(w - kw + 1):
            region = img[i:i + kh, j:j + kw]
            valor = np.sum(region * kernel)
            fila.append(valor)
        salida.append(fila)

    return np.array(salida)

# =========================================================
# 4. FEATURE MAP
# =========================================================
feature_map = conv2d(imagen, filtro)

print("Feature map resultante:")
print(feature_map)
print()

# =========================================================
# 5. FLATTEN
# =========================================================
flatten = feature_map.flatten()

print("Vector flatten:")
print(flatten)
print()

# =========================================================
# 6. NEURONA FINAL
# =========================================================
# Pesos simples para ilustración
W = np.array([0.5] * len(flatten))
b = 0.1

salida = np.dot(flatten, W) + b

print("Salida final de la CNN:", salida)
