"""
Implementación educativa de una red neuronal densa (DNN) usando NumPy.

Este ejemplo tiene como objetivo ilustrar el funcionamiento interno de una
red neuronal multicapa, mostrando de forma explícita:
- Entradas
- Pesos
- Sesgos
- Propagación hacia adelante (forward pass)
- Función de activación ReLU

No incluye entrenamiento ni retropropagación, ya que su propósito es
conceptual y pedagógico.
"""

import numpy as np

# -------------------------
# Funciones de activación
# -------------------------
def relu(x):
    """Función de activación ReLU (Rectified Linear Unit)."""
    return np.maximum(0, x)

# -------------------------
# Entradas
# -------------------------
# Vector de entrada con 3 valores
x = np.array([0.8, 0.4, 0.3])

# -------------------------
# Capa oculta 1 (3 entradas -> 4 neuronas)
# -------------------------
W1 = np.array([
    [0.2, -0.1, 0.4, 0.6],
    [0.5,  0.3, 0.1, -0.2],
    [-0.4, 0.2, 0.7, 0.1]
])

b1 = np.array([0.1, -0.2, 0.05, 0.3])

z1 = np.dot(x, W1) + b1
a1 = relu(z1)

print("Salida capa oculta 1:", a1)

# -------------------------
# Capa oculta 2 (4 entradas -> 3 neuronas)
# -------------------------
W2 = np.array([
    [0.3, 0.6, -0.1],
    [-0.2, 0.1, 0.4],
    [0.5, -0.3, 0.2],
    [0.1, 0.2, 0.5]
])

b2 = np.array([0.05, -0.1, 0.2])

z2 = np.dot(a1, W2) + b2
a2 = relu(z2)

print("Salida capa oculta 2:", a2)

# -------------------------
# Capa de salida (3 entradas -> 1 neurona)
# -------------------------
W3 = np.array([0.4, -0.2, 0.3])
b3 = 0.1

z3 = np.dot(a2, W3) + b3
y = relu(z3)

print("\nSalida final de la red (DNN):", y)
