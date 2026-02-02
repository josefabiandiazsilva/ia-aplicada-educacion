import numpy as np

# Entradas : En este caso se reciben 3 valores que pueden ser
# de cualquier tipo sensores o pixeles.
x = np.array([0.8, 0.4, 0.3])

# Pesos que se asignan a cada una de las entradas del primer grupo
w = np.array([0.5, -0.2, 0.1])

# Sesgo que aplicaré en el cálculo final.
b = 0.05

# Función de activación ReLU -Rectified Linear Unit
#Se asegura de no activar (0) si es negativo.
def relu(z):
    return max(0, z)

# Neurona que calcula pesos y adiciona sesgo
#Tambien indica activación.
#Utiliza dot (multiplica vectores y suma)
z = np.dot(x, w) + b
y = relu(z)

print("Salida de la neurona:", y)
