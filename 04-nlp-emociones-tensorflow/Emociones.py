"""
Demo educativa: Clasificador simple de emociones (positivo / negativo / neutral)
usando TensorFlow/Keras.

 Nota importante:
- El dataset es intencionalmente pequeño y solo sirve para demostrar el flujo:
  tokenización -> padding -> embedding -> entrenamiento -> predicción.
- No es un modelo listo para producción ni tiene validez estadística.
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Reproducibilidad (opcional pero recomendado)
tf.random.set_seed(42)
np.random.seed(42)

# ---------------------------------------------------
# 1. Dataset demo (debería ser mucho más grande en un caso real)
# ---------------------------------------------------
sentences = [
    "Estoy muy feliz", "Me siento genial", "Que buen día",
    "Esto es horrible", "Estoy triste", "No me gusta esto",
    "Está bien", "Es normal", "No siento nada"
]

# 0 negativo, 1 positivo, 2 neutral
labels = [1, 1, 1,  0, 0, 0,  2, 2, 2]

# ---------------------------------------------------
# 2. Tokenización y padding
# ---------------------------------------------------
VOCAB_SIZE = 1000
MAXLEN = 5

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

sequences = tokenizer.texts_to_sequences(sentences)
X = pad_sequences(sequences, maxlen=MAXLEN)
y = np.array(labels)

# ---------------------------------------------------
# 3. Modelo
# ---------------------------------------------------
model = tf.keras.Sequential([
    layers.Embedding(input_dim=VOCAB_SIZE, output_dim=16, input_length=MAXLEN),
    layers.Flatten(),
    layers.Dense(32, activation="relu"),
    layers.Dense(3, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ---------------------------------------------------
# 4. Entrenamiento
# ---------------------------------------------------
def entrenar(epochs=30):
    print("\nEntrenando modelo (demo)...\n")
    model.fit(X, y, epochs=epochs, verbose=1)

# ---------------------------------------------------
# 5. Predicción
# ---------------------------------------------------
CLASES = ["negativo", "positivo", "neutral"]

def predecir(frase: str):
    seq = tokenizer.texts_to_sequences([frase])
    seq = pad_sequences(seq, maxlen=MAXLEN)

    pred = model.predict(seq, verbose=0)
    emocion = int(tf.argmax(pred, axis=1).numpy()[0])

    print("\n==============================")
    print("Frase:", frase)
    print("Vector entrada:", seq[0])
    print("Predicción (softmax):", pred[0])
    print("Emoción detectada:", CLASES[emocion])
    print("==============================\n")

# ---------------------------------------------------
# 6. Ejecución demo
# ---------------------------------------------------
if __name__ == "__main__":
    entrenar(epochs=30)

    predecir("Qué bonito día")
    predecir("Estoy muy molesto")
    predecir("No sé qué sentir")
