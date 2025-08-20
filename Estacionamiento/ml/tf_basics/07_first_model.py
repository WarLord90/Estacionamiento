import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # menos ruido en consola

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("TF:", tf.__version__)

# 1) Datos: x → y = 2x - 1
x = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.float32)
y = 2 * x - 1  # [-7, -5, -3, -1, 1, 3, 5]

# 2) Modelo: una neurona (Dense(1)) dentro de un Sequential
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

# 3) Compilar: decirle cómo aprender (optimizador) y qué objetivo minimizar (loss)
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
              loss='mse')  # mse = error cuadrático medio

# 4) Entrenar (aprender de los ejemplos)
hist = model.fit(x, y, epochs=200, verbose=0)

# Graficar la pérdida
plt.plot(hist.history['loss'])
plt.title("Curva de pérdida durante el entrenamiento")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.grid(True)
plt.show()

# 5) Evaluar y predecir
loss = model.evaluate(x, y, verbose=0)
print("Loss final:", loss)

# La red aprendió w (peso) y b (sesgo). Deberían acercarse a 2 y -1
w, b = model.layers[0].get_weights()
print("Peso (w):", w.flatten()[0])
print("Sesgo (b):", b[0])

# Probar con valores nuevos
x_test = np.array([4, 5, 10], dtype=np.float32)  # y real sería 7, 9, 19
pred = model.predict(x_test, verbose=0).flatten()
for xi, yi in zip(x_test, pred):
    print(f"x={xi:.0f} → pred={yi:.3f}")
