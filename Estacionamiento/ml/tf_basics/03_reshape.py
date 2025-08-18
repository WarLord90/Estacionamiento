import tensorflow as tf

# Crear un tensor con los números del 1 al 6
x = tf.range(1, 7)   # [1, 2, 3, 4, 5, 6]
print("x:", x, "shape:", x.shape)

# Convertirlo en una matriz de 2x3
x2 = tf.reshape(x, (2, 3))
print("x2 reshape (2,3):\n", x2)

# Convertirlo en una matriz de 3x2
x3 = tf.reshape(x, (3, 2))
print("x3 reshape (3,2):\n", x3)
