import tensorflow as tf

print("TF version:", tf.__version__)

# Escalar
escalar = tf.constant(5)
print("Escalar:", escalar, "shape:", escalar.shape, "dtype:", escalar.dtype)

# Vector
vector = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32) 
print("Vector:", vector, "shape:", vector.shape, "dtype:", vector.dtype)

# Matriz
matriz = tf.constant([[1, 2], [3, 4]])
print("Matriz:", matriz, "shape:", matriz.shape, "dtype:", matriz.dtype)

# Tensor 3D
tensor3d = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print("Tensor3D:", tensor3d, "shape:", tensor3d.shape, "dtype:", tensor3d.dtype)
