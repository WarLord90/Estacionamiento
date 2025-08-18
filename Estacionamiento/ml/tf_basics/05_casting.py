import tensorflow as tf

# Tensor entero (int32 por defecto)
i = tf.constant([1, 0, 1, 0])
print("Enteros:", i, "dtype:", i.dtype)

# Convertir a float32
f = tf.cast(i, tf.float32)
print("Convertido a float32:", f, "dtype:", f.dtype)

# Tensor de decimales
x = tf.constant([1.7, 2.2, 3.9], dtype=tf.float32)
print("\nDecimales:", x, "dtype:", x.dtype)

# Convertir a enteros (se truncan los decimales)
xi = tf.cast(x, tf.int32)
print("Convertido a int32:", xi, "dtype:", xi.dtype)

# Tensor booleano
b = tf.constant([True, False, True])
print("\nBooleanos:", b, "dtype:", b.dtype)

# Convertir booleanos a enteros
bi = tf.cast(b, tf.int32)
print("Booleanos a int32:", bi)

# Convertir booleanos a float
bf = tf.cast(b, tf.float32)
print("Booleanos a float32:", bf)
