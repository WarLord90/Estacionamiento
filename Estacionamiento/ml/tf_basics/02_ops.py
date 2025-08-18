import tensorflow as tf

# Dos tensores (vectores)
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])

# Operaciones básicas
print("a:", a)
print("b:", b)
print("a + b =", a + b)      # suma
print("a * b =", a * b)      # multiplicación elemento a elemento
print("a ** 2 =", tf.pow(a, 2))  # elevar al cuadrado


m = tf.constant([[1, 2, 3],
                 [4, 5, 6]])   # forma (2,3)

v = tf.constant([10, 20, 30])  # forma (3,)

print(m + v)