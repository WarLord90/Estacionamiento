import tensorflow as tf

# Creamos una matriz 2x3
m = tf.constant([[1, 2, 3],
                 [4, 5, 6]])
print("Matriz:\n", m)

# Seleccionar una fila
print("Primera fila:", m[0])     # [1 2 3]
print("Segunda fila:", m[1])     # [4 5 6]

# Seleccionar una columna
print("Primera columna:", m[:, 0])   # [1 4]
print("Última columna:", m[:, -1])   # [3 6]

# Seleccionar un elemento (fila 1, col 2)
print("Elemento (1,2):", m[0, 1])    # 2

# Submatriz (primeras 2 filas, columnas 1 y 2)
print("Submatriz:\n", m[:2, :2])

# Tensor 3D
t3d = tf.constant([[[1, 2, 3],
                    [4, 5, 6]],
                   [[7, 8, 9],
                    [10, 11, 12]]])
print("\nTensor 3D:\n", t3d)

# Seleccionar la primera "capa"
print("Primera capa:\n", t3d[0])

# Seleccionar la segunda capa, segunda fila
print("Segunda capa, segunda fila:", t3d[1, 1])
