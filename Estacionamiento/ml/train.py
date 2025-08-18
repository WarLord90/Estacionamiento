# -------- IMPORTS --------
import tensorflow as tf
from tensorflow import keras
layers = keras.layers
models = keras.models
import tf2onnx

# -------- PARÁMETROS --------
IMG_SIZE = (224, 224)
BATCH = 8         # con pocos datos, mejor batches pequeños
EPOCHS = 10
SEED = 123

print("✅ TensorFlow versión:", tf.__version__)

# -------- DATOS: CARGA Y SPLIT 80/20 --------
# Lee imágenes desde ../data/train y separa automáticamente train/val SIN duplicar
print("📦 Cargando dataset desde ../data/train (split 80/20)...")
train = keras.utils.image_dataset_from_directory(
    "../data/train",
    validation_split=0.2, subset="training", seed=SEED,
    image_size=IMG_SIZE, batch_size=BATCH, label_mode="binary", shuffle=True,
)
val = keras.utils.image_dataset_from_directory(
    "../data/train",
    validation_split=0.2, subset="validation", seed=SEED,
    image_size=IMG_SIZE, batch_size=BATCH, label_mode="binary", shuffle=True,
)

# Muestra las clases detectadas (de los nombres de carpeta)
class_names = train.class_names
print("🧾 Clases detectadas:", class_names)

# Prefetch para rendimiento
AUTOTUNE = tf.data.AUTOTUNE
train = train.cache().prefetch(AUTOTUNE)
val = val.cache().prefetch(AUTOTUNE)

# -------- DATA AUGMENTATION --------
# Pequeñas transformaciones para robustecer el modelo
data_aug = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.08),
    layers.RandomContrast(0.1),
], name="data_aug")

# -------- MODELO (TRANSFER LEARNING) --------
# 1) Reescala de 0..255 a 0..1
# 2) Aumentación
# 3) MobileNetV2 (preentrenada en ImageNet) sin la cabeza final
# 4) Pooling global + Dropout
# 5) Capa final densa con 1 neurona (sigmoid) -> probabilidad de LIBRE (por convención)
print("🏗️ Construyendo el modelo (MobileNetV2 como base)...")
base = keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
)
base.trainable = False  # v1: congelamos la base, entrenamos solo la cabeza

model = models.Sequential([
    layers.Rescaling(1./255),
    data_aug,
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])

# Muestra la arquitectura
model.build((None,) + IMG_SIZE + (3,))
print("🧱 Resumen del modelo:")
model.summary()

# -------- COMPILAR Y ENTRENAR --------
# binary_crossentropy: porque hay 2 clases (binario)
# accuracy: métrica fácil de entender
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

cb = keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=3, restore_best_weights=True
)

print("🚀 Entrenando...")
history = model.fit(train, validation_data=val, epochs=EPOCHS, callbacks=[cb])

# -------- EXPORTAR A ONNX --------
# La entrada se llama "input" y es NHWC float32 (N, Alto, Ancho, Canales)
print("💾 Exportando a ONNX...")
spec = (tf.TensorSpec((None,) + IMG_SIZE + (3,), tf.float32, name="input"),)
tf2onnx.convert.from_keras(model, input_signature=spec,
                           output_path="../models/parking_free_vs_busy.onnx")
print("✅ ONNX listo en ../models/parking_free_vs_busy.onnx")

print("🎯 Umbral de decisión: >=0.5 = LIBRE; <0.5 = OCUPADO")
