import os
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape=(32, 32, 1), num_classes=10) -> keras.Model:
    """
    Construye una CNN sencilla para reconocimiento de dígitos (0-9).

    - Entrada: imágenes 32x32 en escala de grises, normalizadas a [0,1]
    - Salida: logits de 10 clases
    """
    inputs = keras.Input(shape=input_shape, name="input")

    # Aseguramos tamaño 32x32 por si se cargan datos con otra resolución
    x = layers.Resizing(32, 32, name="resize")(inputs)

    # Bloque de convoluciones
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_cnn_32x32")
    return model


def load_and_preprocess_data(img_size=32):
    """
    Carga MNIST desde keras.datasets, redimensiona a 32x32 y normaliza a [0,1].
    Aplica data augmentation mediante capas de aumento de datos en el modelo
    (más sencillo de mantener que generadores separados).
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Originalmente 28x28, añadimos canal y redimensionamos a 32x32
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Añadir canal
    x_train = np.expand_dims(x_train, -1)  # (N, 28, 28, 1)
    x_test = np.expand_dims(x_test, -1)

    # Redimensionar a 32x32 usando capas de Keras en tf.data sería posible,
    # pero aquí lo hacemos directamente con tf.image para dejar claro el tamaño.
    x_train = tf.image.resize(x_train, (img_size, img_size)).numpy()
    x_test = tf.image.resize(x_test, (img_size, img_size)).numpy()

    return (x_train, y_train), (x_test, y_test)


def build_augment_layer():
    """
    Capa de data augmentation.
    Para dígitos conviene hacer pequeños giros, zoom y traslaciones.
    """
    data_augmentation = keras.Sequential(
        [
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.1, 0.1),
            layers.RandomTranslation(0.1, 0.1),
        ],
        name="data_augmentation",
    )
    return data_augmentation


def train_model(
    batch_size=128,
    epochs=10,
    img_size=32,
    model_path: Optional[Path] = None,
):
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data(img_size=img_size)

    num_classes = 10

    # Capa de aumento de datos delante del modelo base
    data_augmentation = build_augment_layer()
    base_model = build_model(input_shape=(img_size, img_size, 1), num_classes=num_classes)

    inputs = keras.Input(shape=(img_size, img_size, 1), name="input")
    x = data_augmentation(inputs)
    outputs = base_model(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_cnn_32x32_aug")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    if model_path is not None:
        model.save(model_path)
        print(f"Modelo guardado en: {model_path}")

    return model


def export_to_onnx(model: keras.Model, onnx_path: Path):
    """
    Exporta el modelo Keras a ONNX usando tf2onnx.

    El modelo espera tensores de entrada con forma [batch, 32, 32, 1]
    y valores float32 en [0,1].
    """
    import tf2onnx

    spec = (tf.TensorSpec((None, 32, 32, 1), tf.float32, name="input"),)
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=13,
    )

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with open(onnx_path, "wb") as f:
        f.write(model_proto.SerializeToString())

    print(f"Modelo ONNX exportado a: {onnx_path}")


def main():
    base_dir = Path(__file__).resolve().parent
    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)

    keras_path = models_dir / "mnist_cnn_32x32.h5"
    onnx_path = models_dir / "mnist_cnn_32x32.onnx"

    # Entrenar modelo (o cargar si ya existe)
    if keras_path.exists():
        print(f"Cargando modelo existente desde {keras_path}")
        model = keras.models.load_model(keras_path)
    else:
        model = train_model(model_path=keras_path)

    # Exportar a ONNX
    export_to_onnx(model, onnx_path)


if __name__ == "__main__":
    # Para evitar que TensorFlow se coma toda la GPU en algunos entornos:
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    main()

