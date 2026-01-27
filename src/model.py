import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn(input_shape=(224, 224, 1)):
    model = models.Sequential(name="Pneumonia_CNN")

    # Block 1
    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                            input_shape=input_shape))
    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 2
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 3
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))

    # Classification head
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))

    return model
