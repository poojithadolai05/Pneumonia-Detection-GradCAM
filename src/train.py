import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.model import build_cnn

# -------------------
# Configuration
# -------------------
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
DATA_DIR = "data/raw/chest_xray"
VERSION = "cnn_v1"

# -------------------
# Data generators
# -------------------
def get_data_generators():
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True
    )

    val_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_data = train_gen.flow_from_directory(
        f"{DATA_DIR}/train",
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True
    )

    val_data = val_gen.flow_from_directory(
        f"{DATA_DIR}/val",
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )

    return train_data, val_data


# -------------------
# Save training history
# -------------------
def save_history(history):
    os.makedirs("experiments/metrics", exist_ok=True)
    path = f"experiments/metrics/{VERSION}_history.json"

    with open(path, "w") as f:
        json.dump(history.history, f, indent=4)


# -------------------
# Training
# -------------------
def train():
    model = build_cnn()

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    train_data, val_data = get_data_generators()

    callbacks = [
        EarlyStopping(
            monitor="val_recall",
            patience=5,
            mode="max",
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath=f"experiments/models/{VERSION}_best.keras",
            monitor="val_recall",
            save_best_only=True,
            mode="max"
        )
    ]

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    save_history(history)


if __name__ == "__main__":
    train()
