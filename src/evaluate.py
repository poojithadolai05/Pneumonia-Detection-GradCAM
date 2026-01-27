import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------------
# Configuration
# -------------------
IMG_SIZE = 224
BATCH_SIZE = 32
DATA_DIR = "data/raw/chest_xray"
VERSION = "cnn_v1"
MODEL_PATH = f"experiments/models/{VERSION}_best.keras"


# -------------------
# Load test data
# -------------------
def load_test_data():
    test_gen = ImageDataGenerator(rescale=1.0 / 255)

    test_data = test_gen.flow_from_directory(
        f"{DATA_DIR}/test",
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )

    return test_data


# -------------------
# Save evaluation metrics
# -------------------
def save_metrics(report, cm):
    os.makedirs("experiments/metrics", exist_ok=True)

    with open(f"experiments/metrics/{VERSION}_report.json", "w") as f:
        json.dump(report, f, indent=4)

    np.save(
        f"experiments/metrics/{VERSION}_confusion_matrix.npy",
        cm
    )


# -------------------
# Evaluation
# -------------------
def evaluate():
    model = tf.keras.models.load_model(MODEL_PATH)
    test_data = load_test_data()

    y_true = test_data.classes
    y_probs = model.predict(test_data)
    y_pred = (y_probs > 0.5).astype(int).ravel()

    report = classification_report(
        y_true,
        y_pred,
        target_names=["Normal", "Pneumonia"],
        output_dict=True
    )

    cm = confusion_matrix(y_true, y_pred)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Pneumonia"]))

    print("\nConfusion Matrix:")
    print(cm)

    save_metrics(report, cm)


if __name__ == "__main__":
    evaluate()
