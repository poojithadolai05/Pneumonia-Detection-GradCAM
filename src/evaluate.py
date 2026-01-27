import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 224
BATCH_SIZE = 32
DATA_DIR = "data/raw/chest_xray"

def load_test_data():
    test_gen = ImageDataGenerator(rescale=1./255)

    test_data = test_gen.flow_from_directory(
        f"{DATA_DIR}/test",
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )

    return test_data


def evaluate():
    model = tf.keras.models.load_model("experiments/models/best_model.h5")
    test_data = load_test_data()

    y_true = test_data.classes
    y_pred_probs = model.predict(test_data)
    y_pred = (y_pred_probs > 0.5).astype(int).ravel()

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Pneumonia"]))

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    return cm


if __name__ == "__main__":
    evaluate()
