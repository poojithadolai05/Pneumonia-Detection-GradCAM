import os
import cv2
import numpy as np

IMG_SIZE = 224
CATEGORIES = ["NORMAL", "PNEUMONIA"]

def load_split(split_dir):
    images = []
    labels = []

    for label, category in enumerate(CATEGORIES):
        category_path = os.path.join(split_dir, category)

        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0

            images.append(img)
            labels.append(label)

    images = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    labels = np.array(labels)

    return images, labels


def load_data(base_dir):
    train_dir = os.path.join(base_dir, "train")
    val_dir   = os.path.join(base_dir, "val")
    test_dir  = os.path.join(base_dir, "test")

    X_train, y_train = load_split(train_dir)
    X_val, y_val     = load_split(val_dir)
    X_test, y_test   = load_split(test_dir)

    return X_train, X_val, X_test, y_train, y_val, y_test
