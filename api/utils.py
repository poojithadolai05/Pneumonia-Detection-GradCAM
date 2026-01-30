import io
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

IMG_SIZE = 224
LAST_CONV_LAYER_NAME = "conv2d_4"


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = image.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    return image


def generate_gradcam(model, img_array, save_path="outputs/gradcam.png"):
    os.makedirs("outputs", exist_ok=True)

    # Get last conv layer
    last_conv_layer = model.get_layer(LAST_CONV_LAYER_NAME)

    # Forward pass manually
    with tf.GradientTape() as tape:
        x = img_array
        for layer in model.layers:
            x = layer(x)
            if layer.name == LAST_CONV_LAYER_NAME:
                conv_output = x
                tape.watch(conv_output)

        prediction = x
        class_score = prediction[:, 0]

    # Compute gradients
    grads = tape.gradient(class_score, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    # Save heatmap
    plt.figure(figsize=(4, 4))
    plt.imshow(heatmap, cmap="jet")
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    return save_path

