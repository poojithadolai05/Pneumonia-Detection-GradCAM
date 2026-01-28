import tensorflow as tf
import numpy as np
import cv2
import os

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """
    Grad-CAM implementation that works with:
    - Keras 3
    - Sequential models
    - GlobalAveragePooling + sigmoid
    """

    # 1. Get last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)

    # 2. Build a model that maps input -> (conv output, logits BEFORE sigmoid)
    #    We explicitly stop at the Dense layer BEFORE sigmoid
    classifier_input = last_conv_layer.output
    x = classifier_input

    # Rebuild classifier head manually
    for layer in model.layers:
        if layer.name == last_conv_layer_name:
            start = True
            continue
        if not 'start' in locals():
            continue
        x = layer(x)
        if layer.__class__.__name__ == "Dense":
            break  # stop BEFORE final sigmoid

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, x]
    )

    # 3. Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, logits = grad_model(img_array)
        loss = tf.reduce_mean(logits)

    grads = tape.gradient(loss, conv_outputs)

    # 4. Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 5. Weight the feature maps
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 6. Normalize
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + tf.keras.backend.epsilon()

    return heatmap.numpy()



def overlay_heatmap(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)


def save_gradcam(image_id, overlay, project_root, version="cnn_v1"):
    output_dir = os.path.join(
        project_root,
        "experiments",
        "gradcam_outputs"
    )
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(
        output_dir,
        f"{version}_{image_id}.png"
    )
    cv2.imwrite(path, overlay)

    print("Grad-CAM saved at:", path)

