import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import os

from src.gradcam import make_gradcam_heatmap, overlay_heatmap

IMG_SIZE = 224
LAST_CONV_LAYER = "conv2d_4"


def preprocess_image(image_bytes):
    """
    Convert uploaded bytes → model input + original image
    """
    image = Image.open(BytesIO(image_bytes)).convert("L")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))

    return img_array, np.array(image)


def generate_gradcam(model, img_array, original_img, project_root):
    """
    Generate and save Grad-CAM, return file path
    """
    # Ensure model graph is built
    _ = model(img_array)

    heatmap = make_gradcam_heatmap(
        img_array,
        model,
        last_conv_layer_name=LAST_CONV_LAYER
    )

    overlay = overlay_heatmap(
        cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR),
        heatmap,
        alpha=0.25
    )

    output_dir = os.path.join(
        project_root, "experiments", "gradcam_outputs"
    )
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir, "api_gradcam_output.png"
    )

    cv2.imwrite(output_path, overlay)

    return output_path
