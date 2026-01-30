from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
from huggingface_hub import hf_hub_download
import os

from api.utils import preprocess_image, generate_gradcam

app = FastAPI(title="Pneumonia Grad-CAM API")

# ✅ Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

# ✅ Serve outputs folder
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

MODEL_PATH = hf_hub_download(
    repo_id="poojithadolai/pneumonia-gradcam-cnn",
    filename="cnn_v1_best.keras"
)

model = tf.keras.models.load_model(MODEL_PATH)
model.trainable = False


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img_array = preprocess_image(image_bytes)

    preds = model(img_array)
    confidence = float(preds[0][0])
    label = "Pneumonia" if confidence > 0.5 else "Normal"

    gradcam_path = generate_gradcam(model, img_array)

    return {
        "prediction": label,
        "confidence": confidence,
        "gradcam_image": gradcam_path
    }
