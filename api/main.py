from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import os

from api.utils import preprocess_image, generate_gradcam

app = FastAPI(
    title="Pneumonia Detection API",
    description="CNN-based pneumonia detection with Grad-CAM explainability",
    version="1.0"
)

MODEL_PATH = "experiments/models/cnn_v1_best.keras"
PROJECT_ROOT = os.path.abspath(".")

model = tf.keras.models.load_model(MODEL_PATH)


@app.get("/")
def health_check():
    return {"status": "API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    img_array, original_img = preprocess_image(image_bytes)

    confidence = float(model.predict(img_array)[0][0])
    label = "Pneumonia" if confidence >= 0.5 else "Normal"

    gradcam_path = generate_gradcam(
        model,
        img_array,
        original_img,
        PROJECT_ROOT
    )

    return {
        "prediction": label,
        "confidence": round(confidence, 4),
        "gradcam_image_path": gradcam_path
    }
