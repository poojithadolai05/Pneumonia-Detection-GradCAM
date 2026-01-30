# 🫁 Pneumonia Detection from Chest X-Rays with Grad-CAM Explainability

## 📌 Overview

This project presents an **end-to-end deep learning system** for detecting **Pneumonia** from chest X-ray images, enhanced with **Grad-CAM–based visual explanations** to improve model interpretability.

A custom **Convolutional Neural Network (CNN)** is trained from scratch for binary classification (**Pneumonia / Normal**).  
To enable real-world usability, the trained model is deployed as a **FastAPI-based REST API** and hosted on **Hugging Face Spaces**.

Along with predictions and confidence scores, the system generates **Grad-CAM heatmaps** that highlight lung regions influencing the model’s decision, helping bridge the gap between performance and explainability in medical AI.

---

## 🎯 Problem Statement

Pneumonia is a serious and potentially life-threatening lung infection that requires **early and accurate diagnosis**.  
Chest X-rays are widely used for diagnosis, but they present several challenges:

- Manual interpretation is time-consuming  
- Diagnosis can be subjective, especially in early-stage cases  
- Deep learning models often act as *black boxes*, reducing trust in clinical settings  

### Objectives of this project:
- Automatically classify chest X-ray images as **Pneumonia** or **Normal**  
- Provide **model explainability** using **Grad-CAM** visualizations  
- Deploy the solution as an **accessible inference API**  

This project aims to improve both **diagnostic support** and **model transparency**.

---

## 📂 Dataset

- Public **Chest X-ray Pneumonia** dataset  
- Two classes: `NORMAL`, `PNEUMONIA`  
- Grayscale chest X-ray images  

Dataset structure:

```text
chest_xray/
├── train/
├── val/
└── test/
```
## 🔄 Project Pipeline

1. **Data Loading and Preprocessing**  
   - Load chest X-ray images from the dataset  
   - Convert images to grayscale and resize to a fixed resolution  
   - Normalize pixel values for stable training  

2. **CNN Model Training (From Scratch)**  
   - Design and train a custom Convolutional Neural Network  
   - No transfer learning is used to ensure full architectural transparency  
   - Binary classification: *Pneumonia* vs *Normal*  

3. **Model Evaluation**  
   - Evaluate performance using medical-relevant metrics  
   - Emphasis on **Recall** to reduce false negatives  
   - Track accuracy, precision, recall, and AUC  

4. **Grad-CAM Explainability**  
   - Generate Grad-CAM heatmaps from the final convolutional layer  
   - Visualize lung regions influencing the model’s decision  
   - Use explainability to identify and reduce dataset bias  

5. **Deployment**  
   - Wrap the trained model in a **FastAPI** inference service  
   - Expose REST endpoints for prediction and Grad-CAM generation  
   - Deploy the API on **Hugging Face Spaces** for public access  

---
## 🧠 Model Architecture (CNN Only)

A custom **Convolutional Neural Network (CNN)** was designed **from scratch** to ensure full architectural transparency and interpretability, without relying on transfer learning.

### Key Components

- **Stacked Conv2D layers**  
  Extract hierarchical spatial features from chest X-ray images  

- **MaxPooling layers**  
  Reduce spatial dimensions and capture dominant features  

- **Global Average Pooling (GAP)**  
  Minimizes overfitting and improves generalization  

- **Fully Connected (Dense) layers**  
  Learn high-level representations for final decision making  

- **Sigmoid activation**  
  Enables **binary classification**: *Pneumonia* vs *Normal*  

The architecture is intentionally kept simple and interpretable, making it suitable for explainability techniques such as **Grad-CAM**.

---
## 📊 Training & Evaluation

The model was trained using medically relevant optimization strategies and evaluation metrics to ensure reliable performance in a clinical context.

### Training Configuration

- **Loss Function:** Binary Cross-Entropy  
- **Optimizer:** Adam  

### Evaluation Metrics

- **Accuracy** – Overall correctness of predictions  
- **Precision** – Reliability of Pneumonia-positive predictions  
- **Recall (Sensitivity)** – *Critical for pneumonia detection*  
- **AUC (ROC Curve)** – Class separability across thresholds  

### Medical Relevance

**Recall is strongly emphasized** to minimize **false negatives**, as failing to detect pneumonia can have serious clinical consequences.  
This aligns the model evaluation with real-world medical priorities rather than relying on accuracy alone.

---
## 🔍 Explainability with Grad-CAM

Grad-CAM is used to visualize the regions of a chest X-ray that most strongly influence the model’s prediction, helping interpret the model’s decision-making process.

### 🖼 Grad-CAM Visualization

- Warmer colors indicate regions that contribute more strongly to the prediction  
- Helps verify whether the model focuses on **lung regions** rather than irrelevant artifacts  

### 🔎 Observations

- Initial Grad-CAM outputs showed strong attention along **image borders**  
- This behavior revealed the presence of **dataset-related bias**, likely caused by scanner or padding artifacts  

### ✅ Improvements

After Grad-CAM visualization tuning and preprocessing refinements:

- Activations became more concentrated within **lung regions**  
- Reduced dependence on non-medical image artifacts  

This highlights the critical role of **explainability** in building **trustworthy medical AI systems** and in diagnosing hidden model biases.

---
## 🔌 FastAPI Inference API

This project includes a lightweight **FastAPI-based inference service** that enables real-time model predictions and explainability.

The API performs the following tasks:

- **Pneumonia / Normal classification** from chest X-ray images  
- **Confidence scoring** for each prediction  
- **Grad-CAM heatmap generation** to visualize regions influencing the model’s decision  

The trained model is loaded **once at application startup**, ensuring **low-latency**, production-style inference suitable for real-world deployment.
### 🌐 Live Deployment

The FastAPI inference service is deployed and publicly accessible on **Hugging Face Spaces**:

🔗 **Live API & Demo:**  
https://huggingface.co/spaces/poojithadolai/pneumonia-gradcam-api

---
## 📁 API Structure

The API code is organized in a clean and minimal structure to separate application logic from utility functions.

```text
api/
├── main.py     # FastAPI application (routing, model loading, inference)
└── utils.py    # Image preprocessing and Grad-CAM utility functions
```

## ▶️ Run the API Locally

Start the FastAPI server using **Uvicorn**:

```bash
uvicorn api.main:app --reload
```
Once the server is running, access the API at:
```bash
http://127.0.0.1:8000
```
## 📘 Swagger UI (Interactive API Docs)

FastAPI automatically provides interactive documentation at
```bash
http://127.0.0.1:8000/docs
```
This interface allows you to test endpoints, upload chest X-ray images, and view responses directly from the browser.

## 📤 Prediction Endpoint

### 🔗 Endpoint

```http
POST /predict
```
### 📥 Input

- **Type:** `multipart/form-data`
- **Field:** `file`
- **Supported formats:** `.jpg`, `.png` (Chest X-ray images)

The uploaded image is preprocessed and passed through the trained CNN model for inference and Grad-CAM generation.

### 📤 Output

```json
{
  "prediction": "Pneumonia",
  "confidence": 0.5363,
  "gradcam_image": "outputs/gradcam.png"
}
```
### 🖼 Grad-CAM Output Access

Grad-CAM images are saved on the server and served statically via:

```text
/outputs/gradcam.png
```
This allows direct access to the generated Grad-CAM visualization through a browser or API client.

## 📈 Results

- The CNN learns meaningful **lung-related features** from chest X-ray images  
- **Grad-CAM** provides interpretable visual explanations for model predictions  
- Explainability helped **identify and reduce dataset bias**

---

## ⚠️ Limitations

- No explicit **lung segmentation**
- **Classification-only** model (no lesion localization)
- Dataset bias may still exist

---

## 🚀 Future Work

- Integrate **lung segmentation** for better focus on lung regions  
- Improve localization using **attention mechanisms**  
- Validate performance on **external clinical datasets**  
- Enhance the **user interface** for clinical usage

---

## 🛠 Tech Stack

- Python  
- TensorFlow / Keras  
- FastAPI  
- NumPy  
- Matplotlib  
- Hugging Face Spaces  

