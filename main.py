from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
import tensorflow as tf

app = FastAPI()

# Allow all CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load a dummy pretrained model from Keras Applications
model = tf.keras.applications.MobileNetV2(weights="imagenet")

@app.get("/")
async def root():
    return {"message": "NeuroTrace API is Live"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Get predictions
    preds = model.predict(img_array)
    decoded_preds = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0][0]

    # Simulate brain tumor vs no tumor based on random or confidence placeholder
    label = "tumor" if decoded_preds[1].lower() in ["brain", "head", "tumor"] else "no_tumor"
    confidence = round(float(decoded_preds[2]) * 100, 2)

    return {
        "label": label,
        "confidence": confidence
    }
