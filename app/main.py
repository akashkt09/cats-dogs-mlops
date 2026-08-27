# app/main.py
"""FastAPI service for cats vs dogs image classification."""

import io
import os
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import tensorflow as tf

app = FastAPI(title="Cats vs Dogs Classifier")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "cats_dogs_cnn.h5")
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = (224, 224)
CLASS_NAMES = {0: "cat", 1: "dog"}


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        img_array = preprocess_image(image_bytes)

        prob_dog = float(model.predict(img_array, verbose=0)[0][0])
        prob_cat = 1.0 - prob_dog

        prediction = 1 if prob_dog > 0.5 else 0
        confidence = prob_dog if prediction == 1 else prob_cat

        return JSONResponse({
            "prediction": CLASS_NAMES[prediction],
            "confidence": round(confidence, 4),
            "probabilities": {
                "cat": round(prob_cat, 4),
                "dog": round(prob_dog, 4)
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))