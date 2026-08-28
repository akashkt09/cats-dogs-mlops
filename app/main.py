"""FastAPI service for cats vs dogs image classification."""

import io
import os
import time
import logging
from collections import defaultdict

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import tensorflow as tf

app = FastAPI(title="Cats vs Dogs Classifier")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('api_requests.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('cats-dogs-api')

metrics = {
    'total_requests': 0,
    'prediction_counts': defaultdict(int),
    'total_response_time_ms': 0.0,
    'errors': 0,
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "cats_dogs_cnn.h5")
IMG_SIZE = (224, 224)
CLASS_NAMES = {0: "cat", 1: "dog"}

_model = None


def get_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/metrics")
def get_metrics():
    avg_response_time = (
        metrics['total_response_time_ms'] / metrics['total_requests']
        if metrics['total_requests'] > 0 else 0
    )
    return {
        'total_requests': metrics['total_requests'],
        'errors': metrics['errors'],
        'prediction_distribution': dict(metrics['prediction_counts']),
        'average_response_time_ms': round(avg_response_time, 2)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    metrics['total_requests'] += 1

    if not file.content_type or not file.content_type.startswith("image/"):
        metrics['errors'] += 1
        logger.warning(f"Rejected upload with content_type={file.content_type}")
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        img_array = preprocess_image(image_bytes)
        model = get_model()

        prob_dog = float(model.predict(img_array, verbose=0)[0][0])
        prob_cat = 1.0 - prob_dog

        prediction = 1 if prob_dog > 0.5 else 0
        confidence = prob_dog if prediction == 1 else prob_cat
        label = CLASS_NAMES[prediction]

        response_time_ms = (time.time() - start_time) * 1000
        metrics['prediction_counts'][label] += 1
        metrics['total_response_time_ms'] += response_time_ms

        logger.info(
            f"Prediction served | filename={file.filename} content_type={file.content_type} "
            f"prediction={label} confidence={confidence:.4f} response_time_ms={response_time_ms:.2f}"
        )

        return JSONResponse({
            "prediction": label,
            "confidence": round(confidence, 4),
            "probabilities": {
                "cat": round(prob_cat, 4),
                "dog": round(prob_dog, 4)
            }
        })
    except Exception as e:
        metrics['errors'] += 1
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
# demo trigger Fri Aug 28 15:03:50 IST 2026
