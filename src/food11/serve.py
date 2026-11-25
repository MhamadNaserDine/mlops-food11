"""
Food-11 inference API (Lab 3).

Lab 5: the model is loaded at startup into the module-level variable
`model`, unless the environment variable SKIP_MODEL is set. This lets
unit tests and the CI smoke test run without an MLflow server.
"""
import os
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

import mlflow
from src.food11.data import CLASSES

MODEL_URI = "models:/food11@champion"

# Module-level model: None until loaded (tests can replace it with a fake)
model = None


def load_model():
    """Load the champion model from MLflow, unless SKIP_MODEL is set."""
    global model
    if os.environ.get("SKIP_MODEL"):
        print("SKIP_MODEL is set: starting without a model")
        return
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    model = mlflow.pyfunc.load_model(MODEL_URI)
    print(f"Loaded {MODEL_URI} from {tracking_uri}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once when the app starts (replacement for @app.on_event("startup"))
    load_model()
    yield


app = FastAPI(lifespan=lifespan)

# Same preprocessing as in train.py
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: Annotated[UploadFile, File()]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    tensor = transform(image).unsqueeze(0)
    input_array = tensor.numpy().astype(np.float32)

    prediction = np.array(model.predict(input_array))

    if prediction.ndim == 2:
        # The model returns 11 raw scores (logits) -> softmax -> probabilities
        logits = prediction[0]
        exp_values = np.exp(logits - np.max(logits))
        probabilities = exp_values / exp_values.sum()
        class_index = int(np.argmax(probabilities))
        confidence = float(probabilities[class_index])
    else:
        class_index = int(prediction[0])
        confidence = 1.0

    return {
        "category": CLASSES[class_index],
        "confidence": confidence,
    }