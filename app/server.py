"""FastAPI application exposing wildfire prediction services.

This module loads the pre-trained YOLO detection model together with the
classification and regression models used in the notebooks. It offers three
endpoints:

* ``GET /health`` – readiness probe.
* ``POST /predict`` – estimates the Fire Weather Index (FWI) and fire risk for a
  given set of meteorological features.
* ``POST /detect`` – runs object detection on an uploaded image using the YOLO
  model and returns the bounding boxes and confidences.

Execute the module directly to start a local development server, or import the
``app`` object into an ASGI server of choice.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, root_validator
from sklearn.preprocessing import StandardScaler

from src.models import Model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "Algerian_forest_fires_dataset_CLEANED.csv"
FEATURE_COLUMNS = ["Temperature", "Ws", "FFMC", "DMC", "ISI"]

MODEL_STORE = Model()


class PredictRequest(BaseModel):
    """Schema describing the expected payload for ``/predict``."""

    Temperature: float = Field(..., description="Daily temperature in Celsius")
    Ws: float = Field(..., description="Wind speed (km/h)")
    FFMC: float = Field(..., description="Fine Fuel Moisture Code")
    DMC: float = Field(..., description="Duff Moisture Code")
    ISI: float = Field(..., description="Initial Spread Index")

    @root_validator
    def validate_feature_ranges(cls, values: dict) -> dict:
        temperature = values.get("Temperature")
        if temperature is not None and not (-50 <= temperature <= 60):
            raise ValueError("Temperature must be between -50 and 60°C")
        return values


class PredictResponse(BaseModel):
    fwi_estimation: float = Field(..., description="Predicted Fire Weather Index")
    fire_risk: int = Field(..., description="Binary fire risk classification (1 = risco)")
    status_label: str = Field(..., description="Human readable interpretation of the risk")


class DetectionResult(BaseModel):
    class_name: str = Field(..., description="Predicted class label")
    confidence: float = Field(..., description="Prediction confidence between 0 and 1")
    box: List[float] = Field(..., description="[x1, y1, x2, y2] bounding box in pixels")


class DetectionResponse(BaseModel):
    detections: List[DetectionResult]


# ---------------------------------------------------------------------------
# Lazy loaders for artefacts
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_scaler() -> StandardScaler:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH, usecols=FEATURE_COLUMNS)
    scaler = StandardScaler()
    scaler.fit(df)
    logger.info("StandardScaler fitted with %s samples", len(df))
    return scaler


def _get_classification_model():
    return MODEL_STORE.classification()


def _get_regression_model():
    return MODEL_STORE.regression()


def _get_detection_model():
    return MODEL_STORE.yolo()


def _scale_features(features: np.ndarray) -> np.ndarray:
    scaler = get_scaler()
    return scaler.transform(features)


def _format_prediction(prediction: np.ndarray) -> float:
    return float(np.asarray(prediction).ravel()[0])


# ---------------------------------------------------------------------------
# FastAPI application setup
# ---------------------------------------------------------------------------
app = FastAPI(title="DeepMeca Wildfire Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    """Preload artefacts so the first request is not penalised."""

    logger.info("Initialising models...")
    _get_detection_model()
    _get_classification_model()
    _get_regression_model()
    get_scaler()
    logger.info("Startup completed")


@app.get("/health")
def health() -> dict:
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    features = np.array(
        [[request.Temperature, request.Ws, request.FFMC, request.DMC, request.ISI]]
    )
    scaled = _scale_features(features)

    regression_model = _get_regression_model()
    regression_pred = _format_prediction(regression_model.predict(scaled))

    classification_model = _get_classification_model()
    class_pred = int(_format_prediction(classification_model.predict(scaled)))

    status_label = "🔥 Perigo de incêndio" if class_pred == 1 else "✅ Condição segura"

    return PredictResponse(
        fwi_estimation=round(regression_pred, 2),
        fire_risk=class_pred,
        status_label=status_label,
    )


@app.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)) -> DetectionResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="A valid image file is required")

    contents = await file.read()
    image_array = np.frombuffer(contents, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Unable to decode the uploaded image")

    model = _get_detection_model()
    results = model(frame, verbose=False)[0]

    detections: List[DetectionResult] = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        detections.append(
            DetectionResult(
                class_name=model.names.get(cls_id, str(cls_id)),
                confidence=round(conf, 4),
                box=[round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            )
        )

    return DetectionResponse(detections=detections)


def run(host: str = "0.0.0.0", port: int = int(os.getenv("PORT", "8000"))):
    """Convenience entry-point for starting a Uvicorn development server."""

    import uvicorn

    uvicorn.run("app.server:app", host=host, port=port, reload=False, log_level="info")


if __name__ == "__main__":
    run()
