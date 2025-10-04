"""Quick smoke tests for the locally stored wildfire models."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import argparse
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.models import Model

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "Algerian_forest_fires_dataset_CLEANED.csv"
FEATURE_COLUMNS = ["Temperature", "Ws", "FFMC", "DMC", "ISI"]

# Default sample used for the regression/classification sanity checks.
SAMPLE = np.array([[31.0, 14.0, 82.6, 5.8, 3.1]], dtype=float)

MODEL_STORE = Model()


def _get_scaler() -> StandardScaler:
    """Fit a ``StandardScaler`` using the original training dataset."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH, usecols=FEATURE_COLUMNS)
    scaler = StandardScaler().fit(df)
    return scaler


def test_regression(sample: Optional[np.ndarray] = None) -> float:
    """Run a regression prediction and return the estimated FWI value."""

    features = sample if sample is not None else SAMPLE
    scaler = _get_scaler()
    scaled = scaler.transform(features)

    model = MODEL_STORE.regression()
    prediction = float(model.predict(scaled)[0])
    print(f"FWI estimado: {prediction:.2f}")
    return prediction


def test_classification(sample: Optional[np.ndarray] = None) -> int:
    """Run a classification prediction and return the fire risk label."""

    features = sample if sample is not None else SAMPLE
    scaler = _get_scaler()
    scaled = scaler.transform(features)

    model = MODEL_STORE.classification()
    prediction = int(model.predict(scaled)[0])
    status = "🔥 risco" if prediction == 1 else "✅ seguro"
    print(f"Classificação: {prediction} ({status})")
    return prediction


def test_yolo_image(image_path: Path) -> int:
    """Execute the YOLO detector against an image and return detections count."""

    if not image_path.exists():
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"Não foi possível carregar a imagem: {image_path}")

    model = MODEL_STORE.yolo()
    results = model(frame, verbose=False)[0]
    detections = len(results.boxes)
    print(f"Detecções na imagem: {detections}")
    return detections


def test_yolo_video(video_path: Path, frame_limit: int = 5) -> int:
    """Run YOLO on a few frames of a video and return total detections."""

    if not video_path.exists():
        raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")

    model = MODEL_STORE.yolo()
    total_detections = 0
    processed_frames = 0

    while processed_frames < frame_limit:
        success, frame = capture.read()
        if not success:
            break
        processed_frames += 1

        results = model(frame, verbose=False)[0]
        total_detections += len(results.boxes)

    capture.release()
    print(
        f"Detecções no vídeo ({processed_frames} frames analisados): {total_detections}"
    )
    return total_detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testes locais dos modelos")
    parser.add_argument("--image", type=Path, help="Imagem para testar o YOLO", default=None)
    parser.add_argument("--video", type=Path, help="Vídeo para testar o YOLO", default=None)
    parser.add_argument(
        "--frames",
        type=int,
        default=5,
        help="Quantidade de frames do vídeo a serem avaliados",
    )
    args = parser.parse_args()

    test_regression()
    test_classification()

    if args.image is not None:
        test_yolo_image(args.image)

    if args.video is not None:
        test_yolo_video(args.video, frame_limit=args.frames)
