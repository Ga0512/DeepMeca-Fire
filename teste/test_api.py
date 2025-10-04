"""Utility script to exercise the FastAPI endpoints exposed in ``server.py``.

The script sends a prediction request using mocked meteorological features and,
optionally, runs an image detection request when an image path is provided.

Usage
-----
python teste/test_api.py --base-url http://127.0.0.1:8000 \
    --image-path path/to/image.jpg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import requests

DEFAULT_SAMPLE = {
    "Temperature": 31,
    "Ws": 14,
    "FFMC": 82.6,
    "DMC": 5.8,
    "ISI": 3.1,
}


def request_prediction(base_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(f"{base_url}/predict", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def request_detection(base_url: str, image_path: Path) -> Dict[str, Any]:
    with image_path.open("rb") as file_obj:
        files = {"file": (image_path.name, file_obj, "image/jpeg")}
        response = requests.post(f"{base_url}/detect", files=files, timeout=60)
        response.raise_for_status()
        return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the wildfire FastAPI service")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL where the FastAPI service is running",
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        help="Optional path to an image for exercising the detection endpoint",
    )
    parser.add_argument(
        "--payload",
        type=str,
        help="Optional JSON string overriding the default prediction payload",
    )
    args = parser.parse_args()

    payload = DEFAULT_SAMPLE.copy()
    if args.payload:
        payload.update(json.loads(args.payload))

    print("➡️  Sending prediction request...")
    prediction = request_prediction(args.base_url, payload)
    print(json.dumps(prediction, indent=2, ensure_ascii=False))

    if args.image_path:
        if not args.image_path.exists():
            raise FileNotFoundError(f"Image not found: {args.image_path}")
        print("➡️  Sending detection request...")
        detection = request_detection(args.base_url, args.image_path)
        print(json.dumps(detection, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
