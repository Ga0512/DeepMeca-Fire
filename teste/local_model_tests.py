"""Run sanity checks against the local wildfire models without the API layer."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.models import Model

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "Algerian_forest_fires_dataset_CLEANED.csv"

FEATURE_COLUMNS = ["Temperature", "Ws", "FFMC", "DMC", "ISI"]

MODEL_STORE = Model()


@dataclass
class Sample:
    name: str
    values: List[float]


SAMPLES: Iterable[Sample] = (
    Sample("high_risk_1", [31, 14, 82.6, 5.8, 3.1]),
    Sample("high_risk_2", [33, 13, 88.2, 9.9, 6.4]),
    Sample("low_risk_1", [29, 13, 64.4, 4.1, 1.0]),
    Sample("low_risk_2", [25, 13, 28.6, 1.3, 0.0]),
)


def _load_scaler() -> StandardScaler:
    df = pd.read_csv(DATASET_PATH, usecols=FEATURE_COLUMNS)
    scaler = StandardScaler()
    scaler.fit(df)
    return scaler


def evaluate_samples():
    scaler = _load_scaler()
    classification_model = MODEL_STORE.classification()
    regression_model = MODEL_STORE.regression()

    for sample in SAMPLES:
        features = np.array([sample.values])
        scaled = scaler.transform(features)

        class_pred = int(classification_model.predict(scaled)[0])
        regression_pred = float(regression_model.predict(scaled)[0])

        status = "🔥 risco" if class_pred == 1 else "✅ seguro"
        print(
            f"{sample.name:>12}: class={class_pred} ({status}), FWI={regression_pred:.2f}"
        )


if __name__ == "__main__":
    evaluate_samples()
