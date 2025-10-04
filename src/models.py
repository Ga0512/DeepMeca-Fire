"""Utilities for loading wildfire models used across the project."""
from __future__ import annotations

import bz2
from pathlib import Path
from typing import Any, Callable, Dict, Optional

try:  # pragma: no cover - optional dependency
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore


class Model:
    """Centralised access to the persisted ML artefacts."""

    def __init__(self, models_dir: Optional[Path | str] = None) -> None:
        base_path = Path(models_dir) if models_dir is not None else None
        if base_path is None:
            base_path = Path(__file__).resolve().parents[1] / "models"
        self.models_dir = base_path
        self._cache: Dict[str, Any] = {}

    def _resolve(self, filename: str) -> Path:
        path = self.models_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Model artefact not found: {path}")
        return path

    def _load_pickle(self, path: Path) -> Any:
        import pickle

        with bz2.BZ2File(path, "rb") as file_obj:
            return pickle.load(file_obj)

    def _load_yolo(self, path: Path) -> Any:
        if YOLO is None:
            raise RuntimeError(
                "The 'ultralytics' package is required to load the YOLO model."
            )
        return YOLO(str(path))

    def _get_cached(self, filename: str, loader: Callable[[Path], Any]) -> Any:
        if filename not in self._cache:
            self._cache[filename] = loader(self._resolve(filename))
        return self._cache[filename]

    def regression(self) -> Any:
        return self._get_cached("regression.pkl", self._load_pickle)

    def classification(self) -> Any:
        return self._get_cached("classification.pkl", self._load_pickle)

    def yolo(self) -> Any:
        return self._get_cached("fire_n.pt", self._load_yolo)
