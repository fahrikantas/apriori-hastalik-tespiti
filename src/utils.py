"""Shared utilities for the disease prediction project.

This module centralizes path handling, symptom normalization, and small helper
functions that are reused across preprocessing, model training, and inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import joblib
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
DEFAULT_DELIMITER = ";"
TARGET_COLUMN = "prognosis"
MODEL_EXT = ".pkl"
DATASET_ALIASES: dict[str, tuple[str, ...]] = {
    "Training.csv": ("training_duzenlenmis.csv",),
}


def ensure_directory(path: Path) -> Path:
    """Create a directory if it does not already exist."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_data_path(file_name: str) -> Path:
    """Return the first existing path for a dataset file.

    The project supports both the requested `data/` folder structure and the
    workspace's current flat layout, so the function checks multiple locations.
    """

    alias_names = DATASET_ALIASES.get(file_name, ())
    candidates = [
        DATA_DIR / file_name,
        *(DATA_DIR / alias_name for alias_name in alias_names),
        PROJECT_DIR / file_name,
        *(PROJECT_DIR / alias_name for alias_name in alias_names),
        PROJECT_DIR.parent / file_name,
        *(PROJECT_DIR.parent / alias_name for alias_name in alias_names),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Data file not found: {file_name}")


def resolve_model_path(file_name: str) -> Path:
    """Return the absolute path for a model artifact inside `models/`."""

    ensure_directory(MODELS_DIR)
    return MODELS_DIR / file_name


def load_dataset(file_name: str, delimiter: str = DEFAULT_DELIMITER) -> pd.DataFrame:
    """Load a semicolon-delimited dataset into a pandas DataFrame."""

    file_path = resolve_data_path(file_name)
    return pd.read_csv(file_path, sep=delimiter)


def normalize_symptom_name(value: str) -> str:
    """Normalize symptom labels so lookups are stable across sources."""

    cleaned = value.strip().lower()
    cleaned = cleaned.replace(" ", "_")
    cleaned = cleaned.replace("-", "_")
    cleaned = cleaned.replace("__", "_")
    return cleaned


def humanize_label(value: str) -> str:
    """Convert a normalized symptom label into a display-friendly label."""

    return value.replace("_", " ").strip().title()


def save_model(model: object, file_name: str) -> Path:
    """Persist a trained estimator with joblib."""

    model_path = resolve_model_path(file_name)
    joblib.dump(model, model_path)
    return model_path


def load_model(file_name: str) -> object:
    """Load a persisted estimator from the models directory."""

    model_path = resolve_model_path(file_name)
    return joblib.load(model_path)


def deduplicate_preserve_order(values: Iterable[str]) -> list[str]:
    """Remove duplicates from an iterable while keeping the original order."""

    seen: set[str] = set()
    ordered_values: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered_values.append(value)
    return ordered_values


def ensure_columns_exist(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise a clear error when one or more required columns are missing."""

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
