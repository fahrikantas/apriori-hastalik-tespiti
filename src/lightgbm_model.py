"""LightGBM model training and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import re

from src.model_metadata import build_bundle_metadata
from src.preprocess import preprocess_training_data
from src.split import split_train_test
from src.utils import TARGET_COLUMN, resolve_model_path

LIGHTGBM_MODEL_NAME = "lightgbm.pkl"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_ESTIMATORS = 100


@dataclass(frozen=True)
class LightGBMResult:
    """Training result bundle for LightGBM."""

    model: Any
    label_encoder: LabelEncoder
    accuracy: float
    classification_report_text: str
    feature_importance: pd.DataFrame
    feature_columns: list[str]
    model_path: str


def _prepare_training_split(file_name: str) -> tuple[pd.DataFrame, pd.Series, list[str], Any]:
    preprocessed = preprocess_training_data(file_name)
    frame = preprocessed.frame
    feature_columns = preprocessed.symptom_columns
    return frame[feature_columns], frame[TARGET_COLUMN], feature_columns, preprocessed


def _make_estimator(random_state: int = DEFAULT_RANDOM_STATE, n_estimators: int = DEFAULT_ESTIMATORS) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
        class_weight="balanced",
    )


def train_lightgbm(
    file_name: str = "Training.csv",
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_estimators: int = DEFAULT_ESTIMATORS,
) -> LightGBMResult:
    """Train a LightGBM classifier and persist the fitted model.

    The serving model is fit on the full dataset; the reported accuracy is
    measured with an independently fitted, leak-aware fold model.
    """

    features, target, feature_columns, preprocessed = _prepare_training_split(file_name)
    label_encoder = LabelEncoder()
    encoded_target = label_encoder.fit_transform(target)

    # Sanitize feature names for LightGBM (no special JSON characters)
    # Map original -> safe name and ensure uniqueness
    safe_map: dict[str, str] = {}
    used: dict[str, int] = {}

    def _safe_name(name: str) -> str:
        # Replace any non-alphanumeric/underscore with underscore
        base = re.sub(r"[^0-9A-Za-z_]", "_", str(name))
        # Ensure it doesn't start with a digit (optional but safer)
        if re.match(r"^[0-9]", base):
            base = "f_" + base
        # Collapse multiple underscores
        base = re.sub(r"_+", "_", base).strip("_") or "f"
        # Avoid collisions
        count = used.get(base, 0)
        used[base] = count + 1
        return f"{base}" if count == 0 else f"{base}_{count}"

    sanitized_columns: list[str] = []
    for orig in feature_columns:
        safe = _safe_name(orig)
        safe_map[orig] = safe
        sanitized_columns.append(safe)

    # Create sanitized feature frames for training/prediction
    features_sanitized = features.copy()
    features_sanitized.columns = sanitized_columns

    x_train, x_test, y_train, y_test = split_train_test(
        features_sanitized,
        encoded_target,
        test_size=test_size,
        random_state=random_state,
    )

    serving_model = _make_estimator(random_state, n_estimators)
    serving_model.fit(features_sanitized, encoded_target)

    eval_model = _make_estimator(random_state, n_estimators)
    eval_model.fit(x_train, y_train)

    predictions = eval_model.predict(x_test)
    accuracy = float(accuracy_score(y_test, predictions))

    import numpy as _np

    present_labels = _np.unique(_np.concatenate([_np.asarray(y_test), _np.asarray(predictions)]))
    try:
        present_names = label_encoder.inverse_transform(present_labels)
        report_text = classification_report(
            y_test,
            predictions,
            labels=present_labels.tolist(),
            target_names=present_names.tolist(),
            zero_division=0,
        )
    except Exception:
        report_text = classification_report(y_test, predictions, zero_division=0)

    # Map feature importances back to original feature names
    fi = serving_model.feature_importances_
    # fi aligns with sanitized_columns
    importance_pairs = list(zip(sanitized_columns, fi))
    # invert mapping sanitized -> original (if multiple originals map to same safe, choose first)
    inv_map = {v: k for k, v in safe_map.items()}
    mapped = [(inv_map.get(safe, safe), imp) for safe, imp in importance_pairs]
    feature_importance = (
        pd.DataFrame(mapped, columns=["feature", "importance"]).sort_values(by="importance", ascending=False).reset_index(drop=True)
    )

    model_path = resolve_model_path(LIGHTGBM_MODEL_NAME)
    joblib.dump(
        {
            "model": serving_model,
            "label_encoder": label_encoder,
            "feature_columns": feature_columns,
            "accuracy": accuracy,
            "classification_report": report_text,
            "feature_importance": feature_importance,
            "metadata": build_bundle_metadata(
                preprocessed.frame,
                feature_columns,
                file_name,
                test_size=test_size,
                random_state=random_state,
            ),
        },
        model_path,
    )

    return LightGBMResult(
        model=serving_model,
        label_encoder=label_encoder,
        accuracy=accuracy,
        classification_report_text=report_text,
        feature_importance=feature_importance,
        feature_columns=feature_columns,
        model_path=str(model_path),
    )