"""Model metadata, freshness checks and full-retraining helpers.

Every trained artifact in ``models/`` is a joblib bundle that now carries a
``metadata`` entry. This module knows how to compute a stable data fingerprint
from the cleaned training frame, compare it against persisted metadata and
(retrain) the entire model set in one call so the UI can stay in sync with the
data file.
"""

from __future__ import annotations

import hashlib
import joblib
from datetime import datetime, timezone
from typing import Any, Sequence

import pandas as pd

from src.utils import TARGET_COLUMN, resolve_model_path
from src.versioning import MODEL_SCHEMA_VERSION

DEFAULT_TRAINING_FILE = "Training.csv"
APRIORI_CACHE_NAME = "apriori_rules.pkl"


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    """Return a stable SHA-256 fingerprint for a cleaned training frame."""

    canonical = frame.sort_values(list(frame.columns)).reset_index(drop=True)
    blob = canonical.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def compute_fingerprint(training_file: str = DEFAULT_TRAINING_FILE) -> dict[str, Any]:
    """Compute the data fingerprint summary for the current training file."""

    from src.preprocess import preprocess_training_data

    preprocessed = preprocess_training_data(training_file)
    frame = preprocessed.frame
    feature_columns = preprocessed.symptom_columns
    return {
        "data_fingerprint": _frame_fingerprint(frame[feature_columns + [TARGET_COLUMN]]),
        "data_rows": int(len(frame)),
        "symptom_count": len(feature_columns),
        "class_count": int(frame[TARGET_COLUMN].nunique()),
    }


def build_bundle_metadata(
    frame: pd.DataFrame,
    feature_columns: list[str],
    training_file: str = DEFAULT_TRAINING_FILE,
    test_size: float | None = None,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Build the metadata dict persisted inside a supervised model bundle."""

    metadata: dict[str, Any] = {
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "training_file": training_file,
        "data_fingerprint": _frame_fingerprint(frame[feature_columns + [TARGET_COLUMN]]),
        "data_rows": int(len(frame)),
        "symptom_count": len(feature_columns),
        "class_count": int(frame[TARGET_COLUMN].nunique()),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if test_size is not None:
        metadata["test_size"] = test_size
    if random_state is not None:
        metadata["random_state"] = random_state
    return metadata


def read_bundle_metadata(model_name: str) -> dict[str, Any] | None:
    """Return the metadata of a persisted model bundle, or None if unavailable."""

    model_path = resolve_model_path(model_name)
    if not model_path.exists():
        return None
    try:
        bundle = joblib.load(model_path)
    except Exception:
        return None
    if isinstance(bundle, dict):
        metadata = bundle.get("metadata")
        return metadata if isinstance(metadata, dict) else None
    return None


def model_statuses(model_names: Sequence[str], training_file: str = DEFAULT_TRAINING_FILE) -> dict[str, dict[str, Any]]:
    """Compare persisted metadata against the current training file."""

    try:
        current = compute_fingerprint(training_file)
    except Exception:
        current = {}
    statuses: dict[str, dict[str, Any]] = {}
    for model_name in model_names:
        metadata = read_bundle_metadata(model_name)
        if metadata is None:
            statuses[model_name] = {"fresh": False, "reason": "missing"}
            continue
        fingerprinted = metadata.get("data_fingerprint")
        is_fresh = bool(fingerprinted and fingerprinted == current.get("data_fingerprint"))
        statuses[model_name] = {
            "fresh": is_fresh,
            "reason": "ok" if is_fresh else "stale",
            "trained_at": metadata.get("trained_at"),
            "data_rows": metadata.get("data_rows"),
            "class_count": metadata.get("class_count"),
        }
    return statuses


def models_are_fresh(model_names: Sequence[str]) -> bool:
    """Return True when every persisted model is up to date with the data."""

    statuses = model_statuses(model_names)
    return bool(statuses) and all(status["fresh"] for status in statuses.values())


def _rebuild_apriori_cache(training_file: str, apriori_params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Rebuild and persist the Apriori symptom-to-disease rules."""

    from src.apriori_rules import (
        DEFAULT_MIN_CONFIDENCE,
        DEFAULT_MIN_LIFT,
        DEFAULT_MIN_SUPPORT,
        DEFAULT_MAX_LEN,
        build_symptom_to_disease_rules,
        prepare_apriori_from_training,
    )

    params = apriori_params or {}
    apriori_result = prepare_apriori_from_training(
        training_file,
        min_support=float(params.get("min_support", DEFAULT_MIN_SUPPORT)),
        min_confidence=float(params.get("min_confidence", DEFAULT_MIN_CONFIDENCE)),
        min_lift=float(params.get("min_lift", DEFAULT_MIN_LIFT)),
        max_len=int(params.get("max_len", DEFAULT_MAX_LEN)),
    )
    rules = build_symptom_to_disease_rules(apriori_result.rules)
    metadata = compute_fingerprint(training_file)
    metadata["trained_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cache_path = resolve_model_path(APRIORI_CACHE_NAME)
    joblib.dump(
        {
            "rules": rules,
            "ready": True,
            "metadata": metadata,
            "min_support": float(params.get("min_support", DEFAULT_MIN_SUPPORT)),
            "min_confidence": float(params.get("min_confidence", DEFAULT_MIN_CONFIDENCE)),
            "min_lift": float(params.get("min_lift", DEFAULT_MIN_LIFT)),
            "max_len": int(params.get("max_len", DEFAULT_MAX_LEN)),
        },
        cache_path,
    )
    return {"trained_at": metadata["trained_at"], "rules": len(rules)}


def retrain_all_models(
    training_file: str = DEFAULT_TRAINING_FILE,
) -> list[dict[str, Any]]:
    """Train every supervised model and rebuild the Apriori rules."""

    from src.decision_tree import DECISION_TREE_MODEL_NAME, train_decision_tree
    from src.lightgbm_model import LIGHTGBM_MODEL_NAME, train_lightgbm
    from src.logistic_regression import LOGISTIC_REGRESSION_MODEL_NAME, train_logistic_regression
    from src.naive_bayes import NAIVE_BAYES_MODEL_NAME, train_naive_bayes
    from src.random_forest import RANDOM_FOREST_MODEL_NAME, train_random_forest
    from src.svm import SVM_MODEL_NAME, train_svm
    from src.xgboost_model import XGBOOST_MODEL_NAME, train_xgboost

    trainers = [
        ("Decision Tree", DECISION_TREE_MODEL_NAME, train_decision_tree),
        ("Naive Bayes", NAIVE_BAYES_MODEL_NAME, train_naive_bayes),
        ("Random Forest", RANDOM_FOREST_MODEL_NAME, train_random_forest),
        ("Logistic Regression", LOGISTIC_REGRESSION_MODEL_NAME, train_logistic_regression),
        ("SVM", SVM_MODEL_NAME, train_svm),
        ("XGBoost", XGBOOST_MODEL_NAME, train_xgboost),
        ("LightGBM", LIGHTGBM_MODEL_NAME, train_lightgbm),
    ]

    results: list[dict[str, Any]] = []
    for label, model_name, trainer in trainers:
        result = trainer(training_file)
        metadata = read_bundle_metadata(model_name) or {}
        results.append(
            {
                "model": label,
                "model_name": model_name,
                "accuracy": float(getattr(result, "accuracy", 0.0)),
                "trained_at": metadata.get("trained_at"),
            }
        )
    apriori_info = _rebuild_apriori_cache(training_file)
    results.append(
        {
            "model": "Apriori",
            "model_name": APRIORI_CACHE_NAME,
            "accuracy": None,
            "rules": apriori_info["rules"],
            "trained_at": apriori_info["trained_at"],
        }
    )
    try:
        from src.versioning import write_models_manifest

        write_models_manifest(training_file)
    except Exception:
        pass
    return results