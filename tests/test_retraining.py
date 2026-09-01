# -*- coding: utf-8 -*-
"""Tests for model metadata, fingerprints and model freshness."""

from __future__ import annotations

import pandas as pd

from src.model_metadata import (
    build_bundle_metadata,
    compute_fingerprint,
    model_statuses,
    read_bundle_metadata,
    retrain_all_models,
)
from src.predict import APRIORI_CACHE_NAME


def _supervised_names() -> list[str]:
    from src.decision_tree import DECISION_TREE_MODEL_NAME
    from src.lightgbm_model import LIGHTGBM_MODEL_NAME
    from src.logistic_regression import LOGISTIC_REGRESSION_MODEL_NAME
    from src.naive_bayes import NAIVE_BAYES_MODEL_NAME
    from src.random_forest import RANDOM_FOREST_MODEL_NAME
    from src.svm import SVM_MODEL_NAME
    from src.xgboost_model import XGBOOST_MODEL_NAME

    return [
        DECISION_TREE_MODEL_NAME,
        NAIVE_BAYES_MODEL_NAME,
        RANDOM_FOREST_MODEL_NAME,
        LOGISTIC_REGRESSION_MODEL_NAME,
        SVM_MODEL_NAME,
        XGBOOST_MODEL_NAME,
        LIGHTGBM_MODEL_NAME,
    ]


def test_fingerprint_is_deterministic():
    first = compute_fingerprint()
    second = compute_fingerprint()
    assert first["data_fingerprint"] == second["data_fingerprint"]
    assert first["data_rows"] > 0
    assert first["class_count"] > 10


def test_metadata_builder_shape():
    frame = pd.DataFrame(
        {
            "prognosis": ["A", "B", "A"],
            "itching": [1, 0, 1],
            "skin_rash": [0, 1, 0],
        }
    )
    metadata = build_bundle_metadata(frame, ["itching", "skin_rash"])
    assert "data_fingerprint" in metadata
    assert "trained_at" in metadata
    assert metadata["data_rows"] == 3
    assert metadata["class_count"] == 2


def test_retrain_all_models_rebuilds_everything():
    results = retrain_all_models()
    models = {entry["model"] for entry in results}
    assert {"Decision Tree", "Naive Bayes", "Random Forest", "Logistic Regression", "SVM", "XGBoost", "LightGBM", "Apriori"} <= models
    apriori_entry = next(entry for entry in results if entry["model"] == "Apriori")
    assert apriori_entry["rules"] > 0


def test_model_statuses_all_fresh_after_retrain():
    titles = retrain_all_models()
    statuses = model_statuses(_supervised_names())
    assert len(statuses) == 7
    assert all(status["fresh"] for status in statuses.values()), statuses
    assert read_bundle_metadata(_supervised_names()[0]) is not None