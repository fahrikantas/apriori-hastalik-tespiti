# -*- coding: utf-8 -*-
"""End-to-end prediction pipeline tests."""

from __future__ import annotations

import pandas as pd

from src.predict import predict_from_symptoms
from src.preprocess import preprocess_training_data


def test_pipeline_returns_all_predictions():
    bundle = predict_from_symptoms(["itching", "skin_rash"])
    assert bundle.decision_tree_prediction
    assert bundle.naive_bayes_prediction
    assert bundle.random_forest_prediction
    assert bundle.logistic_regression_prediction
    assert bundle.svm_prediction
    assert bundle.xgboost_prediction
    assert bundle.lightgbm_prediction


def test_naive_bayes_probabilities_sorted():
    bundle = predict_from_symptoms(["itching", "skin_rash"])
    probs = bundle.naive_bayes_probabilities
    assert isinstance(probs, pd.DataFrame)
    assert not probs.empty
    assert list(probs["probability_pct"]) == sorted(probs["probability_pct"], reverse=True)


def test_symptom_vector_shape():
    bundle = predict_from_symptoms(["itching", "skin_rash"])
    preprocessed = preprocess_training_data()
    assert list(bundle.symptom_vector.columns) == preprocessed.symptom_columns


def test_preprocess_adds_severity_and_duration_columns_for_binary_symptoms():
    preprocessed = preprocess_training_data()
    assert "itching_severity" in preprocessed.frame.columns
    assert "itching_duration" in preprocessed.frame.columns
    assert preprocessed.frame["itching_severity"].dtype.kind in "iu"
    assert preprocessed.frame["itching_duration"].dtype.kind in "iu"


def test_selected_symptom_defaults_are_applied_for_severity_and_duration():
    bundle = predict_from_symptoms(["itching"])
    row = bundle.symptom_vector.iloc[0]
    assert int(row["itching"]) == 1
    assert int(row["itching_severity"]) == 2
    assert int(row["itching_duration"]) == 3


def test_selected_symptom_maps_override_defaults_for_severity_and_duration():
    bundle = predict_from_symptoms(
        ["itching"],
        severity_map={"itching": 3},
        duration_map={"itching": 7},
    )
    row = bundle.symptom_vector.iloc[0]
    assert int(row["itching_severity"]) == 3
    assert int(row["itching_duration"]) == 7


def test_pipeline_requires_selection():
    try:
        predict_from_symptoms([])
    except ValueError:
        pass
    else:
        raise AssertionError("predict_from_symptoms([]) must raise ValueError")