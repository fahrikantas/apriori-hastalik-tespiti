"""Unified prediction pipeline for all trained disease models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.apriori_rules import (
    build_symptom_to_disease_rules,
    recommend_diseases_from_symptoms,
    prepare_apriori_from_training,
)
from src.decision_tree import DECISION_TREE_MODEL_NAME, train_decision_tree
from src.naive_bayes import NAIVE_BAYES_MODEL_NAME, train_naive_bayes
from src.preprocess import preprocess_training_data
from src.random_forest import RANDOM_FOREST_MODEL_NAME, train_random_forest
from src.utils import TARGET_COLUMN, normalize_symptom_name, resolve_model_path

APRIORI_CACHE_NAME = "apriori_rules.pkl"


@dataclass(frozen=True)
class PredictionBundle:
    """Combined output returned by the unified prediction pipeline."""

    apriori_rules: pd.DataFrame
    decision_tree_prediction: str
    naive_bayes_prediction: str
    naive_bayes_probabilities: pd.DataFrame
    random_forest_prediction: str
    symptom_vector: pd.DataFrame


def _load_model_bundle(model_name: str) -> dict[str, Any]:
    """Load a persisted supervised-learning bundle from disk."""

    model_path = resolve_model_path(model_name)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    return joblib.load(model_path)


def _ensure_model_bundle(model_name: str, trainer: Any) -> dict[str, Any]:
    """Load a bundle or train it if the artifact is missing."""

    model_path = resolve_model_path(model_name)
    if not model_path.exists():
        trainer()
    return _load_model_bundle(model_name)


def _build_symptom_vector(selected_symptoms: list[str], feature_columns: list[str]) -> pd.DataFrame:
    """Convert selected symptoms into a single-row feature matrix."""

    normalized_selection = {normalize_symptom_name(symptom) for symptom in selected_symptoms}
    row = {
        column: int(column in normalized_selection)
        for column in feature_columns
    }
    return pd.DataFrame([row], columns=feature_columns)


def _predict_label(bundle: dict[str, Any], symptom_vector: pd.DataFrame) -> str:
    """Predict a disease label from a fitted model bundle."""

    model = bundle["model"]
    label_encoder = bundle["label_encoder"]
    encoded_prediction = model.predict(symptom_vector)
    return str(label_encoder.inverse_transform(encoded_prediction)[0])


def _predict_probabilities(bundle: dict[str, Any], symptom_vector: pd.DataFrame) -> pd.DataFrame:
    """Return class probabilities for the Naive Bayes model."""

    model = bundle["model"]
    label_encoder = bundle["label_encoder"]
    probabilities = model.predict_proba(symptom_vector)[0]
    probability_frame = pd.DataFrame(
        {
            TARGET_COLUMN: label_encoder.inverse_transform(np.arange(len(probabilities))),
            "probability": probabilities,
        }
    ).sort_values(by="probability", ascending=False).reset_index(drop=True)
    probability_frame["probability_pct"] = (probability_frame["probability"] * 100).round(2)
    return probability_frame[[TARGET_COLUMN, "probability_pct"]]


def get_apriori_recommendations(selected_symptoms: list[str]) -> pd.DataFrame:
    """Load or build Apriori rules and return the matching disease recommendations."""

    cache_path = resolve_model_path(APRIORI_CACHE_NAME)
    rules: pd.DataFrame
    if cache_path.exists():
        apriori_bundle = joblib.load(cache_path)
        rules = apriori_bundle.get("rules", pd.DataFrame())
        cache_is_raw_rules = not rules.empty and "antecedents" in rules.columns and hasattr(rules.iloc[0]["antecedents"], "issubset")
        if rules.empty or not apriori_bundle.get("ready", False) or not cache_is_raw_rules:
            apriori_result = prepare_apriori_from_training()
            rules = build_symptom_to_disease_rules(apriori_result.rules)
            joblib.dump({"rules": rules, "ready": True}, cache_path)
    else:
        apriori_result = prepare_apriori_from_training()
        rules = build_symptom_to_disease_rules(apriori_result.rules)
        joblib.dump({"rules": rules, "ready": True}, cache_path)
    return recommend_diseases_from_symptoms(selected_symptoms, rules)


def predict_from_symptoms(selected_symptoms: list[str], training_file: str = "Training.csv") -> PredictionBundle:
    """Run all model predictions for the supplied symptom list."""

    if not selected_symptoms:
        raise ValueError("At least one symptom must be selected.")

    preprocessed = preprocess_training_data(training_file)
    symptom_vector = _build_symptom_vector(selected_symptoms, preprocessed.symptom_columns)

    decision_tree_bundle = _ensure_model_bundle(DECISION_TREE_MODEL_NAME, lambda: train_decision_tree(training_file))
    naive_bayes_bundle = _ensure_model_bundle(NAIVE_BAYES_MODEL_NAME, lambda: train_naive_bayes(training_file))
    random_forest_bundle = _ensure_model_bundle(RANDOM_FOREST_MODEL_NAME, lambda: train_random_forest(training_file))

    apriori_rules = get_apriori_recommendations(selected_symptoms)
    decision_tree_prediction = _predict_label(decision_tree_bundle, symptom_vector)
    naive_bayes_prediction = _predict_label(naive_bayes_bundle, symptom_vector)
    random_forest_prediction = _predict_label(random_forest_bundle, symptom_vector)
    naive_bayes_probabilities = _predict_probabilities(naive_bayes_bundle, symptom_vector)

    return PredictionBundle(
        apriori_rules=apriori_rules,
        decision_tree_prediction=decision_tree_prediction,
        naive_bayes_prediction=naive_bayes_prediction,
        naive_bayes_probabilities=naive_bayes_probabilities,
        random_forest_prediction=random_forest_prediction,
        symptom_vector=symptom_vector,
    )
