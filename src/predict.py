"""Unified prediction pipeline for all trained disease models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.apriori_rules import (
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_LIFT,
    DEFAULT_MIN_SUPPORT,
    DEFAULT_MAX_LEN,
    build_symptom_to_disease_rules,
    recommend_diseases_from_symptoms,
    prepare_apriori_from_training,
)
from src.decision_tree import DECISION_TREE_MODEL_NAME, train_decision_tree
from src.lightgbm_model import LIGHTGBM_MODEL_NAME, train_lightgbm
from src.logistic_regression import LOGISTIC_REGRESSION_MODEL_NAME, train_logistic_regression
from src.naive_bayes import NAIVE_BAYES_MODEL_NAME, train_naive_bayes
from src.preprocess import (
    DEFAULT_DURATION_DAYS_WHEN_PRESENT,
    DEFAULT_SEVERITY_WHEN_PRESENT,
    preprocess_training_data,
)
from src.random_forest import RANDOM_FOREST_MODEL_NAME, train_random_forest
from src.svm import SVM_MODEL_NAME, train_svm
from src.xgboost_model import XGBOOST_MODEL_NAME, train_xgboost
from src.utils import (
    TARGET_COLUMN,
    normalize_symptom_name,
    resolve_model_path,
)

APRIORI_CACHE_NAME = "apriori_rules.pkl"


@st.cache_resource(show_spinner=False)
def _get_cached_preprocessed(training_file: str = "Training.csv") -> Any:
    """Cache the cleaned training data and symptom columns per session."""
    return preprocess_training_data(training_file)


@st.cache_resource(show_spinner=False)
def _get_cached_models(training_file: str = "Training.csv") -> dict[str, Any]:
    """Load every fitted model bundle once and reuse it across predictions."""
    return {
        "dt": _ensure_model_bundle(DECISION_TREE_MODEL_NAME, lambda: train_decision_tree(training_file)),
        "nb": _ensure_model_bundle(NAIVE_BAYES_MODEL_NAME, lambda: train_naive_bayes(training_file)),
        "rf": _ensure_model_bundle(RANDOM_FOREST_MODEL_NAME, lambda: train_random_forest(training_file)),
        "lr": _ensure_model_bundle(LOGISTIC_REGRESSION_MODEL_NAME, lambda: train_logistic_regression(training_file)),
        "svm": _ensure_model_bundle(SVM_MODEL_NAME, lambda: train_svm(training_file)),
        "xgb": _ensure_model_bundle(XGBOOST_MODEL_NAME, lambda: train_xgboost(training_file)),
        "lgb": _ensure_model_bundle(LIGHTGBM_MODEL_NAME, lambda: train_lightgbm(training_file)),
    }


@st.cache_resource(show_spinner=False)
def _get_cached_apriori_rules(
    apriori_params: AprioriParams | None = None,
    training_file: str = "Training.csv",
) -> pd.DataFrame:
    """Load or build Apriori rules once per parameter set."""
    params = apriori_params or AprioriParams()
    cache_path = resolve_model_path(APRIORI_CACHE_NAME)
    rules: pd.DataFrame
    if cache_path.exists() and _cache_matches_fingerprint(cache_path, training_file):
        apriori_bundle = joblib.load(cache_path)
        rules = apriori_bundle.get("rules", pd.DataFrame())
        cache_is_raw_rules = not rules.empty and "antecedents" in rules.columns and hasattr(rules.iloc[0]["antecedents"], "issubset")
        params_ok = _params_match(apriori_bundle, params)
        if rules.empty or not apriori_bundle.get("ready", False) or not cache_is_raw_rules or not params_ok:
            apriori_result = prepare_apriori_from_training(
                training_file,
                min_support=params.min_support,
                min_confidence=params.min_confidence,
                min_lift=params.min_lift,
                max_len=params.max_len,
            )
            rules = build_symptom_to_disease_rules(apriori_result.rules)
            _dump_apriori_cache(cache_path, rules, params, training_file)
    else:
        apriori_result = prepare_apriori_from_training(
            training_file,
            min_support=params.min_support,
            min_confidence=params.min_confidence,
            min_lift=params.min_lift,
            max_len=params.max_len,
        )
        rules = build_symptom_to_disease_rules(apriori_result.rules)
        _dump_apriori_cache(cache_path, rules, params, training_file)
    return rules


@dataclass(frozen=True)
class AprioriParams:
    """User-configurable Apriori mining thresholds."""

    min_support: float = DEFAULT_MIN_SUPPORT
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    min_lift: float = DEFAULT_MIN_LIFT
    max_len: int = DEFAULT_MAX_LEN


@dataclass(frozen=True)
class PredictionBundle:
    """Combined output returned by the unified prediction pipeline."""

    apriori_rules: pd.DataFrame
    decision_tree_prediction: str
    naive_bayes_prediction: str
    naive_bayes_probabilities: pd.DataFrame
    random_forest_prediction: str
    logistic_regression_prediction: str
    svm_prediction: str
    xgboost_prediction: str
    lightgbm_prediction: str
    symptom_vector: pd.DataFrame
    # Ensemble outputs
    ensemble_prediction: str | None = None
    ensemble_probabilities: pd.DataFrame | None = None
    # Differential diagnosis (top-N across all models, accuracy-weighted)
    differential_diagnosis: list[dict[str, Any]] | None = None
    # Out-of-distribution check
    ood: dict[str, Any] | None = None


#: Suffixes of the synthetic severity/duration columns appended to every raw
#: symptom column. Those columns are excluded from the out-of-distribution
#: math: overlap counts are defined over the binary "is symptom present" space
#: only, not over the derived severity/duration encodings.
_DERIVED_FEATURE_SUFFIXES = ("_severity", "_duration")


def _binary_symptom_columns(symptom_columns: list[str]) -> list[str]:
    """Return only the raw binary symptom columns, dropping derived ones."""
    return [
        column
        for column in symptom_columns
        if not column.endswith(_DERIVED_FEATURE_SUFFIXES)
    ]


def detect_out_of_distribution(
    symptom_vector: pd.DataFrame,
    training_frame: pd.DataFrame,
    symptom_columns: list[str],
    overlap_threshold: float = 0.5,
) -> dict[str, Any]:
    """Check whether the selected symptom combination resembles training data.

    The nearest training row is found by the largest overlap between the
    selected binary symptoms and each training row's active symptoms. When no
    row contains most of the selected symptoms together, the combination is
    reported as out-of-distribution so the UI can warn the user instead of
    blindly producing a disease name. Severity/duration columns are ignored.
    """

    binary_columns = _binary_symptom_columns(symptom_columns)
    if not binary_columns:
        binary_columns = [column for column in symptom_columns if column in training_frame.columns]
    input_vector = symptom_vector[binary_columns].iloc[0].values.astype(int)
    training_matrix = training_frame[binary_columns].to_numpy().astype(int)

    overlaps = training_matrix @ input_vector
    best_index = int(np.argmax(overlaps))
    best_overlap = float(overlaps[best_index])
    input_size = int(input_vector.sum())

    union = training_matrix.sum(axis=1) + input_size - overlaps
    jaccard = np.divide(
        overlaps,
        union,
        out=np.zeros_like(overlaps, dtype=float),
        where=union > 0,
    )

    overlap_fraction = best_overlap / input_size if input_size > 0 else 0.0
    if input_size == 1:
        is_ood = False
    else:
        is_ood = overlap_fraction < overlap_threshold or best_overlap < 2

    return {
        "is_ood": is_ood,
        "overlap_fraction": round(float(overlap_fraction), 3),
        "max_jaccard": round(float(jaccard[best_index]), 3),
        "nearest_disease": str(training_frame[TARGET_COLUMN].iloc[best_index]),
        "threshold": overlap_threshold,
    }


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


def _build_symptom_vector(
    selected_symptoms: list[str],
    feature_columns: list[str],
    severity_map: dict[str, int] | None = None,
    duration_map: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Convert selected symptoms into a single-row feature matrix.

    Supports optional severity and duration maps keyed by normalized symptom name.
    """

    severity_map = {normalize_symptom_name(k): int(v) for k, v in (severity_map or {}).items()}
    duration_map = {normalize_symptom_name(k): int(v) for k, v in (duration_map or {}).items()}
    normalized_selection = {normalize_symptom_name(symptom) for symptom in selected_symptoms}

    row: dict[str, int] = {}
    for column in feature_columns:
        if column.endswith("_severity"):
            base = column[: -len("_severity")]
            if base in normalized_selection:
                row[column] = int(severity_map.get(base, DEFAULT_SEVERITY_WHEN_PRESENT))
            else:
                row[column] = 0
        elif column.endswith("_duration"):
            base = column[: -len("_duration")]
            if base in normalized_selection:
                row[column] = int(duration_map.get(base, DEFAULT_DURATION_DAYS_WHEN_PRESENT))
            else:
                row[column] = 0
        else:
            # binary symptom column
            row[column] = int(column in normalized_selection)
    return pd.DataFrame([row], columns=feature_columns)


def _predict_label(bundle: dict[str, Any], symptom_vector: pd.DataFrame) -> str:
    """Predict a disease label from a fitted model bundle."""

    model = bundle["model"]
    label_encoder = bundle["label_encoder"]
    encoded_prediction = model.predict(symptom_vector)
    return str(label_encoder.inverse_transform(encoded_prediction)[0])


def _predict_probabilities(bundle: dict[str, Any], symptom_vector: pd.DataFrame) -> pd.DataFrame:
    """Return class probabilities for a probability-capable model.

    If the model does not implement predict_proba this will raise. Callers
    should catch and fallback when necessary.
    """

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


def _get_model_prob_vector(bundle: dict[str, Any], symptom_vector: pd.DataFrame, canonical_classes: list[str]) -> np.ndarray:
    """Return a probability vector aligned to canonical_classes for a single model bundle.

    If the model does not support probabilities, returns a one-hot vector for
    the model's predicted label.
    """

    model = bundle["model"]
    label_encoder = bundle["label_encoder"]
    probs = np.zeros(len(canonical_classes), dtype=float)
    try:
        raw_probs = model.predict_proba(symptom_vector)[0]
        model_classes = list(label_encoder.inverse_transform(np.arange(len(raw_probs))))
        for idx, cls in enumerate(model_classes):
            if cls in canonical_classes:
                probs[canonical_classes.index(cls)] = float(raw_probs[idx])
    except Exception:
        # fallback: use hard prediction
        pred = _predict_label(bundle, symptom_vector)
        if pred in canonical_classes:
            probs[canonical_classes.index(pred)] = 1.0
    return probs


def _compute_weighted_ensemble(
    bundles: list[dict[str, Any]],
    symptom_vector: pd.DataFrame,
    canonical_classes: list[str],
) -> tuple[np.ndarray, list[float]]:
    """Compute weighted soft-voting probabilities and return (probs, weights).

    Weights are taken from bundle.get('accuracy', 0.0). If all weights are zero,
    equal weights are used.
    """

    vectors: list[np.ndarray] = []
    weights: list[float] = []
    for bundle in bundles:
        vectors.append(_get_model_prob_vector(bundle, symptom_vector, canonical_classes))
        weights.append(float(bundle.get("accuracy", 0.0) or 0.0))

    weights = np.array(weights, dtype=float)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()

    stacked = np.vstack(vectors)  # shape (n_models, n_classes)
    ensemble = (weights[:, None] * stacked).sum(axis=0)
    return ensemble, weights.tolist()


def build_differential_diagnosis(
    canonical_classes: list[str],
    ensemble_probs: np.ndarray,
    bundles: list[dict[str, Any]],
    model_names: list[str],
    symptom_vector: pd.DataFrame,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Rank the top-N likely diseases across all models for comorbidity support.

    Uses the accuracy-weighted soft-voting probabilities from the ensemble, and
    annotates every candidate with the models that explicitly voted for it. This
    mirrors how a clinician compiles a differential list instead of a single
    hard label.
    """

    hard_predictions = [_predict_label(bundle, symptom_vector) for bundle in bundles]
    order = np.argsort(ensemble_probs)[::-1]
    results: list[dict[str, Any]] = []
    for index in order[:top_n]:
        disease = canonical_classes[int(index)]
        supporters = [
            name for name, prediction in zip(model_names, hard_predictions) if prediction == disease
        ]
        results.append(
            {
                "disease": disease,
                "score_pct": round(float(ensemble_probs[int(index)]) * 100, 2),
                "support_count": len(supporters),
                "supporting_models": supporters,
            }
        )
    return results


def _params_match(cached: dict[str, Any], params: AprioriParams) -> bool:
    """Return True when cached Apriori params match the requested values."""

    return (
        float(cached.get("min_support", DEFAULT_MIN_SUPPORT)) == params.min_support
        and float(cached.get("min_confidence", DEFAULT_MIN_CONFIDENCE)) == params.min_confidence
        and float(cached.get("min_lift", DEFAULT_MIN_LIFT)) == params.min_lift
        and int(cached.get("max_len", DEFAULT_MAX_LEN)) == params.max_len
    )


def _cache_matches_fingerprint(cache_path: Path, training_file: str) -> bool:
    """Return True when the cached Apriori bundle belongs to the current file."""

    try:
        bundle = joblib.load(cache_path)
        from src.model_metadata import compute_fingerprint

        metadata = bundle.get("metadata") or {}
        return metadata.get("data_fingerprint") == compute_fingerprint(training_file).get("data_fingerprint")
    except Exception:
        return False


def get_apriori_recommendations(
    selected_symptoms: list[str],
    apriori_params: AprioriParams | None = None,
    training_file: str = "Training.csv",
) -> pd.DataFrame:
    """Load or build Apriori rules and return the matching disease recommendations."""

    rules = _get_cached_apriori_rules(apriori_params, training_file)
    return recommend_diseases_from_symptoms(selected_symptoms, rules)


def _dump_apriori_cache(cache_path: Path, rules: pd.DataFrame, params: AprioriParams, training_file: str) -> None:
    """Persist Apriori rules together with the current data fingerprint."""

    from src.model_metadata import compute_fingerprint

    metadata: dict[str, Any] = {}
    try:
        metadata = compute_fingerprint(training_file)
    except Exception:
        metadata = {}
    joblib.dump(
        {
            "rules": rules,
            "ready": True,
            "metadata": metadata,
            "min_support": params.min_support,
            "min_confidence": params.min_confidence,
            "min_lift": params.min_lift,
            "max_len": params.max_len,
        },
        cache_path,
    )


def predict_from_symptoms(
    selected_symptoms: list[str],
    training_file: str = "Training.csv",
    apriori_params: AprioriParams | None = None,
    severity_map: dict[str, int] | None = None,
    duration_map: dict[str, int] | None = None,
) -> PredictionBundle:
    """Run all model predictions for the supplied symptom list.

    severity_map and duration_map are optional mappings from normalized symptom
    name to integer severity/duration values.
    """

    if not selected_symptoms:
        raise ValueError("At least one symptom must be selected.")

    preprocessed = _get_cached_preprocessed(training_file)
    symptom_vector = _build_symptom_vector(selected_symptoms, preprocessed.symptom_columns, severity_map, duration_map)
    ood = detect_out_of_distribution(
        symptom_vector,
        preprocessed.frame,
        preprocessed.symptom_columns,
    )

    models = _get_cached_models(training_file)
    decision_tree_bundle = models["dt"]
    naive_bayes_bundle = models["nb"]
    random_forest_bundle = models["rf"]
    logistic_regression_bundle = models["lr"]
    svm_bundle = models["svm"]
    xgboost_bundle = models["xgb"]
    lightgbm_bundle = models["lgb"]

    apriori_rules = get_apriori_recommendations(selected_symptoms, apriori_params, training_file)
    decision_tree_prediction = _predict_label(decision_tree_bundle, symptom_vector)
    naive_bayes_prediction = _predict_label(naive_bayes_bundle, symptom_vector)
    random_forest_prediction = _predict_label(random_forest_bundle, symptom_vector)
    logistic_regression_prediction = _predict_label(logistic_regression_bundle, symptom_vector)
    svm_prediction = _predict_label(svm_bundle, symptom_vector)
    xgboost_prediction = _predict_label(xgboost_bundle, symptom_vector)
    lightgbm_prediction = _predict_label(lightgbm_bundle, symptom_vector)
    naive_bayes_probabilities = _predict_probabilities(naive_bayes_bundle, symptom_vector)

    # Build canonical class ordering from the training data so probabilities align
    canonical_classes = sorted(preprocessed.frame[TARGET_COLUMN].unique())

    bundles_for_ensemble = [
        decision_tree_bundle,
        naive_bayes_bundle,
        random_forest_bundle,
        logistic_regression_bundle,
        svm_bundle,
        xgboost_bundle,
        lightgbm_bundle,
    ]
    ensemble_model_names = [
        "Decision Tree",
        "Naive Bayes",
        "Random Forest",
        "Logistic Regression",
        "SVM",
        "XGBoost",
        "LightGBM",
    ]

    ensemble_probs, model_weights = _compute_weighted_ensemble(bundles_for_ensemble, symptom_vector, canonical_classes)
    ensemble_prediction = canonical_classes[int(np.argmax(ensemble_probs))]
    ensemble_df = pd.DataFrame({TARGET_COLUMN: canonical_classes, "probability_pct": (ensemble_probs * 100).round(2)})
    differential_diagnosis = build_differential_diagnosis(
        canonical_classes,
        ensemble_probs,
        bundles_for_ensemble,
        ensemble_model_names,
        symptom_vector,
    )

    return PredictionBundle(
        apriori_rules=apriori_rules,
        decision_tree_prediction=decision_tree_prediction,
        naive_bayes_prediction=naive_bayes_prediction,
        naive_bayes_probabilities=naive_bayes_probabilities,
        random_forest_prediction=random_forest_prediction,
        logistic_regression_prediction=logistic_regression_prediction,
        svm_prediction=svm_prediction,
        xgboost_prediction=xgboost_prediction,
        lightgbm_prediction=lightgbm_prediction,
        symptom_vector=symptom_vector,
        ensemble_prediction=ensemble_prediction,
        ensemble_probabilities=ensemble_df,
        differential_diagnosis=differential_diagnosis,
        ood=ood,
    )
