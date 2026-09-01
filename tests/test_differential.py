# -*- coding: utf-8 -*-
"""Differential diagnosis (comorbidity / top-N) tests."""

from __future__ import annotations

from src.predict import build_differential_diagnosis, predict_from_symptoms


def test_differential_is_present():
    bundle = predict_from_symptoms(["itching", "skin_rash"])
    assert bundle.differential_diagnosis
    assert len(bundle.differential_diagnosis) <= 5
    assert len(bundle.differential_diagnosis) >= 1


def test_differential_entry_schema():
    bundle = predict_from_symptoms(["itching", "skin_rash"])
    entry = bundle.differential_diagnosis[0]
    assert set(entry) == {"disease", "score_pct", "support_count", "supporting_models"}
    assert 0.0 <= entry["score_pct"] <= 100.0
    assert entry["support_count"] >= 0
    assert isinstance(entry["supporting_models"], list)


def test_top_candidate_matches_ensemble():
    bundle = predict_from_symptoms(["high_fever", "stiff_neck", "vomiting"])
    top = bundle.differential_diagnosis[0]["disease"]
    assert top == bundle.ensemble_prediction


def test_scores_are_descending():
    bundle = predict_from_symptoms(["cough", "phlegm", "high_fever"])
    scores = [entry["score_pct"] for entry in bundle.differential_diagnosis]
    assert scores == sorted(scores, reverse=True)


def test_support_count_upper_bound():
    bundle = predict_from_symptoms(["itching"])
    for entry in bundle.differential_diagnosis:
        assert 0 <= entry["support_count"] <= 7


def test_builder_is_reproducible():
    preprocessed = None
    from src.preprocess import preprocess_training_data

    preprocessed = preprocess_training_data("Training.csv")
    symptom_vector = preprocessed.frame.iloc[[0]]
    canonical_classes = sorted(preprocessed.frame["prognosis"].unique())
    import numpy as np

    probs = np.full(len(canonical_classes), 0.01)
    probs[0] = 0.5
    first = build_differential_diagnosis(
        canonical_classes, probs, [], [], symptom_vector, top_n=3
    )
    second = build_differential_diagnosis(
        canonical_classes, probs, [], [], symptom_vector, top_n=3
    )
    assert first == second