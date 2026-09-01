# -*- coding: utf-8 -*-
"""Out-of-distribution detection tests."""

from __future__ import annotations

from src.predict import predict_from_symptoms


def test_single_symptom_never_ood():
    bundle = predict_from_symptoms(["itching"])
    assert bundle.ood["is_ood"] is False


def test_real_pair_not_ood():
    bundle = predict_from_symptoms(["itching", "skin_rash"])
    assert bundle.ood["is_ood"] is False


def test_weird_pair_is_ood():
    bundle = predict_from_symptoms(["slurred_speech", "blister"])
    assert bundle.ood["is_ood"] is True


def test_random_mix_is_ood():
    bundle = predict_from_symptoms(
        ["cough", "blackheads", "polyuria", "spinning_movements", "bruising", "stiff_neck"]
    )
    assert bundle.ood["is_ood"] is True


def test_detector_returns_metadata():
    from src.predict import predict_from_symptoms

    bundle = predict_from_symptoms(["itching"])
    meta = bundle.ood
    assert set(meta) == {"is_ood", "overlap_fraction", "max_jaccard", "nearest_disease", "threshold"}
    assert 0.0 <= meta["overlap_fraction"] <= 1.0