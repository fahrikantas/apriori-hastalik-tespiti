# -*- coding: utf-8 -*-
"""Local telemetry tests."""

from __future__ import annotations

from src.telemetry import clear_telemetry, log_prediction, model_agreement, read_records, summarize_disagreements


def test_model_agreement_math():
    assert model_agreement(["A", "A", "B", "C", "D"]) == {
        "model_count": 5,
        "unique_predictions": 4,
        "agreement_fraction": 0.4,
        "disagreement": True,
    }
    assert model_agreement(["A", "A", "A", "A", "A"])["disagreement"] is False
    assert model_agreement(["A", "A", "A", "B", "C"])["disagreement"] is False


def test_log_and_read_roundtrip():
    clear_telemetry()
    log_prediction(
        training_file="Training.csv",
        symptoms=["itching", "pain"],
        model_predictions={"decision_tree": "Fungal", "svm": "Allergy"},
        final_prediction="Fungal",
    )
    records = read_records()
    assert len(records) == 1
    assert records[0]["final_prediction"] == "Fungal"
    assert records[0]["symptoms"] == ["itching", "pain"]
    clear_telemetry()


def test_summarize_counts_disagreements():
    clear_telemetry()
    log_prediction("Training.csv", ["a", "b"], {"dt": "X", "nb": "Y", "rf": "X", "lr": "X", "svm": "X"})
    log_prediction("Training.csv", ["c", "d"], {"dt": "A", "nb": "B", "rf": "C", "lr": "D", "svm": "E"})
    summary = summarize_disagreements()
    assert summary["record_count"] == 2
    assert summary["disagreement_count"] == 1
    clear_telemetry()