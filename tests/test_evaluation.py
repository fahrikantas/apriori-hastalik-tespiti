# -*- coding: utf-8 -*-
"""Evaluation helpers tests."""

from __future__ import annotations

from src.evaluation import (
    cross_validation_summary,
    confusion_matrix_frame,
    per_class_metrics,
)


def test_cross_validation_has_all_models():
    cv = cross_validation_summary()
    assert set(cv["Model"]) == {
        "Decision Tree",
        "Naive Bayes",
        "Random Forest",
        "Logistic Regression",
        "SVM",
        "XGBoost",
        "LightGBM",
    }
    assert ((cv["CV Doğruluğu (%)"] > 0) & (cv["CV Doğruluğu (%)"] <= 100)).all()


def test_per_class_metrics_rows():
    table = per_class_metrics("Naive Bayes")
    assert len(table) >= 40
    assert {"Hastalık", "Precision", "Recall", "F1", "Destek"}.issubset(table.columns)


def test_confusion_matrix_shape():
    matrix = confusion_matrix_frame("Random Forest")
    assert matrix.shape[0] == matrix.shape[1]
    assert len(matrix) >= 41