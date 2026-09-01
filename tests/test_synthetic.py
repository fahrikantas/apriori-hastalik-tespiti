# -*- coding: utf-8 -*-
"""Synthetic dataset generator tests."""

from __future__ import annotations

from src.synthetic_data import generate_synthetic_frame


def test_synthetic_frame_shape():
    frame = generate_synthetic_frame(2000, random_state=42)
    assert len(frame) == (2000 // 41) * 41
    assert frame["prognosis"].nunique() == 41


def test_synthetic_is_deterministic():
    first = generate_synthetic_frame(500, random_state=1)
    second = generate_synthetic_frame(500, random_state=1)
    assert first.equals(second)


def test_synthetic_has_class_balance():
    frame = generate_synthetic_frame(2000, random_state=42)
    counts = frame["prognosis"].value_counts()
    assert counts.min() >= 40


def test_synthetic_feature_columns_are_binary():
    frame = generate_synthetic_frame(500, random_state=1)
    symptom_columns = frame.drop(columns=["prognosis"])
    assert set(symptom_columns.to_numpy().ravel()).issubset({0, 1})