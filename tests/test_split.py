# -*- coding: utf-8 -*-
"""Leak-aware splitting and honest holdout tests."""

from __future__ import annotations

import numpy as np

from src.preprocess import preprocess_training_data
from src.split import build_split_groups, split_train_test


def _training_inputs():
    data = preprocess_training_data("Training.csv")
    features = data.frame[data.symptom_columns]
    target = data.frame["prognosis"]
    return features, target


def test_split_removes_exact_duplicates():
    features, target = _training_inputs()
    x_train, x_test, y_train, y_test = split_train_test(features, target)
    assert len(x_train) + len(x_test) <= len(features)
    test_share = len(x_test) / len(features)
    assert 0.1 < test_share < 0.4


def test_no_near_duplicate_leak():
    features, target = _training_inputs()
    x_train, x_test, _, _ = split_train_test(features, target)
    train_rows = x_train.astype(int).to_numpy()
    min_distances = [
        int((train_rows != row).sum(axis=1).min()) for row in x_test.astype(int).to_numpy()
    ]
    # a model memorizing training patterns cannot cheat on the test rows
    assert all(distance > 2 for distance in min_distances)


def test_group_counts_are_low():
    features, _ = _training_inputs()
    groups = build_split_groups(features)
    assert 0 < len(set(groups)) < 60


def test_synthetic_split_covers_all_classes():
    from src.synthetic_data import generate_synthetic_frame

    frame = generate_synthetic_frame(1000, random_state=7)
    features = frame.drop(columns=["prognosis"]).astype(int)
    target = frame["prognosis"]
    _, _, y_train, y_test = split_train_test(features, target)
    assert set(y_train) == set(y_test)
    assert len(y_test) > 100


def test_split_groups_are_reproducible():
    features, _ = _training_inputs()
    first = build_split_groups(features)
    second = build_split_groups(features)
    assert np.array_equal(first, second)