"""Leak-aware train/test splitting for honest model evaluation.

Random splits of this dataset place near-identical rows of the same disease
into both the train and test sides, so models memorize disease-defining
patterns and report unrealistically high accuracy. To prevent that, every row
is assigned a group id from its symptom signature and ``StratifiedGroupKFold``
keeps all rows of one group (exact and near-duplicate signatures) on the same
side of the split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_CV_FOLDS = 5

#: Rows of the same disease whose signatures differ by at most this many
#: symptoms are near-duplicates and must never straddle the split either.
NEAR_DUPLICATE_DISTANCE = 2

#: Suffixes of synthetic severity/duration feature columns. Those columns are
#: model inputs but are not part of the "symptom signature" used for leak
#: grouping; a row should never be split from another just because the encoded
#: severity of a shared symptom differs.
DERIVED_FEATURE_SUFFIXES = ("_severity", "_duration")


def _signature_features(features: pd.DataFrame) -> pd.DataFrame:
    """Return only the raw binary symptom columns used for signature grouping."""
    return features[[
        column
        for column in features.columns
        if not column.endswith(DERIVED_FEATURE_SUFFIXES)
    ]]


def build_leak_groups(features: pd.DataFrame) -> np.ndarray:
    """Assign one group id to each unique symptom signature.

    Rows that share the exact symptom pattern (regardless of symptom severity,
    duration, or disease label) belong to the same group, so no split can ever
    leak a feature vector from training into the test set.
    """

    signature = _signature_features(features).astype(str).agg("|".join, axis=1)
    unique = pd.unique(signature)
    mapping = {value: index for index, value in enumerate(unique)}
    return signature.map(mapping).to_numpy()


def _near_duplicate_groups(features: pd.DataFrame, max_distance: int = NEAR_DUPLICATE_DISTANCE) -> np.ndarray:
    """Group rows whose (binary) signatures differ by at most ``max_distance`` symptoms.

    Uses a greedy union-find over sorted Hamming-neighbour pairs, so identical
    signatures collapse into one group and near-identical signatures (same
    disease, one or two symptom differences) cannot be split between train and
    test. Derived severity/duration columns are ignored here; only the binary
    symptom signature matters for leak grouping.
    """

    signature_frame = _signature_features(features)
    unique_sigs = signature_frame.drop_duplicates().to_numpy()
    parent = np.arange(len(unique_sigs))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for index in range(len(unique_sigs)):
        diffs = (unique_sigs != unique_sigs[index]).sum(axis=1)
        for neighbour in np.flatnonzero((diffs > 0) & (diffs <= max_distance)):
            union(index, int(neighbour))

    root_lookup = {tuple(sig): find(index) for index, sig in enumerate(unique_sigs)}
    return signature_frame.apply(
        lambda row: root_lookup[tuple(row)],
        axis=1,
    ).to_numpy()


def build_split_groups(features: pd.DataFrame) -> np.ndarray:
    """Group ids that cover both exact and near-duplicate signatures."""

    return _near_duplicate_groups(features)


def split_train_test(
    features: pd.DataFrame,
    target: pd.Series | np.ndarray,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified, group-aware train/test split with honest semantics.

    Groups (exact and near-duplicate signatures) are kept intact so no test
    row is memorized during training, while stratification ensures every class
    appears on both sides of the split so the held-out data remains solvable.
    """

    features = features.reset_index(drop=True)
    if isinstance(target, np.ndarray):
        target = pd.Series(target, index=features.index)
    target = pd.Series(target.tolist(), index=features.index)
    frame = pd.concat([features, target.rename("__target__")], axis=1)
    frame = frame.drop_duplicates().reset_index(drop=True)
    features = frame.drop(columns="__target__")
    target = frame["__target__"]

    groups = build_split_groups(features)
    test_folds = int(round(1 / test_size)) if test_size > 0 else 5
    splitter = StratifiedGroupKFold(
        n_splits=test_folds,
        shuffle=True,
        random_state=random_state,
    )
    train_indices, test_indices = next(iter(splitter.split(features, target, groups=groups)))

    # Validate that the split contains overlapping class labels between
    # train and test. In some pathological datasets (very rare classes and
    # aggressive grouping), StratifiedGroupKFold can produce splits where the
    # train and test label sets do not overlap; this causes downstream
    # classifiers to report 0.0 accuracy because the model never sees the
    # test classes during training. When that happens, fall back to a more
    # forgiving stratified `train_test_split`.
    train_y = target.iloc[train_indices]
    test_y = target.iloc[test_indices]
    if not set(test_y.unique()).issubset(set(train_y.unique())):
        from sklearn.model_selection import train_test_split

        return train_test_split(features, target, test_size=test_size, random_state=random_state, stratify=target)

    return (
        features.iloc[train_indices].reset_index(drop=True),
        features.iloc[test_indices].reset_index(drop=True),
        target.iloc[train_indices].reset_index(drop=True),
        target.iloc[test_indices].reset_index(drop=True),
    )


def stratified_group_folds(
    features: pd.DataFrame,
    target: pd.Series,
    n_splits: int = DEFAULT_CV_FOLDS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> StratifiedGroupKFold:
    """A stratified CV splitter that also guards against symptom-signature leaks."""

    groups = build_split_groups(features)
    return StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    ).split(features, target, groups=groups)