"""Data preprocessing utilities for the disease prediction project.

This module reads the training dataset, standardizes column names, validates the
schema, and prepares the feature matrix used by all downstream models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.utils import (
    TARGET_COLUMN,
    deduplicate_preserve_order,
    ensure_columns_exist,
    load_dataset,
    normalize_symptom_name,
)

DEFAULT_SEVERITY_WHEN_PRESENT = 2
DEFAULT_DURATION_DAYS_WHEN_PRESENT = 3


@dataclass(frozen=True)
class PreprocessingSummary:
    """Compact report describing the state of the raw dataset."""

    row_count: int
    column_count: int
    missing_values: int
    duplicate_rows: int
    target_class_count: int
    symptom_column_count: int


@dataclass(frozen=True)
class PreprocessedData:
    """Container for the cleaned training data and derived metadata."""

    frame: pd.DataFrame
    symptom_columns: list[str]
    target_column: str = TARGET_COLUMN


def load_training_data(file_name: str = "Training.csv") -> pd.DataFrame:
    """Load the raw training dataset from disk."""

    return load_dataset(file_name)


def standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to a stable, machine-friendly format."""

    renamed_columns = {
        column: normalize_symptom_name(column) if column != TARGET_COLUMN else TARGET_COLUMN
        for column in frame.columns
    }
    standardized_frame = frame.rename(columns=renamed_columns)
    standardized_frame.columns = deduplicate_preserve_order(standardized_frame.columns)
    return standardized_frame


def identify_symptom_columns(frame: pd.DataFrame) -> list[str]:
    """Return every feature column except the prognosis label."""

    ensure_columns_exist(frame, [TARGET_COLUMN])
    return [column for column in frame.columns if column != TARGET_COLUMN]


def _is_binary_feature(series: pd.Series) -> bool:
    """Return True when a feature contains only 0/1 values."""

    if series.empty:
        return False
    # Coerce to numeric and drop non-finite values before testing
    values = pd.to_numeric(series, errors="coerce")
    # Remove NaN and infinite values for the binary check
    finite = values.replace([pd.NA, None], pd.NA).dropna()
    finite = finite[finite.apply(pd.api.types.is_number) | finite.map(lambda x: pd.notna(x))]
    finite = finite[pd.to_numeric(finite, errors="coerce").replace([float("inf"), float("-inf")], pd.NA).dropna()]
    if finite.empty:
        return False
    try:
        unique_values = set(pd.to_numeric(finite, errors="coerce").astype(int).unique())
    except Exception:
        return False
    return unique_values.issubset({0, 1})


def _ensure_severity_duration_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Create optional severity/duration feature columns for binary symptoms."""

    expanded_frame = frame.copy()
    binary_columns = [
        column
        for column in identify_symptom_columns(expanded_frame)
        if not column.endswith(("_severity", "_duration")) and _is_binary_feature(expanded_frame[column])
    ]
    derived_values: dict[str, list[int]] = {}
    for column in binary_columns:
        severity_column = f"{column}_severity"
        duration_column = f"{column}_duration"
        # Safely coerce numeric values and treat non-finite as 0 (absent)
        base_values = pd.to_numeric(expanded_frame[column], errors="coerce")
        base_values = base_values.replace([float("inf"), float("-inf")], pd.NA).fillna(0)
        base_values = base_values.astype(int)
        present_values = (base_values > 0).astype(int)
        if severity_column not in expanded_frame.columns:
            derived_values[severity_column] = (present_values * DEFAULT_SEVERITY_WHEN_PRESENT).tolist()
        if duration_column not in expanded_frame.columns:
            derived_values[duration_column] = (present_values * DEFAULT_DURATION_DAYS_WHEN_PRESENT).tolist()
    if derived_values:
        expanded_frame = pd.concat([expanded_frame, pd.DataFrame(derived_values, index=expanded_frame.index)], axis=1)
    return expanded_frame


def clean_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean the training data so it can be consumed by all models."""

    cleaned_frame = standardize_columns(frame.copy())
    ensure_columns_exist(cleaned_frame, [TARGET_COLUMN])
    cleaned_frame = _ensure_severity_duration_columns(cleaned_frame)

    symptom_columns = identify_symptom_columns(cleaned_frame)
    cleaned_frame[symptom_columns] = cleaned_frame[symptom_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    cleaned_frame[symptom_columns] = cleaned_frame[symptom_columns].fillna(0).astype(int)
    cleaned_frame[TARGET_COLUMN] = cleaned_frame[TARGET_COLUMN].astype(str).str.strip()
    cleaned_frame = cleaned_frame.drop_duplicates().reset_index(drop=True)
    return cleaned_frame


def inspect_training_data(frame: pd.DataFrame) -> PreprocessingSummary:
    """Produce a lightweight summary of the raw training dataset."""

    missing_values = int(frame.isna().sum().sum())
    duplicate_rows = int(frame.duplicated().sum())
    target_class_count = int(frame[TARGET_COLUMN].nunique()) if TARGET_COLUMN in frame.columns else 0
    symptom_column_count = max(len(frame.columns) - 1, 0)
    return PreprocessingSummary(
        row_count=int(frame.shape[0]),
        column_count=int(frame.shape[1]),
        missing_values=missing_values,
        duplicate_rows=duplicate_rows,
        target_class_count=target_class_count,
        symptom_column_count=symptom_column_count,
    )


def preprocess_training_data(file_name: str = "Training.csv") -> PreprocessedData:
    """Load, clean, and describe the training dataset."""

    raw_frame = load_training_data(file_name)
    cleaned_frame = clean_training_frame(raw_frame)
    symptom_columns = identify_symptom_columns(cleaned_frame)
    return PreprocessedData(frame=cleaned_frame, symptom_columns=symptom_columns)


def frame_to_transactions(frame: pd.DataFrame) -> list[list[str]]:
    """Convert the cleaned frame into Apriori-ready transactions.

    Each transaction contains active symptom labels and the disease label as a
    final item so association rules can produce disease recommendations.
    """

    symptom_columns = [
        column
        for column in identify_symptom_columns(frame)
        if not column.endswith(("_severity", "_duration")) and _is_binary_feature(frame[column])
    ]
    transactions: list[list[str]] = []
    for _, row in frame.iterrows():
        active_items = [column for column in symptom_columns if int(row[column]) == 1]
        active_items.append(f"disease_{row[TARGET_COLUMN]}")
        transactions.append(active_items)
    return transactions


def get_available_symptoms(frame: pd.DataFrame) -> list[str]:
    """Return the sorted symptom vocabulary expected by the UI."""

    symptom_columns = [
        column
        for column in identify_symptom_columns(frame)
        if not column.endswith(("_severity", "_duration")) and _is_binary_feature(frame[column])
    ]
    return sorted(symptom_columns)


def build_dataset_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    """Return a dictionary with the most relevant preprocessing diagnostics."""

    summary = inspect_training_data(frame)
    return {
        "row_count": summary.row_count,
        "column_count": summary.column_count,
        "missing_values": summary.missing_values,
        "duplicate_rows": summary.duplicate_rows,
        "target_class_count": summary.target_class_count,
        "symptom_column_count": summary.symptom_column_count,
    }
