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


def clean_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean the training data so it can be consumed by all models."""

    cleaned_frame = standardize_columns(frame.copy())
    ensure_columns_exist(cleaned_frame, [TARGET_COLUMN])

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

    symptom_columns = identify_symptom_columns(frame)
    transactions: list[list[str]] = []
    for _, row in frame.iterrows():
        active_items = [column for column in symptom_columns if int(row[column]) == 1]
        active_items.append(f"disease_{row[TARGET_COLUMN]}")
        transactions.append(active_items)
    return transactions


def get_available_symptoms(frame: pd.DataFrame) -> list[str]:
    """Return the sorted symptom vocabulary expected by the UI."""

    symptom_columns = identify_symptom_columns(frame)
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
