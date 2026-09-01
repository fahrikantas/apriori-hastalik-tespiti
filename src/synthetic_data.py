"""Synthetic, richer alternative training dataset.

The original dataset contains 4,920 rows; this generator produces a
deterministic, larger variant (5,000 rows) that reuses the same symptom
vocabulary and disease list but adds per-row noise and cross-disease
distractor symptoms, making the classification problem harder and more
realistic. The generated CSV is persisted once and then behaves exactly like
any other training file (fingerprint, model statuses, retraining).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocess import preprocess_training_data
from src.utils import DATA_DIR, DEFAULT_DELIMITER, TARGET_COLUMN, ensure_directory

SYNTHETIC_FILE_NAME = "synthetic_dataset.csv"
SYNTHETIC_ALIAS = "Synthetic.csv"
DEFAULT_N_ROWS = 5000
DEFAULT_RANDOM_STATE = 42


def generate_synthetic_frame(
    n_rows: int = DEFAULT_N_ROWS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Build a deterministic, richer synthetic dataset from the original."""

    base = preprocess_training_data("Training.csv")
    feature_columns = base.symptom_columns
    diseases = sorted(base.frame[TARGET_COLUMN].unique())

    profiles = base.frame.groupby(TARGET_COLUMN)[feature_columns].mean()
    rng = np.random.default_rng(random_state)

    rows: list[list[object]] = []
    per_class = max(2, n_rows // len(diseases))
    for disease in diseases:
        probabilities = np.clip(profiles.loc[disease].to_numpy(), 0.05, 0.95)
        for _ in range(per_class):
            row = (rng.random(len(feature_columns)) < probabilities).astype(int)
            distractors = np.where(row == 0)[0]
            n_distractors = int(rng.integers(0, 4))
            if n_distractors and len(distractors):
                chosen = rng.choice(
                    distractors,
                    size=min(n_distractors, len(distractors)),
                    replace=False,
                )
                row[chosen] = 1
            if int(row.sum()) < 2:
                forced = rng.choice(len(feature_columns), size=2, replace=False)
                row[forced] = 1
            rows.append([disease, *row.tolist()])

    return pd.DataFrame(rows, columns=[TARGET_COLUMN, *feature_columns])


def generate_and_persist_synthetic_dataset(
    n_rows: int = DEFAULT_N_ROWS,
    random_state: int = DEFAULT_RANDOM_STATE,
    overwrite: bool = False,
) -> Path:
    """Generate (once) the synthetic dataset file and return its path."""

    ensure_directory(DATA_DIR)
    target_path = DATA_DIR / SYNTHETIC_FILE_NAME
    if target_path.exists() and not overwrite:
        return target_path
    frame = generate_synthetic_frame(n_rows=n_rows, random_state=random_state)
    frame.to_csv(target_path, sep=DEFAULT_DELIMITER, index=False)
    return target_path