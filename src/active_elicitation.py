"""Adaptive symptom elicitation helpers.

Provides a simple mutual-information based heuristic to suggest the next most
informative symptom to ask about given the currently selected symptoms.

The implementation is intentionally lightweight: it uses the preprocessed
training frame and sklearn.feature_selection.mutual_info_classif to score each
candidate symptom (not yet selected) and returns the top-ranked symptom code.
If mutual information cannot be computed for any reason, it falls back to a
frequency-based heuristic (most common symptom among training records).
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from src.preprocess import preprocess_training_data


def suggest_next_symptom(selected_symptoms: Iterable[str], top_n: int = 1) -> Optional[str]:
    """Suggest the next symptom code to ask about.

    Args:
        selected_symptoms: iterable of symptom column codes already selected.
        top_n: number of top suggestions to return (currently returns first when != 1).

    Returns:
        A symptom column name (string) or None if no suggestion is available.
    """
    selected = set(selected_symptoms or [])

    pre = preprocess_training_data()
    frame = pre.frame
    symptom_cols = [
        column
        for column in pre.symptom_columns
        if not column.endswith(("_severity", "_duration"))
    ]
    target_col = pre.frame.columns[-1] if pre.frame is not None else None

    # Candidate symptoms are those in symptom_cols not already selected
    candidates = [c for c in symptom_cols if c not in selected]
    if not candidates:
        return None

    try:
        # Compute mutual information for each candidate against the target
        from sklearn.feature_selection import mutual_info_classif

        X = frame[candidates].astype(float).fillna(0.0)
        y = frame[target_col]
        # mutual_info_classif requires 2D array
        mi = mutual_info_classif(X.values, y.values, discrete_features=True, random_state=0)
        # pick top n
        ranked_idx = np.argsort(mi)[::-1]
        top_idx = ranked_idx[:top_n]
        suggestions = [candidates[i] for i in top_idx if mi[i] > 0]
        if not suggestions:
            # fallback to frequency
            raise RuntimeError("mutual info produced no positive scores")
        return suggestions[0] if top_n == 1 else suggestions
    except Exception:
        # Fallback: choose the most frequent symptom (by column sum)
        try:
            freqs = frame[candidates].sum(axis=0)
            best = freqs.sort_values(ascending=False).index.tolist()
            return best[0] if best else None
        except Exception:
            return None
