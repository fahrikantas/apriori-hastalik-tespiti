"""SHAP and LIME explainability helpers for model predictions."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import display_symptom_name, humanize_label


def _active_features(symptom_vector: pd.DataFrame) -> list[str]:
    row = symptom_vector.iloc[0]
    return [col for col in symptom_vector.columns if int(row[col]) == 1]


def compute_shap_values(
    model_bundle: dict[str, Any],
    symptom_vector: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Compute SHAP values for tree-based models and return top features."""

    import shap

    model = model_bundle["model"]
    feature_columns = model_bundle.get("feature_columns", list(symptom_vector.columns))
    background = pd.DataFrame(
        np.zeros((1, len(feature_columns)), dtype=int),
        columns=feature_columns,
    )
    explainer = shap.TreeExplainer(model, data=background, feature_perturbation="tree_path_dependent")
    shap_values = explainer.shap_values(symptom_vector[feature_columns])
    if isinstance(shap_values, list):
        prediction_index = int(model.predict(symptom_vector[feature_columns])[0])
        values = shap_values[prediction_index][0]
    else:
        values = shap_values[0]

    frame = pd.DataFrame(
        {
            "feature": feature_columns,
            "shap_value": values,
            "abs_shap": np.abs(values),
        }
    ).sort_values(by="abs_shap", ascending=False)
    return frame.head(top_n).reset_index(drop=True)


def plot_shap_bar(
    shap_frame: pd.DataFrame,
    language: str = "tr",
    figsize: tuple[int, int] = (8, 5),
) -> plt.Figure:
    """Plot horizontal bar chart of SHAP contributions."""

    labels = [display_symptom_name(feature, language) for feature in shap_frame["feature"]]
    fig, axis = plt.subplots(figsize=figsize)
    colors = ["#16a34a" if value >= 0 else "#dc2626" for value in shap_frame["shap_value"]]
    axis.barh(labels[::-1], shap_frame["shap_value"][::-1], color=colors[::-1])
    axis.set_xlabel("SHAP value")
    axis.set_title("SHAP — Belirti Katkıları" if language == "tr" else "SHAP — Symptom Contributions")
    fig.tight_layout()
    return fig


def compute_lime_explanation(
    model_bundle: dict[str, Any],
    symptom_vector: pd.DataFrame,
    label_encoder: Any,
    top_n: int = 8,
) -> pd.DataFrame:
    """Explain a single prediction with LIME."""

    try:
        from lime.lime_tabular import LimeTabularExplainer
    except Exception:  # pragma: no cover - environment dependent
        return pd.DataFrame(columns=["feature_label", "weight"])

    model = model_bundle["model"]
    feature_columns = model_bundle.get("feature_columns", list(symptom_vector.columns))
    training_matrix = np.zeros((2, len(feature_columns)), dtype=int)
    explainer = LimeTabularExplainer(
        training_matrix,
        feature_names=feature_columns,
        class_names=list(label_encoder.classes_),
        mode="classification",
        discretize_continuous=False,
    )

    def predict_proba_wrapper(matrix: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(matrix, columns=feature_columns)
        if hasattr(model, "predict_proba"):
            return model.predict_proba(frame)
        predictions = model.predict(frame)
        probabilities = np.zeros((len(predictions), len(label_encoder.classes_)))
        for index, prediction in enumerate(predictions):
            probabilities[index, int(prediction)] = 1.0
        return probabilities

    explanation = explainer.explain_instance(
        symptom_vector.iloc[0].values,
        predict_proba_wrapper,
        num_features=min(top_n, len(feature_columns)),
        top_labels=1,
    )
    label_index = explanation.top_labels[0]
    rows = []
    for feature, weight in explanation.as_list(label=label_index):
        rows.append({"feature_label": feature, "weight": weight})
    return pd.DataFrame(rows)


def plot_lime_bar(
    lime_frame: pd.DataFrame,
    predicted_label: str,
    language: str = "tr",
    figsize: tuple[int, int] = (8, 5),
) -> plt.Figure:
    """Plot LIME feature weights."""

    fig, axis = plt.subplots(figsize=figsize)
    colors = ["#2563eb" if value >= 0 else "#f97316" for value in lime_frame["weight"]]
    axis.barh(lime_frame["feature_label"][::-1], lime_frame["weight"][::-1], color=colors[::-1])
    title = (
        f"LIME — {humanize_label(predicted_label)}"
        if language == "tr"
        else f"LIME — {humanize_label(predicted_label)}"
    )
    axis.set_title(title)
    axis.set_xlabel("Weight")
    fig.tight_layout()
    return fig


def shap_supported(model_label: str) -> bool:
    """Return True when SHAP tree explainer is appropriate."""

    return model_label in {"Decision Tree", "Random Forest", "XGBoost", "LightGBM"}
