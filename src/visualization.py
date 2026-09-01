"""Plotting utilities for model analysis and Streamlit visualizations."""

from __future__ import annotations

from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import plot_tree

from src.utils import TARGET_COLUMN

FIGURE_STYLE = "seaborn-v0_8-whitegrid"
FIGURE_WIDTH = 30
FIGURE_HEIGHT = 15


def _create_figure(title: str, figsize: tuple[float, float] | None = None) -> tuple[plt.Figure, plt.Axes]:
    """Create a consistently styled Matplotlib figure."""

    plt.style.use(FIGURE_STYLE)
    figure_size = figsize or (FIGURE_WIDTH, FIGURE_HEIGHT)
    figure, axis = plt.subplots(figsize=figure_size)
    axis.set_title(title, fontsize=16, fontweight="bold")
    return figure, axis


def plot_disease_distribution(frame: pd.DataFrame) -> plt.Figure:
    """Plot the frequency of each disease class in the training data."""

    figure, axis = _create_figure("Disease Distribution")
    counts = frame[TARGET_COLUMN].value_counts().sort_values(ascending=True)
    axis.barh(counts.index, counts.values, color="#1f77b4")
    axis.set_title("Disease Distribution", fontsize=22, fontweight="bold")
    axis.set_xlabel("Count", fontsize=22)
    axis.set_ylabel("Disease", fontsize=22)
    axis.tick_params(axis="x", labelsize=18)
    axis.tick_params(axis="y", labelsize=18)
    figure.tight_layout()
    return figure


def plot_top_symptoms(frame: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    """Plot the most common symptoms across all training rows."""

    figure, axis = _create_figure("Most Frequent Symptoms")
    symptom_counts = frame.drop(columns=[TARGET_COLUMN]).sum().sort_values(ascending=False).head(top_n)
    symptom_counts.plot(kind="bar", ax=axis, color="#ff7f0e")
    axis.set_title("Most Frequent Symptoms", fontsize=22, fontweight="bold")
    axis.set_xlabel("Symptom", fontsize=22)
    axis.set_ylabel("Frequency", fontsize=22)
    axis.tick_params(axis="x", rotation=45, labelsize=18)
    axis.tick_params(axis="y", labelsize=18)
    figure.tight_layout()
    return figure


def plot_naive_bayes_probabilities(
    probabilities: pd.DataFrame,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot Naive Bayes class probabilities."""

    figure, axis = _create_figure("Naive Bayes Probability Distribution", figsize=figsize)
    sorted_probabilities = probabilities.sort_values(by="probability_pct", ascending=True)
    axis.barh(sorted_probabilities[TARGET_COLUMN], sorted_probabilities["probability_pct"], color="#2ca02c")
    axis.set_title("Naive Bayes Probability Distribution", fontsize=22, fontweight="bold")
    axis.set_xlabel("Probability (%)", fontsize=22)
    axis.set_ylabel("Disease", fontsize=22)
    axis.tick_params(axis="x", labelsize=16)
    axis.tick_params(axis="y", labelsize=16)
    figure.tight_layout()
    return figure


def plot_feature_importance(
    feature_importance: pd.DataFrame,
    top_n: int = 15,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot the most important features from the Random Forest model."""

    figure, axis = _create_figure("Random Forest Feature Importance", figsize=figsize)
    top_features = feature_importance.head(top_n).iloc[::-1]
    axis.barh(top_features["feature"], top_features["importance"], color="#9467bd")
    axis.set_xlabel("Importance", fontsize=12)
    axis.set_ylabel("Feature", fontsize=12)
    axis.tick_params(axis="x", labelsize=9)
    axis.tick_params(axis="y", labelsize=9)
    figure.tight_layout()
    return figure


def plot_model_accuracy_comparison(
    accuracies: Mapping[str, float],
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot a simple bar chart comparing model accuracies."""

    figure, axis = _create_figure("Model Accuracy Comparison", figsize=figsize)
    model_names = list(accuracies.keys())
    model_scores = list(accuracies.values())
    axis.bar(model_names, model_scores, color="#d62728")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Accuracy", fontsize=12)
    axis.tick_params(axis="x", rotation=20, labelsize=9)
    axis.tick_params(axis="y", labelsize=9)
    figure.tight_layout()
    return figure


def plot_confusion_matrix(
    confusion_df: pd.DataFrame,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Render a labeled confusion matrix heatmap."""

    figure, axis = _create_figure("Confusion Matrix", figsize=figsize or (12, 10))
    axis.imshow(confusion_df.to_numpy(), interpolation="nearest", cmap="Blues")
    axis.set_title("Confusion Matrix", fontsize=18, fontweight="bold")
    axis.set_xlabel("Predicted", fontsize=12)
    axis.set_ylabel("Actual", fontsize=12)
    axis.set_xticks(range(len(confusion_df.columns)))
    axis.set_xticklabels(list(confusion_df.columns), rotation=90, fontsize=7)
    axis.set_yticks(range(len(confusion_df.index)))
    axis.set_yticklabels(list(confusion_df.index), fontsize=7)
    figure.tight_layout()
    return figure


def _strip_value_line(label: str) -> str:
    """Remove value, sample, gini, and pure numeric lines from a tree node label."""

    cleaned_lines = []
    for line in label.splitlines():
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("value") or lower.startswith("gini") or lower.startswith("samples"):
            continue
        if stripped and all(char in "0123456789.,[]- " for char in stripped):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def plot_decision_tree_model(
    model_bundle: Mapping[str, Any],
    figsize: tuple[float, float] | None = None,
    max_depth: int = 4,
    show_values: bool = False,
) -> plt.Figure:
    """Render a compact visual of the trained decision tree model."""

    figure, axis = _create_figure("Decision Tree Model", figsize=figsize)
    model = model_bundle.get("model")
    feature_columns = model_bundle.get("feature_columns", [])
    label_encoder = model_bundle.get("label_encoder")
    class_names = [str(label) for label in label_encoder.classes_] if label_encoder is not None else None

    annotations = plot_tree(
        model,
        feature_names=list(feature_columns),
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=max_depth,
        ax=axis,
    )

    if not show_values:
        for annotation in annotations:
            annotation.set_text(_strip_value_line(annotation.get_text()))

    axis.set_title("Decision Tree Model", fontsize=18, fontweight="bold")
    figure.tight_layout()
    return figure

