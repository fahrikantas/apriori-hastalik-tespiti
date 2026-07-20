"""Plotting utilities for model analysis and Streamlit visualizations."""

from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt
import pandas as pd

from src.utils import TARGET_COLUMN

FIGURE_STYLE = "seaborn-v0_8-whitegrid"
FIGURE_WIDTH = 12
FIGURE_HEIGHT = 7


def _create_figure(title: str) -> tuple[plt.Figure, plt.Axes]:
    """Create a consistently styled Matplotlib figure."""

    plt.style.use(FIGURE_STYLE)
    figure, axis = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    axis.set_title(title, fontsize=16, fontweight="bold")
    return figure, axis


def plot_disease_distribution(frame: pd.DataFrame) -> plt.Figure:
    """Plot the frequency of each disease class in the training data."""

    figure, axis = _create_figure("Disease Distribution")
    counts = frame[TARGET_COLUMN].value_counts().sort_values(ascending=False)
    counts.plot(kind="bar", ax=axis, color="#1f77b4")
    axis.set_xlabel("Disease")
    axis.set_ylabel("Count")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    return figure


def plot_top_symptoms(frame: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    """Plot the most common symptoms across all training rows."""

    figure, axis = _create_figure("Most Frequent Symptoms")
    symptom_counts = frame.drop(columns=[TARGET_COLUMN]).sum().sort_values(ascending=False).head(top_n)
    symptom_counts.plot(kind="bar", ax=axis, color="#ff7f0e")
    axis.set_xlabel("Symptom")
    axis.set_ylabel("Frequency")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    return figure


def plot_naive_bayes_probabilities(probabilities: pd.DataFrame) -> plt.Figure:
    """Plot Naive Bayes class probabilities."""

    figure, axis = _create_figure("Naive Bayes Probability Distribution")
    probabilities.plot(kind="bar", x=TARGET_COLUMN, y="probability_pct", ax=axis, color="#2ca02c")
    axis.set_xlabel("Disease")
    axis.set_ylabel("Probability (%)")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    return figure


def plot_feature_importance(feature_importance: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    """Plot the most important features from the Random Forest model."""

    figure, axis = _create_figure("Random Forest Feature Importance")
    top_features = feature_importance.head(top_n).iloc[::-1]
    axis.barh(top_features["feature"], top_features["importance"], color="#9467bd")
    axis.set_xlabel("Importance")
    axis.set_ylabel("Feature")
    figure.tight_layout()
    return figure


def plot_model_accuracy_comparison(accuracies: Mapping[str, float]) -> plt.Figure:
    """Plot a simple bar chart comparing model accuracies."""

    figure, axis = _create_figure("Model Accuracy Comparison")
    model_names = list(accuracies.keys())
    model_scores = list(accuracies.values())
    axis.bar(model_names, model_scores, color="#d62728")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Accuracy")
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    return figure
