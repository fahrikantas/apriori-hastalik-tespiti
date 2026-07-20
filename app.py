"""Streamlit application for symptom-based disease prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import streamlit as st

from src.decision_tree import DECISION_TREE_MODEL_NAME
from src.naive_bayes import NAIVE_BAYES_MODEL_NAME
from src.predict import predict_from_symptoms
from src.preprocess import preprocess_training_data
from src.random_forest import RANDOM_FOREST_MODEL_NAME
from src.utils import humanize_label, normalize_symptom_name, resolve_model_path
from src.visualization import (
    plot_disease_distribution,
    plot_feature_importance,
    plot_model_accuracy_comparison,
    plot_naive_bayes_probabilities,
    plot_top_symptoms,
)

APP_TITLE = "Semptom Verilerine Dayalı Hastalık Tahmin ve Karar Destek Sistemi"
APP_ICON = "🩺"
MODEL_LABELS = {
    "decision_tree": "Decision Tree",
    "naive_bayes": "Naive Bayes",
    "random_forest": "Random Forest",
}


@st.cache_data(show_spinner=False)
def load_clean_training_data() -> tuple[Any, list[str]]:
    """Load and cache the cleaned training frame with symptom columns."""

    preprocessed = preprocess_training_data()
    return preprocessed.frame, preprocessed.symptom_columns


def load_model_bundle(model_name: str) -> dict[str, Any]:
    """Load a saved model bundle from the models directory."""

    model_path = resolve_model_path(model_name)
    return joblib.load(model_path)


def load_model_accuracies() -> dict[str, float]:
    """Collect the saved accuracies for the trained supervised models."""

    accuracies: dict[str, float] = {}
    for model_name, label in (
        (DECISION_TREE_MODEL_NAME, MODEL_LABELS["decision_tree"]),
        (NAIVE_BAYES_MODEL_NAME, MODEL_LABELS["naive_bayes"]),
        (RANDOM_FOREST_MODEL_NAME, MODEL_LABELS["random_forest"]),
    ):
        bundle = load_model_bundle(model_name)
        accuracies[label] = float(bundle.get("accuracy", 0.0))
    return accuracies


def build_sidebar(symptom_columns: list[str]) -> list[str]:
    """Render the symptom selection sidebar and return selected symptoms."""

    selected_symptoms: list[str] = []
    st.sidebar.markdown("### Semptom Seçimi")
    st.sidebar.caption("Sol panelden semptomları işaretleyin ve analizi çalıştırın.")

    for index, symptom in enumerate(symptom_columns):
        label = humanize_label(symptom)
        if st.sidebar.checkbox(label, key=f"symptom_{index}"):
            selected_symptoms.append(symptom)

    return selected_symptoms


def render_summary_cards(results: dict[str, str]) -> None:
    """Show the primary model outputs in a compact layout."""

    first_column, second_column, third_column = st.columns(3)
    with first_column:
        st.metric("Decision Tree", results["decision_tree"])
    with second_column:
        st.metric("Naive Bayes", results["naive_bayes"])
    with third_column:
        st.metric("Random Forest", results["random_forest"])


def render_final_result(predictions: list[str]) -> str:
    """Compute a simple consensus result from the three classifiers."""

    counts: dict[str, int] = {}
    for prediction in predictions:
        counts[prediction] = counts.get(prediction, 0) + 1
    majority_vote = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
    return majority_vote


def main() -> None:
    """Execute the Streamlit application."""

    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #e8eef7 0%, #f7f9fc 45%, #ffffff 100%);
            color: #0f172a;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp h4,
        .stApp h5,
        .stApp h6,
        .stApp p,
        .stApp span,
        .stApp label,
        .stApp li,
        .stApp div {
            color: #0f172a;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8fbff 0%, #edf4ff 100%);
        }
        [data-testid="stSidebar"] * {
            color: #0f172a !important;
        }
        .stButton button {
            background: linear-gradient(90deg, #1d4ed8 0%, #2563eb 100%);
            color: white !important;
            border: none;
            border-radius: 12px;
            padding: 0.6rem 1.1rem;
            font-weight: 700;
            box-shadow: 0 8px 18px rgba(37, 99, 235, 0.22);
        }
        .stButton button:hover {
            background: linear-gradient(90deg, #1e40af 0%, #1d4ed8 100%);
            color: white !important;
        }
        .stMetric {
            background: rgba(255, 255, 255, 0.96);
            border-radius: 16px;
            padding: 0.75rem 1rem;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.12);
            border: 1px solid rgba(148, 163, 184, 0.25);
        }
        .stDataFrame, .stTable {
            background: white;
            border-radius: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title(APP_TITLE)
    st.write(
        "Kullanıcı yalnızca semptomları seçer; sistem arka planda Apriori, Decision Tree, Naive Bayes ve Random Forest analizlerini birlikte çalıştırır."
    )

    training_frame, symptom_columns = load_clean_training_data()
    selected_symptoms = build_sidebar(symptom_columns)

    analyze_clicked = st.sidebar.button("Analiz Et", use_container_width=True)

    if not analyze_clicked:
        st.info("Analize başlamak için soldan semptom seçip Analiz Et butonuna basın.")
        st.subheader("Veri Görselleri")
        left_column, right_column = st.columns(2)
        with left_column:
            st.pyplot(plot_disease_distribution(training_frame), clear_figure=True)
            st.pyplot(plot_top_symptoms(training_frame), clear_figure=True)
        with right_column:
            accuracies = load_model_accuracies() if all(
                resolve_model_path(model_name).exists()
                for model_name in (
                    DECISION_TREE_MODEL_NAME,
                    NAIVE_BAYES_MODEL_NAME,
                    RANDOM_FOREST_MODEL_NAME,
                )
            ) else {}
            if accuracies:
                st.pyplot(plot_model_accuracy_comparison(accuracies), clear_figure=True)
        return

    if not selected_symptoms:
        st.warning("Lütfen en az bir semptom seçin.")
        return

    with st.spinner("Modeller çalıştırılıyor..."):
        prediction_bundle = predict_from_symptoms(selected_symptoms)

    decision_tree_bundle = load_model_bundle(DECISION_TREE_MODEL_NAME)
    naive_bayes_bundle = load_model_bundle(NAIVE_BAYES_MODEL_NAME)
    random_forest_bundle = load_model_bundle(RANDOM_FOREST_MODEL_NAME)
    accuracies = load_model_accuracies()

    st.subheader("Analiz Sonuçları")
    render_summary_cards(
        {
            "decision_tree": prediction_bundle.decision_tree_prediction,
            "naive_bayes": prediction_bundle.naive_bayes_prediction,
            "random_forest": prediction_bundle.random_forest_prediction,
        }
    )

    st.divider()

    result_column, probability_column = st.columns([1.2, 1])
    with result_column:
        st.markdown("### Apriori Sonuçları")
        if prediction_bundle.apriori_rules.empty:
            st.info("Seçilen semptomlar için güçlü bir Apriori kuralı bulunamadı.")
        else:
            st.dataframe(prediction_bundle.apriori_rules, use_container_width=True)

        st.markdown("### Decision Tree Sonucu")
        st.success(f"Tahmin edilen hastalık: {prediction_bundle.decision_tree_prediction}")

        st.markdown("### Random Forest Sonucu")
        st.success(f"Tahmin edilen hastalık: {prediction_bundle.random_forest_prediction}")

    with probability_column:
        st.markdown("### Naive Bayes Olasılıkları")
        st.dataframe(prediction_bundle.naive_bayes_probabilities, use_container_width=True)
        st.pyplot(
            plot_naive_bayes_probabilities(prediction_bundle.naive_bayes_probabilities),
            clear_figure=True,
        )

    st.subheader("Feature Importance")
    feature_importance = random_forest_bundle["feature_importance"]
    st.pyplot(plot_feature_importance(feature_importance), clear_figure=True)

    st.subheader("Model Accuracy Karşılaştırması")
    st.pyplot(plot_model_accuracy_comparison(accuracies), clear_figure=True)

    st.subheader("Nihai Sonuç")
    final_prediction = render_final_result(
        [
            prediction_bundle.decision_tree_prediction,
            prediction_bundle.naive_bayes_prediction,
            prediction_bundle.random_forest_prediction,
        ]
    )
    st.success(final_prediction)
    st.caption("Bu sistem yalnızca karar destek amaçlıdır. Kesin tanı yerine geçmez.")


if __name__ == "__main__":
    main()
