"""Streamlit application for symptom-based disease prediction."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.chatbot import (
    KnowledgeBase,
    build_chatbot_response,
    build_knowledge_base,
    stream_chat_reply,
)
from src.decision_tree import DECISION_TREE_MODEL_NAME
from src.disease_info import build_disease_card, list_all_diseases
from src.evaluation import (
    confusion_matrix_frame,
    cross_validation_summary,
    naive_bayes_calibration,
    per_class_metrics,
)
from src.explainability import (
    compute_lime_explanation,
    plot_lime_bar,
)
from src.i18n import LANGUAGES, translate
from src.icd10 import get_icd10_code, icd10_chapter
from src.lightgbm_model import LIGHTGBM_MODEL_NAME, train_lightgbm
from src.llm import llm_available
from src.logistic_regression import LOGISTIC_REGRESSION_MODEL_NAME
from src.model_metadata import (
    compute_fingerprint,
    model_statuses,
)
from src.naive_bayes import NAIVE_BAYES_MODEL_NAME
from src.predict import (
    AprioriParams,
    _get_cached_apriori_rules,
    _get_cached_models,
    _get_cached_preprocessed,
    predict_from_symptoms,
)
from src.preprocess import get_available_symptoms, preprocess_training_data
from src.random_forest import RANDOM_FOREST_MODEL_NAME
from src.red_flags import check_red_flags
from src.reports import build_pdf_report
from src.svm import SVM_MODEL_NAME
from src.telemetry import (
    clear_telemetry,
    log_prediction,
    model_agreement,
    summarize_disagreements,
)
from src.utils import (
    display_symptom_name,
    expand_search_terms,
    humanize_label,
    normalize_search_text,
    resolve_model_path,
)
from src.visualization import (
    plot_confusion_matrix,
    plot_decision_tree_model,
    plot_disease_distribution,
    plot_feature_importance,
    plot_model_accuracy_comparison,
    plot_top_symptoms,
)
from src.xgboost_model import XGBOOST_MODEL_NAME, train_xgboost

APP_TITLE = "Symptom-Based Disease Prediction and Decision Support System"
APP_ICON = "🩺"

SUPERVISED_MODEL_NAMES = [
    DECISION_TREE_MODEL_NAME,
    NAIVE_BAYES_MODEL_NAME,
    RANDOM_FOREST_MODEL_NAME,
    LOGISTIC_REGRESSION_MODEL_NAME,
    SVM_MODEL_NAME,
    XGBOOST_MODEL_NAME,
    LIGHTGBM_MODEL_NAME,
]

MODEL_TRAINERS = {
    "Decision Tree": (DECISION_TREE_MODEL_NAME, lambda: __import__("src.decision_tree", fromlist=["train_decision_tree"]).train_decision_tree()),
    "Naive Bayes": (NAIVE_BAYES_MODEL_NAME, lambda: __import__("src.naive_bayes", fromlist=["train_naive_bayes"]).train_naive_bayes()),
    "Random Forest": (RANDOM_FOREST_MODEL_NAME, lambda: __import__("src.random_forest", fromlist=["train_random_forest"]).train_random_forest()),
    "Logistic Regression": (LOGISTIC_REGRESSION_MODEL_NAME, lambda: __import__("src.logistic_regression", fromlist=["train_logistic_regression"]).train_logistic_regression()),
    "SVM": (SVM_MODEL_NAME, lambda: __import__("src.svm", fromlist=["train_svm"]).train_svm()),
    "XGBoost": (XGBOOST_MODEL_NAME, train_xgboost),
    "LightGBM": (LIGHTGBM_MODEL_NAME, train_lightgbm),
}

EVAL_MODEL_NAMES = list(MODEL_TRAINERS.keys())


@st.cache_data(show_spinner=False)
def load_clean_training_data(training_file: str = "Training.csv") -> tuple[Any, list[str]]:
    preprocessed = preprocess_training_data(training_file)
    return preprocessed.frame, get_available_symptoms(preprocessed.frame)


@st.cache_data(show_spinner=False)
def get_data_fingerprint(training_file: str = "Training.csv") -> dict[str, Any]:
    return compute_fingerprint(training_file)


@st.cache_data(show_spinner=False)
def load_knowledge_base(training_file: str = "Training.csv") -> KnowledgeBase:
    training_frame, symptom_columns = load_clean_training_data(training_file)
    return build_knowledge_base(training_frame, symptom_columns)


@st.cache_data(show_spinner=False)
def cached_cross_validation(training_file: str = "Training.csv") -> pd.DataFrame:
    return cross_validation_summary(training_file)


@st.cache_data(show_spinner=False)
def cached_per_class_metrics(model_label: str, training_file: str = "Training.csv") -> pd.DataFrame:
    return per_class_metrics(model_label, training_file)


@st.cache_data(show_spinner=False)
def cached_confusion_matrix(model_label: str, training_file: str = "Training.csv") -> pd.DataFrame:
    return confusion_matrix_frame(model_label, training_file)


@st.cache_data(show_spinner=False)
def cached_calibration(training_file: str = "Training.csv") -> dict[str, Any]:
    return naive_bayes_calibration(training_file)


@st.cache_resource(show_spinner=False)
def load_model_bundle(model_name: str) -> dict[str, Any]:
    model_path = resolve_model_path(model_name)
    return joblib.load(model_path)


@st.cache_data(show_spinner=False)
def load_model_accuracies(training_file: str = "Training.csv") -> dict[str, float]:
    from src.decision_tree import train_decision_tree
    from src.logistic_regression import train_logistic_regression
    from src.naive_bayes import train_naive_bayes
    from src.random_forest import train_random_forest
    from src.svm import train_svm

    trainers = {
        "Decision Tree": (DECISION_TREE_MODEL_NAME, train_decision_tree),
        "Naive Bayes": (NAIVE_BAYES_MODEL_NAME, train_naive_bayes),
        "Random Forest": (RANDOM_FOREST_MODEL_NAME, train_random_forest),
        "Logistic Regression": (LOGISTIC_REGRESSION_MODEL_NAME, train_logistic_regression),
        "SVM": (SVM_MODEL_NAME, train_svm),
        "XGBoost": (XGBOOST_MODEL_NAME, train_xgboost),
        "LightGBM": (LIGHTGBM_MODEL_NAME, train_lightgbm),
    }
    accuracies: dict[str, float] = {}
    current_fingerprint = compute_fingerprint(training_file).get("data_fingerprint")
    for label, (model_name, trainer) in trainers.items():
        model_path = resolve_model_path(model_name)
        if not model_path.exists():
            trainer(training_file)
        bundle = load_model_bundle(model_name)
        metadata = bundle.get("metadata") or {}
        if metadata.get("data_fingerprint") != current_fingerprint:
            trainer(training_file)
            bundle = load_model_bundle(model_name)
        # If a persisted bundle has a missing or zero accuracy (e.g. produced by
        # an earlier training run), retrain to compute a valid reported accuracy.
        try:
            stored_acc = float(bundle.get("accuracy", 0.0))
        except Exception:
            stored_acc = 0.0
        if stored_acc <= 0.0:
            trainer(training_file)
            bundle = load_model_bundle(model_name)
        accuracies[label] = float(bundle.get("accuracy", 0.0))
    return accuracies


def all_model_predictions(prediction_bundle: Any) -> list[str]:
    return [
        prediction_bundle.decision_tree_prediction,
        prediction_bundle.naive_bayes_prediction,
        prediction_bundle.random_forest_prediction,
        prediction_bundle.logistic_regression_prediction,
        prediction_bundle.svm_prediction,
        prediction_bundle.xgboost_prediction,
        prediction_bundle.lightgbm_prediction,
    ]


def render_final_result(predictions: list[str]) -> str:
    counts: dict[str, int] = {}
    for prediction in predictions:
        counts[prediction] = counts.get(prediction, 0) + 1
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def compute_confidence(probability_frame: Any, language: str = "tr") -> tuple[str, str]:
    if probability_frame is None or probability_frame.empty:
        unknown = "Bilinmiyor" if language == "tr" else "Unknown"
        return unknown, translate(language, "no_symptoms_selected")
    top = float(probability_frame.iloc[0]["probability_pct"])
    if top >= 60:
        level = translate(language, "high")
    elif top >= 30:
        level = translate(language, "medium")
    else:
        level = translate(language, "low")
    detail = (
        f"En yüksek olasılık %{top:.1f} ({display_symptom_name(probability_frame.iloc[0]['prognosis'], language)})"
        if language == "tr"
        else f"Highest probability {top:.1f}% ({display_symptom_name(probability_frame.iloc[0]['prognosis'], language)})"
    )
    return level, detail


def format_selected_symptoms(selected_symptoms: list[str], limit: int = 5, language: str = "tr") -> str:
    if not selected_symptoms:
        return translate(language, "no_symptoms_selected")
    formatted_symptoms = [display_symptom_name(symptom, language) for symptom in selected_symptoms]
    if len(formatted_symptoms) <= limit:
        return ", ".join(formatted_symptoms)
    remaining_count = len(formatted_symptoms) - limit
    if language == "tr":
        return f"{', '.join(formatted_symptoms[:limit])} ve {remaining_count} tane daha"
    return f"{', '.join(formatted_symptoms[:limit])} and {remaining_count} more"


def build_prediction_report(
    selected_symptoms: list[str],
    final_prediction: str,
    prediction_bundle: Any,
    model_accuracies: dict[str, float],
) -> str:
    lines = [
        "SEMPTOMA DAYALI HASTALIK TAHMİN RAPORU",
        "=" * 40,
        f"Seçilen belirtiler: {format_selected_symptoms(selected_symptoms, limit=20)}",
        "",
        "MODEL TAHMİNLERİ",
        f"Decision Tree       : {prediction_bundle.decision_tree_prediction}",
        f"Naive Bayes         : {prediction_bundle.naive_bayes_prediction}",
        f"Random Forest       : {prediction_bundle.random_forest_prediction}",
        f"Logistic Regression : {prediction_bundle.logistic_regression_prediction}",
        f"SVM                 : {prediction_bundle.svm_prediction}",
        f"XGBoost             : {prediction_bundle.xgboost_prediction}",
        f"LightGBM            : {prediction_bundle.lightgbm_prediction}",
        f"Birleşik sonuç      : {final_prediction}",
        f"ICD-10 kodu         : {get_icd10_code(final_prediction) or 'bulunamadı'}",
        "",
        "AYIRICI TANI LİSTESİ (tüm modeller)",
    ]
    differential = getattr(prediction_bundle, "differential_diagnosis", None) or []
    for entry in differential:
        code = get_icd10_code(entry["disease"]) or "-"
        models = ", ".join(entry.get("supporting_models", [])) or "-"
        lines.append(f"* {entry['disease']} (ICD-10: {code}) — skor %{entry['score_pct']} — destek: {models}")
    lines.extend(
        [
            "",
            "MODEL DOĞRULUKLARI (doğrulama verisi)",
        ]
    )
    lines.extend(f"{model}: %{round(score * 100, 1)}" for model, score in model_accuracies.items())
    if not prediction_bundle.naive_bayes_probabilities.empty:
        lines.append("")
        lines.append("NAIVE BAYES OLASILIK SIRALAMASI (ilk 5)")
        for _, row in prediction_bundle.naive_bayes_probabilities.head(5).iterrows():
            lines.append(f"{row['prognosis']}: %{row['probability_pct']:.2f}")
    if not prediction_bundle.apriori_rules.empty:
        lines.append("")
        lines.append("EN GÜÇLÜ APRIORI KURALI")
        top_rule = prediction_bundle.apriori_rules.iloc[0]
        lines.append(f"{top_rule['antecedents']} -> {top_rule['consequent']}")
        lines.append(f"Güven: %{top_rule['confidence_pct']}, Destek: %{top_rule['support_pct']}, Lift: {top_rule['lift']}")
    lines.append("")
    lines.append("Bu rapor karar destek amaçlıdır; tıbbi tanı koymaz.")
    return "\n".join(lines)


def build_prediction_report_html(
    selected_symptoms: list[str],
    final_prediction: str,
    prediction_bundle: Any,
    model_accuracies: dict[str, float],
    language: str = "tr",
) -> str:
    confidence_level, confidence_detail = compute_confidence(prediction_bundle.naive_bayes_probabilities, language)
    rows = "\n".join(
        f"<tr><td><b>{model}</b></td><td>%{round(score * 100, 1)}</td></tr>"
        for model, score in model_accuracies.items()
    )
    probability_rows = "\n".join(
        f"<tr><td>{row['prognosis']}</td><td>%{row['probability_pct']:.2f}</td></tr>"
        for _, row in prediction_bundle.naive_bayes_probabilities.head(5).iterrows()
    )
    differential_rows = "\n".join(
        (
            f"<tr><td><b>{entry['disease']}</b></td>"
            f"<td>{get_icd10_code(entry['disease']) or '-'}</td>"
            f"<td>%{entry['score_pct']}</td>"
            f"<td>{', '.join(entry.get('supporting_models', [])) or '-'}</td></tr>"
        )
        for entry in (getattr(prediction_bundle, "differential_diagnosis", None) or [])
    )
    rule = ""
    if not prediction_bundle.apriori_rules.empty:
        top_rule = prediction_bundle.apriori_rules.iloc[0]
        rule = (
            f"<p><b>En güçlü Apriori kuralı:</b> "
            f"{top_rule['antecedents']} → {top_rule['consequent']} "
            f"(güven %{top_rule['confidence_pct']}, destek %{top_rule['support_pct']}, lift {top_rule['lift']})</p>"
        )
    differential_html = ""
    if differential_rows:
        differential_html = (
            f"<h3>{translate(language, 'differential_title')}</h3>"
            f"<table><tr><th>{translate(language, 'differential_disease')}</th>"
            f"<th>ICD-10</th><th>{translate(language, 'differential_score')}</th>"
            f"<th>{translate(language, 'differential_support')}</th></tr>"
            f"{differential_rows}</table>"
        )
    return f"""<!DOCTYPE html>
<html lang="{language}">
<head>
<meta charset="utf-8">
<title>Hastalık Tahmin Raporu</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; color: #0f172a; }}
h1 {{ color: #1d4ed8; }}
table {{ border-collapse: collapse; width: 60%; margin: 12px 0; }}
th, td {{ border: 1px solid #cbd5e1; padding: 8px 10px; text-align: left; }}
th {{ background: #eef2ff; }}
.symptoms {{ color: #475569; }}
.footer {{ margin-top: 28px; font-size: 13px; color: #64748b; }}
</style>
</head>
<body>
<h1>{translate(language, 'app_title')}</h1>
<p class="symptoms"><b>{format_selected_symptoms(selected_symptoms, limit=20, language=language)}</b></p>
<p><b>{translate(language, 'final_result')}:</b> {final_prediction}
<span style="background:#eef2ff;border:1px solid #c7d2fe;border-radius:6px;padding:2px 8px;font-family:monospace;">
{translate(language, 'icd10_label')} {get_icd10_code(final_prediction) or translate(language, 'icd10_missing')}</span></p>
<p><b>{translate(language, 'confidence_level')}</b> {confidence_level} — {confidence_detail}</p>
{rule}
{differential_html}
<h3>{translate(language, 'model_predictions')}</h3>
<table>
<tr><th>Model</th><th>Sonuç</th></tr>
<tr><td>Decision Tree</td><td>{prediction_bundle.decision_tree_prediction}</td></tr>
<tr><td>Naive Bayes</td><td>{prediction_bundle.naive_bayes_prediction}</td></tr>
<tr><td>Random Forest</td><td>{prediction_bundle.random_forest_prediction}</td></tr>
<tr><td>Logistic Regression</td><td>{prediction_bundle.logistic_regression_prediction}</td></tr>
<tr><td>SVM</td><td>{prediction_bundle.svm_prediction}</td></tr>
<tr><td>XGBoost</td><td>{prediction_bundle.xgboost_prediction}</td></tr>
<tr><td>LightGBM</td><td>{prediction_bundle.lightgbm_prediction}</td></tr>
</table>
<h3>{translate(language, 'model_accuracy')}</h3>
<table>{rows}</table>
<h3>{translate(language, 'nb_probabilities')}</h3>
<table>{probability_rows}</table>
<p class="footer">{datetime.now().strftime('%d.%m.%Y %H:%M')}<br>{translate(language, 'decision_support_only')}</p>
</body>
</html>"""


def build_chat_export(language: str) -> str:
    messages = st.session_state.get("chat_messages", [])
    title = translate(language, "chat_export")
    lines = [title, "=" * 40]
    for message in messages:
        role = "Asistan" if message["role"] == "assistant" else "Kullanıcı"
        if language == "en":
            role = "Assistant" if message["role"] == "assistant" else "User"
        lines.append(f"[{role}] {message['content']}")
    lines.append("")
    lines.append(translate(language, "decision_support_only"))
    return "\n\n".join(lines)


def build_sidebar_apriori_params(language: str) -> AprioriParams:
    t = lambda key: translate(language, key)
    with st.sidebar.expander(t("apriori_settings"), expanded=False):
        min_support = st.slider(t("apriori_min_support"), 0.005, 0.10, 0.01, 0.005, key="apriori_min_support")
        min_confidence = st.slider(t("apriori_min_confidence"), 0.30, 0.95, 0.50, 0.05, key="apriori_min_confidence")
        min_lift = st.slider(t("apriori_min_lift"), 0.5, 3.0, 1.0, 0.1, key="apriori_min_lift")
        max_len = st.slider(t("apriori_max_len"), 2, 8, 5, 1, key="apriori_max_len")
    return AprioriParams(
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
        max_len=max_len,
    )


def build_sidebar(symptom_columns: list[str], language: str) -> tuple[list[str], bool, dict, dict]:
    t = lambda key: translate(language, key)
    # Present symptom labels in Turkish (stable labels for tests and clinical users)
    label_to_code = {display_symptom_name(symptom, 'tr'): symptom for symptom in symptom_columns}
    st.sidebar.markdown(f"### {t('symptom_selection')}")
    st.sidebar.caption(t("sidebar_search_hint"))

    if st.sidebar.button(t("reset"), width="stretch", type="primary"):
        st.session_state["symptom_search"] = ""
        st.session_state["selected_symptoms_display"] = []

    search_term = st.sidebar.text_input(
        t("symptom_search"),
        key="symptom_search",
        placeholder=t("symptom_search_placeholder"),
        label_visibility="collapsed",
    )
    search_terms = expand_search_terms(search_term) if search_term else {""}
    available = []
    for label, symptom in label_to_code.items():
        symptom_text = normalize_search_text(label)
        if search_term and not any(term in symptom_text or term in normalize_search_text(symptom) for term in search_terms):
            continue
        available.append(label)

    if not available:
        st.sidebar.warning(t("no_match"))
        return [], st.sidebar.button(t("analyze"), width="stretch", type="primary"), {}, {}

    selected_labels = st.sidebar.multiselect(
        "",
        options=available,
        key="selected_symptoms_display",
        placeholder=t("symptoms_select"),
        label_visibility="collapsed",
    )
    selected_symptoms = [label_to_code[label] for label in selected_labels if label in label_to_code]

    # Per-symptom severity and duration inputs
    severity_map: dict = {}
    duration_map: dict = {}
    if selected_symptoms:
        with st.sidebar.expander(t('symptom_details'), expanded=False):
            st.caption(t('symptom_details_caption'))
            for symptom in selected_symptoms:
                key_safe = symptom.replace(" ", "_")
                sev_key = f"sev_{key_safe}"
                dur_key = f"dur_{key_safe}"
                # restore state if exists
                default_sev = int(st.session_state.get(sev_key, 1))
                default_dur = int(st.session_state.get(dur_key, 1))
                severity = st.slider(f"{display_symptom_name(symptom, language)} — {t('severity')}", 0, 3, default_sev, key=sev_key)
                duration = st.number_input(f"{display_symptom_name(symptom, language)} — {t('duration_days')}", min_value=0, value=default_dur, step=1, key=dur_key)
                severity_map[symptom] = int(severity)
                duration_map[symptom] = int(duration)

    analyze_clicked = st.sidebar.button(t("analyze"), width="stretch", type="primary")

    return selected_symptoms, analyze_clicked, severity_map, duration_map


def initialize_chatbot_state(language: str, reset: bool = False) -> None:
    if reset or "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [
            {"role": "assistant", "content": translate(language, "chat_welcome"), "streamed": False}
        ]
    if reset or "chat_context" not in st.session_state:
        st.session_state["chat_context"] = {}


def submit_chat_message(
    message_text: str,
    selected_symptoms: list[str],
    final_prediction: str | None,
    prediction_bundle: Any | None,
    knowledge_base: KnowledgeBase | None,
    model_accuracies: dict[str, float] | None,
    language: str,
    llm_settings: dict[str, Any],
    training_file: str = "Training.csv",
) -> None:
    message_text = str(message_text).strip()
    if not message_text:
        return
    if knowledge_base is None:
        knowledge_base = load_knowledge_base(training_file)

    st.session_state["chat_messages"].append({"role": "user", "content": message_text})
    chat_context = dict(st.session_state.get("chat_context", {}))
    assistant_reply, updated_context = build_chatbot_response(
        message_text,
        selected_symptoms=selected_symptoms,
        final_prediction=final_prediction,
        prediction_bundle=prediction_bundle,
        context=chat_context,
        kb=knowledge_base,
        model_accuracies=model_accuracies,
        language=language,
        llm_settings=llm_settings,
    )
    st.session_state["chat_context"] = updated_context
    st.session_state["chat_messages"].append({"role": "assistant", "content": assistant_reply, "streamed": False})
    st.session_state["chat_scroll_requested"] = True


def handle_quick_select(language: str, llm_settings: dict[str, Any]) -> None:
    selection = st.session_state.get("chat_quick_pills")
    if not selection:
        return
    call_context = st.session_state.get("chat_call_context", {})
    submit_chat_message(
        str(selection),
        call_context.get("selected_symptoms", []),
        call_context.get("final_prediction"),
        call_context.get("prediction_bundle"),
        st.session_state.get("chat_kb"),
        call_context.get("model_accuracies"),
        language,
        llm_settings,
        call_context.get("training_file", "Training.csv"),
    )
    st.session_state["chat_quick_pills"] = None


def resolve_chat_context(
    selected_symptoms: list[str],
    final_prediction: str | None,
    prediction_bundle: Any | None,
    apriori_params: AprioriParams,
    severity_map: dict | None = None,
    duration_map: dict | None = None,
    training_file: str = "Training.csv",
) -> tuple[str | None, Any | None]:
    if selected_symptoms and (final_prediction is None or prediction_bundle is None):
        resolved_bundle = predict_from_symptoms(
            selected_symptoms,
            training_file,
            severity_map=severity_map or {},
            duration_map=duration_map or {},
            apriori_params=apriori_params,
        )
        resolved_prediction = getattr(resolved_bundle, "ensemble_prediction", render_final_result(all_model_predictions(resolved_bundle)))
        return resolved_prediction, resolved_bundle
    return final_prediction, prediction_bundle


def render_chatbot_section(
    selected_symptoms: list[str],
    final_prediction: str | None,
    prediction_bundle: Any | None,
    model_accuracies: dict[str, float] | None,
    language: str,
    llm_settings: dict[str, Any],
    apriori_params: AprioriParams,
    severity_map: dict | None = None,
    duration_map: dict | None = None,
    training_file: str = "Training.csv",
) -> None:
    t = lambda key: translate(language, key)
    initialize_chatbot_state(language)
    final_prediction, prediction_bundle = resolve_chat_context(
        selected_symptoms,
        final_prediction,
        prediction_bundle,
        apriori_params,
        severity_map=severity_map or {},
        duration_map=duration_map or {},
        training_file=training_file,
    )
    knowledge_base = load_knowledge_base(training_file)
    st.session_state["chat_kb"] = knowledge_base
    st.session_state["chat_call_context"] = {
        "selected_symptoms": selected_symptoms,
        "final_prediction": final_prediction,
        "prediction_bundle": prediction_bundle,
        "model_accuracies": model_accuracies,
        "severity_map": severity_map or {},
        "duration_map": duration_map or {},
        "training_file": training_file,
    }

    st.subheader(t("chat_title"))
    st.caption(t("chat_caption"))
    if llm_available(llm_settings):
        st.info(t("llm_active"))

    context_bits: list[str] = []
    if selected_symptoms:
        context_bits.append(f"🩹 **{t('chat_selected')}:** {format_selected_symptoms(selected_symptoms, language=language)}")
    if final_prediction:
        context_bits.append(f"🎯 **{t('chat_result')}:** {humanize_label(final_prediction)}")
    if context_bits:
        st.markdown("&nbsp;&nbsp;|&nbsp;&nbsp;".join(context_bits))

    if st.button(t("chat_clear")):
        initialize_chatbot_state(language, reset=True)

    quick_actions = [
        t("chat_quick_explain"),
        t("chat_quick_apriori"),
        t("chat_quick_next"),
        t("chat_quick_eval"),
        t("chat_quick_accuracy"),
        t("chat_quick_how"),
    ]
    if st.session_state.get("chat_quick_pills") not in quick_actions:
        st.session_state["chat_quick_pills"] = None
    st.pills(
        "quick",
        quick_actions,
        selection_mode="single",
        key="chat_quick_pills",
        label_visibility="collapsed",
        on_change=lambda: handle_quick_select(language, llm_settings),
    )

    prompt = st.chat_input(t("chat_input_placeholder"), key="chat_prompt_input")
    if prompt:
        submit_chat_message(
            str(prompt),
            selected_symptoms,
            final_prediction,
            prediction_bundle,
            knowledge_base,
            model_accuracies,
            language,
            llm_settings,
        )

    messages = st.session_state["chat_messages"]
    last_message_index = len(messages) - 1
    for index, message in enumerate(messages):
        if message["role"] == "user":
            with st.chat_message("user", avatar="🧑"):
                st.markdown(message["content"])
            continue
        with st.chat_message("assistant", avatar="👨‍⚕️"):
            if index == last_message_index and not message.get("streamed"):
                message["content"] = st.write_stream(stream_chat_reply(message["content"]))
                message["streamed"] = True
            else:
                st.markdown(message["content"])

    if st.session_state.pop("chat_scroll_requested", False):
        components.html(
            """
            <script>
                window.setTimeout(() => {
                    const chatMessages = window.parent.document.querySelectorAll('[data-testid="stChatMessage"]');
                    const chatAnchor = chatMessages[chatMessages.length - 1];
                    if (chatAnchor) {
                        chatAnchor.scrollIntoView({ behavior: "smooth", block: "end" });
                    }
                }, 50);
            </script>
            """,
            height=0,
            scrolling=False,
        )

    st.markdown(f"**{t('chat_feedback_question')}**")
    chat_action_cols = st.columns([1, 1, 1.4])
    with chat_action_cols[0]:
        if st.button(t("chat_feedback_positive"), key="feedback_positive", width="stretch"):
            st.session_state["chat_feedback"] = "positive"
    with chat_action_cols[1]:
        if st.button(t("chat_feedback_negative"), key="feedback_negative", width="stretch"):
            st.session_state["chat_feedback"] = "negative"
    with chat_action_cols[2]:
        st.download_button(
            t("chat_download"),
            data=build_chat_export(language),
            file_name="sohbet_dokumu.txt",
            mime="text/plain",
            type="primary",
            width="stretch",
        )
    if st.session_state.get("chat_feedback") == "positive":
        st.success(t("chat_feedback_thanks"))
    elif st.session_state.get("chat_feedback") == "negative":
        st.info(t("chat_feedback_note"))


def render_disease_info_tab(
    training_frame: Any,
    symptom_columns: list[str],
    final_prediction: str | None,
    language: str,
) -> None:
    t = lambda key: translate(language, key)
    st.subheader(t("disease_info_title"))
    st.caption(t("disease_info_caption"))
    diseases = list_all_diseases(training_frame)
    default_index = diseases.index(final_prediction) if final_prediction in diseases else 0
    selected_disease = st.selectbox(t("disease_select"), diseases, index=default_index)
    card = build_disease_card(selected_disease, training_frame, symptom_columns, language)
    st.markdown(f"### {card['name']}")
    st.write(card["summary"])
    if card["top_symptoms"]:
        st.markdown(f"**{t('disease_top_symptoms')}:**")
        st.markdown(", ".join(f"• {symptom}" for symptom in card["top_symptoms"]))
    st.info(f"**{t('disease_when_doctor')}** {card['when_to_see_doctor']}")


def render_explainability_tab(
    prediction_bundle: Any,
    language: str,
) -> None:
    t = lambda key: translate(language, key)
    st.subheader(t("explainability_title"))
    st.caption(t("explainability_caption"))
    selected_model = st.selectbox(t("explain_model_select"), EVAL_MODEL_NAMES)
    model_name, _ = MODEL_TRAINERS[selected_model]
    bundle = load_model_bundle(model_name)
    predicted = {
        "Decision Tree": prediction_bundle.decision_tree_prediction,
        "Naive Bayes": prediction_bundle.naive_bayes_prediction,
        "Random Forest": prediction_bundle.random_forest_prediction,
        "Logistic Regression": prediction_bundle.logistic_regression_prediction,
        "SVM": prediction_bundle.svm_prediction,
        "XGBoost": prediction_bundle.xgboost_prediction,
        "LightGBM": prediction_bundle.lightgbm_prediction,
    }[selected_model]

    st.markdown(f"### {t('lime_chart')} — {selected_model}")
    lime_frame = compute_lime_explanation(bundle, prediction_bundle.symptom_vector, bundle["label_encoder"])
    if lime_frame.empty:
        st.warning(
            "LIME explanation is unavailable because the optional 'lime' package is not installed. "
            "Install it with 'pip install lime' or 'pip install -r requirements.txt'."
        )
        return
    st.pyplot(plot_lime_bar(lime_frame, predicted, language), clear_figure=True)
    st.dataframe(lime_frame, width="stretch")


def render_prediction_tab(
    selected_symptoms: list[str],
    final_prediction: str,
    prediction_bundle: Any,
    model_accuracies: dict[str, float],
    language: str,
) -> None:
    t = lambda key: translate(language, key)

    icd10_code = get_icd10_code(final_prediction)
    chapter = icd10_chapter(icd10_code) if icd10_code else ""
    icd10_html = (
        f'<span class="icd10-badge"><b>{t("icd10_label")}</b> {icd10_code}</span>'
        f'<div class="icd10-chapter">{chapter}</div>'
        if icd10_code
        else f'<span class="icd10-badge">{t("icd10_missing")}</span>'
    )
    st.markdown(
        f"""
        <div class="result-hero">
            <div>
                <div class="result-label">{t('result_card_label')}</div>
                <div class="result-disease">{humanize_label(final_prediction)}</div>
                <div class="result-card-note" style="font-size:0.82rem;color:var(--text-mute);margin-top:0.2rem;">
                    {t('result_card_note')}
                </div>
            </div>
            <div>{icd10_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    confidence_level, confidence_detail = compute_confidence(prediction_bundle.naive_bayes_probabilities, language)
    top_pct = float(prediction_bundle.naive_bayes_probabilities.iloc[0]["probability_pct"])
    level_symbol = "🟢" if confidence_level == t("high") else ("🟡" if confidence_level == t("medium") else "🔴")
    fill_pct = max(0, min(100, top_pct))
    st.markdown(
        f"""
        <div class="confidence-wrap">
            <div class="confidence-header">
                <span class="confidence-label">{t('confidence_bar_label')} {level_symbol}</span>
                <span class="confidence-value">{confidence_level} — %{top_pct:.1f}</span>
            </div>
            <div class="confidence-track"><div class="confidence-fill" style="width:{fill_pct}%;"></div></div>
            <div class="confidence-detail">{confidence_detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    all_predictions = all_model_predictions(prediction_bundle)
    agreement_pct = 0
    if all_predictions:
        top_prediction = render_final_result(all_predictions)
        agreement_pct = int(100.0 * sum(1 for p in all_predictions if p == top_prediction) / len(all_predictions))
    rules = getattr(prediction_bundle, "apriori_rules", None)
    n_rules = 0 if rules is None else len(rules)
    n_diseases = len(prediction_bundle.naive_bayes_probabilities)
    st.markdown(
        f"""
        <div class="stat-strip">
            <div class="stat-card"><div class="stat-value">{len(selected_symptoms)}</div><div class="stat-label">{t('stat_symptoms')}</div></div>
            <div class="stat-card"><div class="stat-value">{n_rules}</div><div class="stat-label">{t('stat_rules')}</div></div>
            <div class="stat-card"><div class="stat-value">{len(SUPERVISED_MODEL_NAMES)}</div><div class="stat-label">{t('stat_models')}</div></div>
            <div class="stat-card"><div class="stat-value">{n_diseases}</div><div class="stat-label">{t('stat_diseases')}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ood = getattr(prediction_bundle, "ood", None)
    if ood and ood.get("is_ood"):
        st.warning(t("ood_warning").format(round(ood.get("overlap_fraction", 0) * 100)))
    elif ood:
        nearest_pct = round(ood.get("overlap_fraction", 0) * 100)
        nearest_disease = humanize_label(ood.get("nearest_disease", ""))
        st.caption(f"ℹ️ {t('ood_ok').format(nearest_pct)} — {nearest_disease}")

    red_flags = check_red_flags(selected_symptoms)
    if red_flags:
        st.error(t("red_flag_title"))
        for flag in red_flags:
            advice = flag.get("advice_tr") if language == "tr" else flag.get("advice_en")
            severity_symbol = "🚨" if flag.get("severity") == "critical" else "⚠️"
            matched = ", ".join(humanize_label(item) for item in flag["matched_symptoms"])
            st.markdown(f"{severity_symbol} **{matched}:** {advice}")
        st.caption(t("red_flag_caption"))
    else:
        st.caption(f"✅ {t('red_flag_none')}")

    if len(set(all_predictions)) >= 4:
        st.warning(t("ambiguous_warning"))
    elif confidence_level == t("low"):
        st.warning(t("low_confidence_warning"))

    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">{t('top_conditions_title')}</div>
            <div class="card-caption" style="font-size:0.82rem;color:var(--text-mute);margin-bottom:0.4rem;">{t('top_conditions_caption')}</div>
        """,
        unsafe_allow_html=True,
    )
    medals = ["🥇", "🥈", "🥉"]
    top_conditions = list(prediction_bundle.naive_bayes_probabilities.head(3).iterrows())
    for index, (_, row) in enumerate(top_conditions):
        medal_class = {0: "g1", 1: "g2", 2: "g3"}[index]
        st.markdown(
            f"""
            <div class="condition-row">
                <div class="condition-medal {medal_class}">{medals[index]}</div>
                <div class="condition-name">{display_symptom_name(row['prognosis'], language)}</div>
                <div class="condition-pct">%{float(row['probability_pct']):.1f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">{t('model_predictions')}</div>
        """,
        unsafe_allow_html=True,
    )
    model_predictions = [
        ("Decision Tree", prediction_bundle.decision_tree_prediction),
        ("Naive Bayes", prediction_bundle.naive_bayes_prediction),
        ("Random Forest", prediction_bundle.random_forest_prediction),
        ("Log. Reg.", prediction_bundle.logistic_regression_prediction),
        ("SVM", prediction_bundle.svm_prediction),
        ("XGBoost", prediction_bundle.xgboost_prediction),
        ("LightGBM", prediction_bundle.lightgbm_prediction),
    ]
    model_columns = st.columns(4)
    for index, (model_label, prediction) in enumerate(model_predictions):
        agree = prediction == final_prediction
        with model_columns[index % 4]:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-model">{model_label}</div>
                    <div class="metric-value {'agree' if agree else ''}">{display_symptom_name(prediction, language)}</div>
                    <div style="font-size:0.72rem;color:var(--text-mute);">{'✓ ' + t('agreement_label') if agree else ''}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader(t("differential_title"))
    st.caption(t("differential_caption"))
    differential = getattr(prediction_bundle, "differential_diagnosis", None) or []
    if differential:
        differential_rows = [
            {
                t("differential_disease"): entry["disease"],
                "ICD-10": get_icd10_code(entry["disease"]) or "-",
                t("differential_score"): round(entry["score_pct"], 2),
                t("differential_support_count"): entry["support_count"],
                t("differential_support"): ", ".join(entry.get("supporting_models", [])) or "-",
            }
            for entry in differential
        ]
        st.dataframe(differential_rows, width="stretch")
        st.caption(t("differential_note"))

    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">{t('downloads_title')}</div>
            <div class="card-caption" style="font-size:0.82rem;color:var(--text-mute);margin-bottom:0.6rem;">{t('downloads_caption')}</div>
        """,
        unsafe_allow_html=True,
    )
    report_text = build_prediction_report(selected_symptoms, final_prediction, prediction_bundle, model_accuracies)
    report_html = build_prediction_report_html(selected_symptoms, final_prediction, prediction_bundle, model_accuracies, language)
    try:
        report_pdf = build_pdf_report(
            selected_symptoms,
            final_prediction,
            prediction_bundle,
            model_accuracies,
            confidence_level,
            confidence_detail,
            language,
        )
    except Exception as exc:
        st.error(f"PDF generation skipped: {exc}")
        report_pdf = b""
    download_cols = st.columns(3)
    with download_cols[0]:
        st.download_button(t("download_txt"), data=report_text, file_name="hastalik_tahmin_raporu.txt", mime="text/plain", type="primary")
    with download_cols[1]:
        st.download_button(t("download_html"), data=report_html, file_name="hastalik_tahmin_raporu.html", mime="text/html", type="primary")
    with download_cols[2]:
        st.download_button(t("download_pdf"), data=report_pdf, file_name="hastalik_tahmin_raporu.pdf", mime="application/pdf", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)


def render_visualization_tab(
    training_frame: Any,
    prediction_bundle: Any,
    language: str,
) -> None:
    t = lambda key: translate(language, key)
    decision_tree_bundle = load_model_bundle(DECISION_TREE_MODEL_NAME)
    random_forest_bundle = load_model_bundle(RANDOM_FOREST_MODEL_NAME)

    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown(
            f'<div class="chart-card"><div class="chart-card-title">{t("apriori_results")}</div>',
            unsafe_allow_html=True,
        )
        if prediction_bundle.apriori_rules.empty:
            st.info(t("no_apriori_rule"))
        else:
            st.dataframe(prediction_bundle.apriori_rules, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="chart-card"><div class="chart-card-title">{t("dt_visualization")}</div>',
            unsafe_allow_html=True,
        )
        st.pyplot(plot_decision_tree_model(decision_tree_bundle, figsize=(14, 8)), clear_figure=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right_column:
        st.markdown(
            f'<div class="chart-card"><div class="chart-card-title">{t("nb_probabilities")}</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(prediction_bundle.naive_bayes_probabilities, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="chart-card"><div class="chart-card-title">{t("feature_importance")}</div>',
            unsafe_allow_html=True,
        )
        st.pyplot(
            plot_feature_importance(random_forest_bundle["feature_importance"], figsize=(8, 4)),
            clear_figure=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.subheader(t("data_visualizations"))
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown(
            '<div class="chart-card" style="margin-bottom:0;">',
            unsafe_allow_html=True,
        )
        st.pyplot(plot_disease_distribution(training_frame), clear_figure=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with chart_right:
        st.markdown(
            '<div class="chart-card" style="margin-bottom:0;">',
            unsafe_allow_html=True,
        )
        st.pyplot(plot_top_symptoms(training_frame), clear_figure=True)
        st.markdown("</div>", unsafe_allow_html=True)


def render_evaluation_tab(model_accuracies: dict[str, float], model_status_state: dict[str, dict[str, Any]], language: str, training_file: str = "Training.csv") -> None:
    t = lambda key: translate(language, key)
    st.markdown(
        f'<div class="chart-card"><div class="chart-card-title">{t("model_accuracy")}</div>',
        unsafe_allow_html=True,
    )
    st.pyplot(plot_model_accuracy_comparison(model_accuracies, figsize=(8, 4)), clear_figure=True)
    st.dataframe(
        [{"Model": model_name, "Accuracy (%)": round(score * 100, 2)} for model_name, score in model_accuracies.items()],
        width="stretch",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        f'<div class="chart-card"><div class="chart-card-title">{t("advanced_eval")}</div>',
        unsafe_allow_html=True,
    )
    st.caption(t("cv_caption"))
    with st.spinner(t("cv_running")):
        try:
            cv_summary = cached_cross_validation(training_file)
        except Exception as exc:
            st.warning(f"CV summary unavailable: {exc}")
            import pandas as _pd

            cv_summary = _pd.DataFrame()
    st.dataframe(cv_summary, width="stretch")

    selected_eval_model = st.selectbox(t("eval_model_select"), EVAL_MODEL_NAMES)
    metrics_table = cached_per_class_metrics(selected_eval_model, training_file)
    confusion_df = cached_confusion_matrix(selected_eval_model, training_file)
    metric_column, matrix_column = st.columns([1.1, 1])
    with metric_column:
        st.markdown(f"### {t('per_class_metrics')} — {selected_eval_model}")
        st.dataframe(metrics_table, width="stretch")
    with matrix_column:
        st.markdown(f"### {t('confusion_matrix')} — {selected_eval_model}")
        st.pyplot(plot_confusion_matrix(confusion_df), clear_figure=True)

    st.markdown(f"### {t('calibration')}")
    st.caption(t("calibration_caption"))
    with st.spinner(t("cv_running")):
        calibration = cached_calibration(training_file)
    if calibration["summary"].empty:
        st.info(t("calibration") + " — " + t("missing"))
    else:
        st.dataframe(calibration["summary"], width="stretch")
    st.markdown(
        """
        - **Accuracy →** Model doğru tahmin ediyor mu?
        - **Brier →** Olasılık tahminleri ne kadar hatalı?
        - **ECE →** Modelin söylediği güven ile gerçek başarı ne kadar uyuşuyor?
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_model_statuses(training_file: str = "Training.csv") -> dict[str, dict[str, Any]]:
    return model_statuses(SUPERVISED_MODEL_NAMES, training_file)


def models_are_up_to_date(statuses: dict[str, dict[str, Any]]) -> bool:
    return bool(statuses) and all(status["fresh"] for status in statuses.values())


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")

    # Ensure a default language is present for test runs and first-time users
    if "language" not in st.session_state:
        st.session_state["language"] = "tr"

    language = st.sidebar.selectbox(
        "Dil / Language",
        list(LANGUAGES.keys()),
        format_func=lambda code: LANGUAGES[code],
        index=list(LANGUAGES.keys()).index(st.session_state.get("language", "tr")),
        key="language",
    )
    t = lambda key: translate(language, key)

    training_file = "Training.csv"

    styles_path = Path(__file__).resolve().parent / "assets" / "style.css"
    if styles_path.exists():
        css = styles_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="app-bar">
            <div class="app-brand">
                <div class="app-logo">{APP_ICON}</div>
                <div class="app-name">{t('app_short_name')}
                    <small>{t('app_brand_sub')}</small>
                </div>
            </div>
            <div class="app-nav">
                <span class="app-pill accent">{t('app_badge')}</span>
                <span class="app-pill">{language.upper()}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-title">{t('app_title')}</div>
            <div class="hero-steps">
                <div class="hero-step">
                    <div class="hero-step-num">1</div>
                    <div><b>{t('hero_step1_title')}</b><span>{t('hero_step1_text')}</span></div>
                </div>
                <div class="hero-step">
                    <div class="hero-step-num">2</div>
                    <div><b>{t('hero_step2_title')}</b><span>{t('hero_step2_text')}</span></div>
                </div>
                <div class="hero-step">
                    <div class="hero-step-num">3</div>
                    <div><b>{t('hero_step3_title')}</b><span>{t('hero_step3_text')}</span></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        get_data_fingerprint(training_file)
    except Exception as exc:
        st.error(f"{t('data_missing')} :: {exc}")
        return

    training_frame, symptom_columns = load_clean_training_data(training_file)
    apriori_params = build_sidebar_apriori_params(language)
    llm_settings = {"provider": "off"}

    cache_key = f"models_loaded_{training_file}"
    if cache_key not in st.session_state:
        with st.spinner(t("dataset_loading")):
            _ = _get_cached_preprocessed(training_file)
        st.session_state[cache_key] = True

    selected_symptoms, analyze_clicked, severity_map, duration_map = build_sidebar(symptom_columns, language)

    model_status_state = get_model_statuses(training_file)
    if not models_are_up_to_date(model_status_state):
        st.sidebar.warning(t("stale_warning"))

    model_accuracies = load_model_accuracies(training_file)

    if analyze_clicked and not selected_symptoms:
        st.warning(t("select_at_least_one"))

    if analyze_clicked and selected_symptoms:
        with st.spinner(t("running_models")):
            prediction_bundle = predict_from_symptoms(
                selected_symptoms,
                training_file,
                severity_map=severity_map if 'severity_map' in locals() else {},
                duration_map=duration_map if 'duration_map' in locals() else {},
                apriori_params=apriori_params,
            )
        st.session_state["prediction_bundle"] = prediction_bundle
        st.session_state["analyzed_symptoms"] = list(selected_symptoms)
        per_model = {
            "decision_tree": prediction_bundle.decision_tree_prediction,
            "naive_bayes": prediction_bundle.naive_bayes_prediction,
            "random_forest": prediction_bundle.random_forest_prediction,
            "logistic_regression": prediction_bundle.logistic_regression_prediction,
            "svm": prediction_bundle.svm_prediction,
            "xgboost": prediction_bundle.xgboost_prediction,
            "lightgbm": prediction_bundle.lightgbm_prediction,
        }
        log_prediction(
            training_file=training_file,
            symptoms=list(selected_symptoms),
            model_predictions=per_model,
            final_prediction=getattr(prediction_bundle, "ensemble_prediction", render_final_result(all_model_predictions(prediction_bundle))),
            confidence_level=compute_confidence(prediction_bundle.naive_bayes_probabilities, language)[0],
            ood=prediction_bundle.ood,
            agreement=model_agreement(list(per_model.values())),
        )
    else:
        prediction_bundle = st.session_state.get("prediction_bundle")

    main_column, chat_column = st.columns([2.2, 1], gap="large")
    with chat_column:
        final_prediction = None
        if prediction_bundle is not None:
            final_prediction = getattr(prediction_bundle, "ensemble_prediction", render_final_result(all_model_predictions(prediction_bundle)))
        render_chatbot_section(
            st.session_state.get("analyzed_symptoms", selected_symptoms),
            final_prediction,
            prediction_bundle,
            model_accuracies,
            language,
            llm_settings,
            apriori_params,
            severity_map,
            duration_map,
            training_file,
        )

    with main_column:
        tab_prediction, tab_disease, tab_visualization, tab_evaluation, tab_explainability = st.tabs(
            [
                t("tab_prediction"),
                t("tab_disease_info"),
                t("tab_visualization"),
                t("tab_evaluation"),
                t("tab_explainability"),
            ]
        )

        with tab_disease:
            final_prediction = None
            if prediction_bundle is not None:
                final_prediction = getattr(prediction_bundle, "ensemble_prediction", render_final_result(all_model_predictions(prediction_bundle)))
            render_disease_info_tab(training_frame, symptom_columns, final_prediction, language)

        if prediction_bundle is None:
            with tab_prediction:
                st.info(t("select_first"))
            with tab_visualization:
                st.subheader(t("data_visualizations"))
                left_column, right_column = st.columns(2)
                with left_column:
                    st.pyplot(plot_disease_distribution(training_frame), clear_figure=True)
                with right_column:
                    st.pyplot(plot_top_symptoms(training_frame), clear_figure=True)
            with tab_evaluation:
                render_evaluation_tab(model_accuracies, model_status_state, language, training_file)
            with tab_explainability:
                st.info(t("select_first"))
            st.caption(t("privacy"))
            return

        selected_symptoms = st.session_state.get("analyzed_symptoms", selected_symptoms)
        final_prediction = getattr(prediction_bundle, "ensemble_prediction", render_final_result(all_model_predictions(prediction_bundle)))

        with tab_prediction:
            render_prediction_tab(selected_symptoms, final_prediction, prediction_bundle, model_accuracies, language)

        with tab_visualization:
            render_visualization_tab(training_frame, prediction_bundle, language)

        with tab_evaluation:
            render_evaluation_tab(model_accuracies, model_status_state, language, training_file)

        with tab_explainability:
            render_explainability_tab(prediction_bundle, language)

        st.caption(t("privacy"))


if __name__ == "__main__":
    main()
