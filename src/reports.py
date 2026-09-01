"""Report generation helpers (plain text, HTML, PDF)."""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Any

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from src.icd10 import get_icd10_code
from src.utils import display_symptom_name, humanize_label


def _cell_next_line(pdf: FPDF, height: float, text: str) -> None:
    pdf.cell(0, height, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def _sanitize_for_pdf(s: str, max_token: int = 80, max_total: int = 1000) -> str:
    import re

    if s is None:
        return ""
    s = str(s)
    s = s.replace("�", "-")
    s = s.replace("Ö", "O").replace("ö", "o")
    s = s.replace("Ü", "U").replace("ü", "u")
    s = s.replace("Ç", "C").replace("ç", "c")
    s = s.replace("Ş", "S").replace("ş", "s")
    s = s.replace("Ğ", "G").replace("ğ", "g")
    s = s.replace("ı", "i")
    s = re.sub(r"\S{%d,}" % max_token, lambda m: m.group(0)[: max_token - 3] + "...", s)
    return s[:max_total]


def _multi_cell(pdf: FPDF, height: float, text: str) -> None:
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, height, _sanitize_for_pdf(text))


def build_pdf_report(
    selected_symptoms: list[str],
    final_prediction: str,
    prediction_bundle: Any,
    model_accuracies: dict[str, float],
    confidence_level: str,
    confidence_detail: str,
    language: str = "tr",
) -> bytes:
    """Render a downloadable PDF report."""

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)

    title = "Hastalik Tahmin Raporu" if language == "tr" else "Disease Prediction Report"
    _cell_next_line(pdf, 10, title)
    pdf.set_font("Helvetica", size=11)
    pdf.ln(4)

    # Wrap generation in try/except to avoid FPDF unicode font issues on some systems.
    try:
        symptoms_text = ", ".join(display_symptom_name(symptom, language) for symptom in selected_symptoms)
        _multi_cell(pdf, 7, f"{'Secilen belirtiler' if language == 'tr' else 'Selected symptoms'}: {symptoms_text}")
        _multi_cell(pdf, 7, f"{'Birlesik sonuc' if language == 'tr' else 'Combined result'}: {humanize_label(final_prediction)}")
        icd10_code = get_icd10_code(final_prediction)
        _multi_cell(pdf, 7, f"{'ICD-10 kodu' if language == 'tr' else 'ICD-10 code'}: {icd10_code or '-'}")
        _multi_cell(pdf, 7, f"{'Guven seviyesi' if language == 'tr' else 'Confidence'}: {confidence_level} - {confidence_detail}")

        pdf.ln(4)
        differential = getattr(prediction_bundle, "differential_diagnosis", None) or []
        if differential:
            pdf.set_font("Helvetica", "B", 12)
            _cell_next_line(pdf, 8, "Ayirici Tani Listesi" if language == "tr" else "Differential Diagnosis")
            pdf.set_font("Helvetica", size=10)
            for entry in differential:
                entry_code = get_icd10_code(entry["disease"]) or "-"
                _cell_next_line(pdf, 6, f"{entry['disease']} (ICD-10: {entry_code}) - %{entry['score_pct']}")

        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 12)
        _cell_next_line(pdf, 8, "Model Tahminleri" if language == "tr" else "Model Predictions")
        pdf.set_font("Helvetica", size=10)
        model_predictions = [
            ("Decision Tree", prediction_bundle.decision_tree_prediction),
            ("Naive Bayes", prediction_bundle.naive_bayes_prediction),
            ("Random Forest", prediction_bundle.random_forest_prediction),
            ("Logistic Regression", prediction_bundle.logistic_regression_prediction),
            ("SVM", prediction_bundle.svm_prediction),
            ("XGBoost", prediction_bundle.xgboost_prediction),
            ("LightGBM", prediction_bundle.lightgbm_prediction),
        ]
        for model_name, prediction in model_predictions:
            _cell_next_line(pdf, 6, f"{model_name}: {humanize_label(prediction)}")

        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 12)
        _cell_next_line(pdf, 8, "Model Dogruluklari (%)" if language == "tr" else "Model Accuracies (%)")
        pdf.set_font("Helvetica", size=10)
        for model_name, score in model_accuracies.items():
            _cell_next_line(pdf, 6, f"{model_name}: {round(score * 100, 1)}")

        if not prediction_bundle.naive_bayes_probabilities.empty:
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 12)
            _cell_next_line(
                pdf,
                8,
                "Naive Bayes Olasiliklari (Top 5)" if language == "tr" else "Naive Bayes Probabilities (Top 5)",
            )
            pdf.set_font("Helvetica", size=10)
            for _, row in prediction_bundle.naive_bayes_probabilities.head(5).iterrows():
                _cell_next_line(pdf, 6, f"{row['prognosis']}: %{row['probability_pct']:.2f}")

        pdf.ln(4)
        disclaimer = (
            "Bu rapor karar destek amaclidir; tibbi tani koymaz."
            if language == "tr"
            else "This report is for decision support only; it is not a medical diagnosis."
        )
        pdf.set_font("Helvetica", "I", 9)
        _multi_cell(pdf, 5, disclaimer)
        _cell_next_line(pdf, 5, datetime.now().strftime("%d.%m.%Y %H:%M"))

    except Exception:
        # Fallback: use English safe labels to avoid unicode issues with core fonts
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        _cell_next_line(pdf, 10, "Disease Prediction Report")
        pdf.set_font("Helvetica", size=11)
        pdf.ln(4)

        symptoms_text = ", ".join(display_symptom_name(symptom, "en") for symptom in selected_symptoms)
        _multi_cell(pdf, 7, f"Selected symptoms: {symptoms_text}")
        _multi_cell(pdf, 7, f"Combined result: {humanize_label(final_prediction)}")
        icd10_code = get_icd10_code(final_prediction)
        _multi_cell(pdf, 7, f"ICD-10 code: {icd10_code or '-'}")
        _multi_cell(pdf, 7, f"Confidence: {confidence_level} - {confidence_detail}")
        pdf.ln(4)

        differential = getattr(prediction_bundle, "differential_diagnosis", None) or []
        if differential:
            pdf.set_font("Helvetica", "B", 12)
            _cell_next_line(pdf, 8, "Differential Diagnosis")
            pdf.set_font("Helvetica", size=10)
            for entry in differential:
                entry_code = get_icd10_code(entry["disease"]) or "-"
                _cell_next_line(
                    pdf,
                    6,
                    f"{entry['disease']} (ICD-10: {entry_code}) - %{entry['score_pct']}",
                )
                pdf.ln(3)

        pdf.set_font("Helvetica", "B", 12)
        _cell_next_line(pdf, 8, "Model Predictions")
        pdf.set_font("Helvetica", size=10)
        for model_name, prediction in [
            ("Decision Tree", prediction_bundle.decision_tree_prediction),
            ("Naive Bayes", prediction_bundle.naive_bayes_prediction),
            ("Random Forest", prediction_bundle.random_forest_prediction),
            ("Logistic Regression", prediction_bundle.logistic_regression_prediction),
            ("SVM", prediction_bundle.svm_prediction),
            ("XGBoost", prediction_bundle.xgboost_prediction),
            ("LightGBM", prediction_bundle.lightgbm_prediction),
        ]:
            _cell_next_line(pdf, 6, f"{model_name}: {humanize_label(prediction)}")

        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 12)
        _cell_next_line(pdf, 8, "Model Accuracies (%)")
        pdf.set_font("Helvetica", size=10)
        for model_name, score in model_accuracies.items():
            _cell_next_line(pdf, 6, f"{model_name}: {round(score * 100, 1)}")

        pdf.ln(4)
        pdf.set_font("Helvetica", "I", 9)
        _multi_cell(pdf, 5, "This report is for decision support only; it is not a medical diagnosis.")
        _cell_next_line(pdf, 5, datetime.now().strftime("%d.%m.%Y %H:%M"))

    buffer = BytesIO()
    pdf.output(buffer)
    return buffer.getvalue()
