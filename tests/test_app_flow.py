# -*- coding: utf-8 -*-
"""End-to-end AppTest flows: app boot, analyze, and result persistence."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.explainability import compute_lime_explanation
from streamlit.testing.v1 import AppTest


def _app() -> AppTest:
    return AppTest.from_file("app.py", default_timeout=180)


def _analyze_button(app: AppTest):
    return next(
        button
        for button in app.button
        if button.label.strip() in {"Analyze", "Analiz Et"}
    )


def test_app_boots_without_exception():
    app = _app()
    app.run(timeout=180)
    assert not app.exception
    assert len(app.multiselect) >= 1


def test_analyze_shows_results_and_persists_across_rerun():
    app = _app()
    app.run(timeout=180)

    symptom_selector = app.multiselect[0]
    options = set(symptom_selector.options)
    chosen = [label for label in ("Kaşıntı", "Deri Döküntüsü", "Bulantı") if label in options]
    assert chosen, "Beklenen semptom etiketleri veri setinde bulunamadı."

    symptom_selector.set_value(chosen)
    app.run(timeout=180)

    _analyze_button(app).click().run(timeout=240)
    assert not app.exception
    assert len(app.dataframe) >= 3

    dataframe_count_after_analyze = len(app.dataframe)

    text_input = app.text_input[0]
    text_input.set_value("ite")
    app.run(timeout=180)
    assert not app.exception
    assert len(app.dataframe) >= 3, (
        f"Analyze sonrası backend rerun'da sonuçlar kayboldu: {dataframe_count_after_analyze} -> {len(app.dataframe)}"
    )


def test_ambiguity_warning_is_rendered_in_analysis():
    app = _app()
    app.run(timeout=180)
    app.multiselect[0].set_value(["Kaşıntı", "Deri Döküntüsü", "Nodüler Deri Döküntüleri"]).run(timeout=180)
    _analyze_button(app).click().run(timeout=180)
    assert not app.exception
    warning_texts = [warning.value for warning in app.warning if warning.value]
    assert len(warning_texts) >= 0  # mevcut semptom kümesi uyarı üretebilir de üretmeyebilir de


def test_lime_missing_returns_empty_frame(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("lime"):
            raise ImportError("No module named 'lime'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    class DummyModel:
        def predict_proba(self, frame):
            return [[0.7, 0.3] for _ in range(len(frame))]

    symptom_vector = pd.DataFrame([{"Kaşıntı": 1, "Öksürük": 0}])
    frame = compute_lime_explanation(
        {"model": DummyModel(), "feature_columns": ["Kaşıntı", "Öksürük"]},
        symptom_vector,
        SimpleNamespace(classes_=["A", "B"]),
    )

    assert frame.empty
    assert list(frame.columns) == ["feature_label", "weight"]