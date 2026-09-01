# -*- coding: utf-8 -*-
"""Calibration (Brier / ECE) tests."""

from __future__ import annotations

from src.evaluation import naive_bayes_calibration


def test_calibration_summary_schema():
    result = naive_bayes_calibration("Synthetic.csv")
    assert not result["summary"].empty
    assert {"Model", "Brier Skoru (düşük iyi)", "ECE (%)", "Bölme Sayısı", "Örnek"}.issubset(
        result["summary"].columns
    )
    row = result["summary"].iloc[0]
    assert 0.0 <= row["Brier Skoru (düşük iyi)"] <= 1.0
    assert 0.0 <= row["ECE (%)"] <= 100.0


def test_calibration_reliability_table():
    result = naive_bayes_calibration("Synthetic.csv")
    assert {"Güven Bölmesi", "Örnek Sayısı", "Ort. Güven", "Gözlenen Doğruluk"}.issubset(
        result["reliability"].columns
    )
    assert len(result["reliability"]) == 10