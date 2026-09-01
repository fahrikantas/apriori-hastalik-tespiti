# -*- coding: utf-8 -*-
"""Red-flag rules (single + combination) tests."""

from __future__ import annotations

from src.chatbot import _red_flag_warning
from src.red_flags import RED_FLAG_ADVICE, RED_FLAG_COMBINATIONS, check_red_flags


def test_single_symptom_flag_triggers():
    flags = check_red_flags(["breathlessness"])
    assert len(flags) == 1
    assert flags[0]["rule"] == "single"
    assert flags[0]["matched_symptoms"] == ["breathlessness"]
    assert flags[0]["advice_tr"]
    assert flags[0]["advice_en"]


def test_new_single_flags_are_registered():
    assert "weakness_of_one_body_side" in RED_FLAG_ADVICE
    assert "slurred_speech" in RED_FLAG_ADVICE
    assert "dehydration" in RED_FLAG_ADVICE
    assert len(RED_FLAG_ADVICE) > 15


def test_combination_requires_all_symptoms():
    # high fever alone: no meningitis rule
    flags = check_red_flags(["high_fever"])
    assert not [flag for flag in flags if flag["id"] == "menengitis_suspicion"]

    flags = check_red_flags(["high_fever", "stiff_neck"])
    assert [flag for flag in flags if flag["id"] == "menengitis_suspicion"]


def test_stroke_combination():
    flags = check_red_flags(["weakness_of_one_body_side", "slurred_speech"])
    assert [flag for flag in flags if flag["id"] == "stroke_hint"]


def test_no_flags_for_benign_selection():
    assert check_red_flags(["itching", "skin_rash"]) == []
    assert check_red_flags([]) == []


def test_combinations_use_real_dataset_symptoms():
    from src.preprocess import preprocess_training_data

    preprocessed = preprocess_training_data("Training.csv")
    vocabulary = set(preprocessed.symptom_columns)
    unknown = []
    for combination in RED_FLAG_COMBINATIONS:
        unknown.extend(s for s in combination["symptoms"] if s not in vocabulary)
    assert not unknown


def test_chatbot_warning_includes_combination():
    warning = _red_flag_warning(["high_fever", "stiff_neck"])
    assert "Dikkat" in warning
    assert "menenjit" in warning.lower()

    assert _red_flag_warning(["itching"]) == ""