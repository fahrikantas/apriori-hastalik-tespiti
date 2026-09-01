# -*- coding: utf-8 -*-
"""ICD-10 code mapping tests."""

from __future__ import annotations

import re

from src.icd10 import (
    DISEASE_ICD10,
    all_disease_codes,
    get_icd10_code,
    icd10_chapter,
    icd10_summary,
)
from src.preprocess import preprocess_training_data

ICD10_PATTERN = re.compile(r"^[A-TV-Z][0-9][0-9AB](?:\.[0-9A-Z]{1,2})?$")


def test_every_dataset_disease_has_a_code():
    preprocessed = preprocess_training_data("Training.csv")
    diseases = sorted(preprocessed.frame["prognosis"].unique())
    assert len(diseases) == len(DISEASE_ICD10)
    missing = [disease for disease in diseases if get_icd10_code(disease) is None]
    assert not missing


def test_codes_follow_icd10_syntax():
    for code in DISEASE_ICD10.values():
        assert ICD10_PATTERN.match(code), code


def test_unknown_disease_returns_none():
    assert get_icd10_code("Not a real disease") is None
    assert get_icd10_code(None) is None
    assert get_icd10_code("") is None


def test_lookup_is_case_insensitive():
    assert get_icd10_code("fungal infection") == get_icd10_code("Fungal infection")
    assert get_icd10_code("HEPATITIS A") == "B15.9"


def test_chapter_lookup():
    assert icd10_chapter("I10") == "Diseases of the circulatory system"
    assert icd10_chapter("K21.9") == "Diseases of the digestive system"
    assert icd10_chapter(None) is None


def test_summary_shape():
    summary = icd10_summary("Diabetes")
    assert summary == {"code": "E14.9", "chapter": "Endocrine, nutritional and metabolic diseases"}
    assert icd10_summary("Unknown") == {"code": None, "chapter": None}


def test_all_disease_codes_is_a_stable_copy():
    mapping = all_disease_codes()
    mapping["Diabetes"] = "X99"
    assert all_disease_codes()["Diabetes"] == "E14.9"