"""Tests for the chatbot content registry (data/content/*.json)."""

import json

import pytest

from src.chatbot import (
    CONTENT_DIR,
    EXTRA_SYMPTOM_ALIASES,
    SYMPTOM_ADVICE,
    SYMPTOM_DESCRIPTIONS,
    TURKISH_DISEASE_ALIASES,
    _load_json_content,
    _load_str_dict,
)
from src.utils import TARGET_COLUMN, load_dataset, normalize_search_text


def _training_frame():
    return load_dataset("Training.csv")


def _dataset_symptom_columns():
    frame = _training_frame()
    return [col for col in frame.columns if col != TARGET_COLUMN]


def _normalized_columns():
    return {col.replace(" ", "_").strip("_") for col in _dataset_symptom_columns()}


def test_content_files_exist_and_are_valid_json():
    for name in (
        "symptom_advice.json",
        "symptom_descriptions.json",
        "turkish_disease_aliases.json",
        "extra_symptom_aliases.json",
    ):
        path = CONTENT_DIR / name
        assert path.is_file(), f"missing {path}"
        json.loads(path.read_text(encoding="utf-8"))


def test_content_has_expected_sizes():
    assert len(SYMPTOM_ADVICE) == 23
    assert len(SYMPTOM_DESCRIPTIONS) == 106
    assert len(TURKISH_DISEASE_ALIASES) == 41
    assert len(EXTRA_SYMPTOM_ALIASES) == 13


def test_every_dataset_disease_has_turkish_aliases():
    frame = _training_frame()
    diseases = set(frame[TARGET_COLUMN].unique())
    assert diseases
    normalized_aliases = {normalize_search_text(name) for name in TURKISH_DISEASE_ALIASES}
    for disease in diseases:
        assert normalize_search_text(disease) in normalized_aliases, f"no aliases for {disease}"


def test_alias_values_are_truthy_ascii_strings():
    for disease, aliases in TURKISH_DISEASE_ALIASES.items():
        assert disease and isinstance(disease, str)
        assert isinstance(aliases, tuple) and len(aliases) >= 1
        for alias in aliases:
            assert alias and isinstance(alias, str)
            assert all(ord(ch) < 128 for ch in alias), f"alias not ASCII: {alias!r}"


def test_symptom_descriptions_cover_dataset_symptoms():
    cols = _normalized_columns()
    missing = [col for col in sorted(cols) if col not in SYMPTOM_DESCRIPTIONS]
    assert len(missing) <= 29, f"too many missing descriptions: {missing}"


def test_symptom_descriptions_keys_are_known_symptoms():
    cols = _normalized_columns()
    known_extras = {"foul_smell_of_urine", "killed_by_insects", "paleness", "spotting_urination"}
    unexpected = set(SYMPTOM_DESCRIPTIONS) - cols - known_extras
    assert not unexpected, f"unexpected description keys: {sorted(unexpected)}"


def test_symptom_advice_keys_are_dataset_symptoms():
    cols = _normalized_columns()
    unexpected = set(SYMPTOM_ADVICE) - cols
    assert unexpected <= {"sore_throat"}, f"unexpected advice keys: {sorted(unexpected)}"


def test_extra_aliases_map_to_real_symptoms():
    assert set(EXTRA_SYMPTOM_ALIASES) <= set(SYMPTOM_DESCRIPTIONS)


def test_missing_file_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        _load_json_content("does_not_exist.json")


def test_invalid_json_raises_value_error(tmp_path, monkeypatch):
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr("src.chatbot.CONTENT_DIR", tmp_path)
    with pytest.raises(ValueError):
        _load_str_dict("bad.json")
