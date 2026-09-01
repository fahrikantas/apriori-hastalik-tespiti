"""Tests for the optional LLM integration (incl. Anthropic symptom extraction)."""

import json

import pytest

from src import llm
from src.llm import (
    _clean_symptom_code,
    _parse_symptom_json,
    extract_symptoms_with_llm,
    generate_llm_reply,
    llm_available,
)

VALID = {"high_fever", "cough", "headache", "nausea", "stiff_neck", "fatigue", "chest_pain"}


@pytest.mark.parametrize(
    "settings,expected",
    [
        ({"provider": "off"}, False),
        ({"provider": "ollama", "ollama_url": "", "ollama_model": "llama3.2"}, False),
        ({"provider": "ollama", "ollama_url": "http://localhost:11434", "ollama_model": "llama3.2"}, True),
        ({"provider": "openai", "openai_api_key": "", "openai_model": "gpt-4o-mini"}, False),
        ({"provider": "openai", "openai_api_key": "sk-123", "openai_model": "gpt-4o-mini"}, True),
        ({"provider": "anthropic", "anthropic_api_key": "", "anthropic_model": "claude-sonnet-4-20250514"}, False),
        ({"provider": "anthropic", "anthropic_api_key": "sk-ant-123", "anthropic_model": "claude-sonnet-4-20250514"}, True),
    ],
)
def test_llm_available(settings, expected):
    assert llm_available(settings) is expected


def test_llm_available_anthropic_env_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
    settings = {"provider": "anthropic", "anthropic_model": "claude-sonnet-4-20250514"}
    assert llm_available(settings) is True


def test_clean_symptom_code_accepts_exact_and_cleans():
    assert _clean_symptom_code("high_fever", VALID) == "high_fever"
    assert _clean_symptom_code("high fever", VALID) == "high_fever"
    assert _clean_symptom_code("High-Fever", VALID) == "high_fever"
    assert _clean_symptom_code('"stiff_neck"', VALID) == "stiff_neck"
    assert _clean_symptom_code("unknown_code", VALID) is None
    assert _clean_symptom_code("", VALID) is None


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('["high_fever", "cough"]', ["high_fever", "cough"]),
        ('Here are the symptoms: ["high_fever","cough"]', ["high_fever", "cough"]),
        ("high_fever, cough", ["high_fever", "cough"]),
        ("", []),
        (None, []),
        ("high_fever\ncough", ["high_fever"]),
        ('{"symptoms": ["high_fever"]}', ["high_fever"]),
    ],
)
def test_parse_symptom_json(raw, expected):
    assert _parse_symptom_json(raw) == expected


def test_extract_symptoms_filters_unknown_codes(monkeypatch):
    monkeypatch.setattr(
        llm, "_call_anthropic",
        lambda messages, settings: json.dumps(["high_fever", "unknown", "stiff_neck", "zzz"]),
    )
    settings = {"provider": "anthropic", "anthropic_api_key": "sk-ant-123", "anthropic_model": "m"}
    codes, raw = extract_symptoms_with_llm("Ateşim var ve boynum tutuldu", VALID, settings)
    assert codes == ["high_fever", "stiff_neck"]
    assert raw


def test_extract_symptoms_deduplicates(monkeypatch):
    monkeypatch.setattr(
        llm, "_call_anthropic",
        lambda messages, settings: json.dumps(["cough", "cough", "cough"]),
    )
    settings = {"provider": "anthropic", "anthropic_api_key": "sk-ant-123", "anthropic_model": "m"}
    codes, _ = extract_symptoms_with_llm("Öksürüyorum", VALID, settings)
    assert codes == ["cough"]


def test_extract_symptoms_returns_empty_when_llm_off():
    codes, raw = extract_symptoms_with_llm("Ateşim var", VALID, {"provider": "off"})
    assert codes == []
    assert raw == ""


def test_extract_symptoms_returns_empty_on_exception(monkeypatch):
    def boom(messages, settings):
        raise TimeoutError("no response")

    monkeypatch.setattr(llm, "_call_openai", boom)
    settings = {"provider": "openai", "openai_api_key": "sk-123", "openai_model": "gpt-4o-mini"}
    codes, raw = extract_symptoms_with_llm("Ateşim var", VALID, settings)
    assert codes == []
    assert raw == ""


def test_extract_symptoms_empty_vocabulary(monkeypatch):
    monkeypatch.setattr(llm, "_call_openai", lambda messages, settings: '["x"]')
    settings = {"provider": "openai", "openai_api_key": "sk-123", "openai_model": "gpt-4o-mini"}
    codes, _ = extract_symptoms_with_llm("Ateşim var", [], settings)
    assert codes == []


def test_generate_llm_reply_returns_none_when_unconfigured():
    assert generate_llm_reply("merhaba", "context", {"provider": "off"}) is None


def test_generate_llm_reply_anthropic(monkeypatch):
    def fake_call(messages, settings):
        return "Anthropic yanıtı"

    monkeypatch.setattr(llm, "_call_anthropic", fake_call)
    settings = {"provider": "anthropic", "anthropic_api_key": "sk-ant-123", "anthropic_model": "m"}
    reply = generate_llm_reply("merhaba", "context", settings, "tr")
    assert reply == "Anthropic yanıtı"


def test_generate_llm_reply_swallows_network_error(monkeypatch):
    def boom(messages, settings):
        raise json.JSONDecodeError("bad json", "", 0)

    monkeypatch.setattr(llm, "_call_anthropic", boom)
    settings = {"provider": "anthropic", "anthropic_api_key": "sk-ant-123", "anthropic_model": "m"}
    assert generate_llm_reply("merhaba", "context", settings) is None
