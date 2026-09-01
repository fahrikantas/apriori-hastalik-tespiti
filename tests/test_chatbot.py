# -*- coding: utf-8 -*-
"""Chatbot behavior tests."""

from __future__ import annotations

from src.chatbot import (
    DISCLAIMER,
    build_chatbot_response,
    detect_diseases,
    detect_symptoms,
    stream_chat_reply,
)
from tests.conftest import build_reply


def test_greeting(knowledge_base):
    reply = build_reply(knowledge_base, "merhaba")
    assert "asistanıyım" in reply


def test_thanks(knowledge_base):
    reply = build_reply(knowledge_base, "teşekkürler")
    assert "Rica ederim" in reply


def test_help(knowledge_base):
    reply = build_reply(knowledge_base, "yardım eder misin?")
    assert "yardımcı olabileceğim konular" in reply


def test_disease_info_dengue(knowledge_base):
    reply = build_reply(knowledge_base, "Dengue nedir?")
    assert "Dengue" in reply
    assert DISCLAIMER in reply


def test_disease_info_malaria(knowledge_base):
    reply = build_reply(knowledge_base, "sıtma ne demek")
    assert "Malaria" in reply


def test_symptom_detection(knowledge_base):
    symptoms = detect_symptoms("ateşim ve öksürüğüm var", knowledge_base)
    assert "cough" in symptoms
    assert "high_fever" in symptoms


def test_disease_detection(knowledge_base):
    diseases = detect_diseases("dengue nedir", knowledge_base)
    assert "dengue" in [d.lower() for d in diseases]


def test_boundary_no_false_positive(knowledge_base):
    # 'yapma' must not match symptom 'yapmalıyım' style tokens
    symptoms = detect_symptoms("yapma", knowledge_base)
    assert "yapma" not in symptoms


def test_frequency_question(knowledge_base):
    reply = build_reply(knowledge_base, "yüksek ateş dengue'de görülür mü?")
    assert "Dengue" in reply
    assert "%" in reply


def test_how_it_works(knowledge_base, model_accuracies_dict):
    reply = build_reply(knowledge_base, "Sistem nasıl çalışıyor?", model_accuracies=model_accuracies_dict)
    assert "Apriori" in reply


def test_unknown_fallback(knowledge_base):
    reply = build_reply(knowledge_base, "xyzabc çok anlamsız bir cümle")
    assert reply  # non-empty fallback


def test_stream_yields_text(knowledge_base):
    chunks = list(stream_chat_reply("Merhaba dünya", word_delay=0))
    assert "".join(chunks).strip() == "Merhaba dünya"