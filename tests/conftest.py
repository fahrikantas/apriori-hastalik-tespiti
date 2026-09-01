# -*- coding: utf-8 -*-
"""Shared pytest fixtures for the disease prediction project."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.chatbot import build_chatbot_response, build_knowledge_base
from src.preprocess import preprocess_training_data


@pytest.fixture(scope="session")
def knowledge_base():
    preprocessed = preprocess_training_data()
    return build_knowledge_base(preprocessed.frame, preprocessed.symptom_columns)


@pytest.fixture(scope="session")
def model_accuracies_dict():
    return {"Decision Tree": 0.7, "Naive Bayes": 0.98, "Random Forest": 1.0, "Logistic Regression": 0.95, "SVM": 0.96}


def build_reply(kb, message, **kwargs):
    return build_chatbot_response(
        message,
        selected_symptoms=kwargs.get("selected_symptoms", []),
        final_prediction=kwargs.get("final_prediction"),
        prediction_bundle=kwargs.get("prediction_bundle"),
        context=kwargs.get("context", {}),
        kb=kb,
        model_accuracies=kwargs.get("model_accuracies"),
    )[0]