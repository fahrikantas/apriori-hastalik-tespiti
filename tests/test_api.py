# -*- coding: utf-8 -*-
"""REST API layer tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

import api


def test_health_endpoint():
    client = TestClient(api.app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_schema_version"] >= 1


def test_symptoms_endpoint_lists_all():
    client = TestClient(api.app)
    response = client.get("/api/symptoms")
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 132
    assert {"code", "label_tr", "label_en"} == set(payload["symptoms"][0])


def test_predict_endpoint_returns_full_result():
    client = TestClient(api.app)
    response = client.post("/api/predict", json={"symptoms": ["itching", "skin_rash"]})
    assert response.status_code == 200
    payload = response.json()
    assert payload["final_prediction"]
    assert set(payload["per_model_predictions"]) == {
        "decision_tree",
        "naive_bayes",
        "random_forest",
        "logistic_regression",
        "svm",
        "xgboost",
        "lightgbm",
    }
    assert "out_of_distribution" in payload
    assert "apriori_recommendations" in payload
    assert "icd10" in payload
    assert payload["icd10"] is not None
    assert isinstance(payload["differential_diagnosis"], list)
    assert payload["differential_diagnosis"]
    assert isinstance(payload["red_flags"], list)


def test_predict_rejects_empty_symptoms():
    client = TestClient(api.app)
    response = client.post("/api/predict", json={"symptoms": []})
    assert response.status_code == 422


def test_predict_rejects_unknown_dataset():
    client = TestClient(api.app)
    response = client.post(
        "/api/predict",
        json={"symptoms": ["itching"], "training_file": "nope.csv"},
    )
    assert response.status_code == 400