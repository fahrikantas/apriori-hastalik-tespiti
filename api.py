"""REST API layer exposing the prediction pipeline.

Run with::

    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:

- ``GET  /``               API overview
- ``GET  /health``         service + artifact status
- ``GET  /datasets``       available training datasets
- ``GET  /api/symptoms``   known symptom codes with Turkish labels
- ``POST /api/predict``    run all models on a symptom selection

The model ensemble is the same one the Streamlit UI uses, so the API and the
UI stay consistent. Every prediction is also appended to the local telemetry
log (same privacy guarantees as the UI).
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.decision_tree import DECISION_TREE_MODEL_NAME
from src.icd10 import get_icd10_code
from src.lightgbm_model import LIGHTGBM_MODEL_NAME
from src.logistic_regression import LOGISTIC_REGRESSION_MODEL_NAME
from src.model_metadata import model_statuses
from src.naive_bayes import NAIVE_BAYES_MODEL_NAME
from src.predict import predict_from_symptoms
from src.preprocess import get_available_symptoms, preprocess_training_data
from src.random_forest import RANDOM_FOREST_MODEL_NAME
from src.red_flags import check_red_flags
from src.svm import SVM_MODEL_NAME
from src.telemetry import log_prediction, model_agreement
from src.utils import (
    DATASET_ALIASES,
    display_symptom_name,
    resolve_data_path,
)
from src.versioning import MODEL_SCHEMA_VERSION, manifest_status
from src.xgboost_model import XGBOOST_MODEL_NAME

APP_TITLE = "Symptom-Based Disease Prediction API"
APP_VERSION = "2.0.0"

app = FastAPI(title=APP_TITLE, version=APP_VERSION)

SUPERVISED_MODEL_NAMES = [
    DECISION_TREE_MODEL_NAME,
    NAIVE_BAYES_MODEL_NAME,
    RANDOM_FOREST_MODEL_NAME,
    LOGISTIC_REGRESSION_MODEL_NAME,
    SVM_MODEL_NAME,
    XGBOOST_MODEL_NAME,
    LIGHTGBM_MODEL_NAME,
]

MODEL_ORDER = [
    "decision_tree",
    "naive_bayes",
    "random_forest",
    "logistic_regression",
    "svm",
    "xgboost",
    "lightgbm",
]


class PredictRequest(BaseModel):
    """The symptom selection to evaluate."""

    symptoms: list[str] = Field(..., min_length=1, description="Symptom codes to analyze")
    training_file: str = Field(default="Training.csv", description="Dataset to use")

    @field_validator("symptoms")
    @classmethod
    def symptoms_not_empty(cls, value: list[str]) -> list[str]:
        if not value or not any(str(item).strip() for item in value):
            raise ValueError("At least one non-empty symptom is required.")
        return [str(item).strip() for item in value if str(item).strip()]


class HealthResponse(BaseModel):
    """Service and artifact health."""

    status: str
    app_version: str
    model_schema_version: int
    manifest: dict[str, Any]
    models_up_to_date: bool


def _confidence_level(probability_frame: Any) -> str:
    if probability_frame is None or probability_frame.empty:
        return "unknown"
    top = float(probability_frame.iloc[0]["probability_pct"])
    if top >= 60:
        return "high"
    if top >= 30:
        return "medium"
    return "low"


def _apriori_rows(rules: Any, top_n: int = 5) -> list[dict[str, Any]]:
    if rules is None or getattr(rules, "empty", True):
        return []
    columns = list(rules.columns)
    rows: list[dict[str, Any]] = []
    for _, row in rules.head(top_n).iterrows():
        entry: dict[str, Any] = {}
        for column in columns:
            value = row[column]
            if hasattr(value, "issubset"):
                try:
                    value = sorted(str(item) for item in value)
                except Exception:
                    value = str(value)
            entry[column] = value
        rows.append(entry)
    return rows


def _datasets() -> list[str]:
    available: list[str] = []
    for alias in DATASET_ALIASES:
        try:
            path = resolve_data_path(alias)
            if path.exists():
                available.append(alias)
        except Exception:
            continue
    return available or ["Training.csv"]


@app.get("/", include_in_schema=False)
def root() -> dict[str, Any]:
    return {
        "app": APP_TITLE,
        "version": APP_VERSION,
        "docs": "/docs",
        "endpoints": [
            "/health",
            "/datasets",
            "/api/symptoms",
            "/api/predict",
        ],
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    manifest = manifest_status("Training.csv")
    statuses = model_statuses(SUPERVISED_MODEL_NAMES, "Training.csv")
    up_to_date = bool(statuses) and all(status["fresh"] for status in statuses.values())
    return HealthResponse(
        status="ok",
        app_version=APP_VERSION,
        model_schema_version=MODEL_SCHEMA_VERSION,
        manifest=manifest,
        models_up_to_date=up_to_date,
    )


@app.get("/datasets")
def datasets() -> dict[str, Any]:
    return {"datasets": _datasets()}


@app.get("/api/symptoms")
def list_symptoms() -> dict[str, Any]:
    """Return every symptom code the models understand, with a Turkish label."""

    preprocessed = preprocess_training_data("Training.csv")
    symptom_codes = get_available_symptoms(preprocessed.frame)
    return {
        "count": len(symptom_codes),
        "symptoms": [
            {
                "code": code,
                "label_tr": display_symptom_name(code, "tr"),
                "label_en": display_symptom_name(code, "en"),
            }
            for code in symptom_codes
        ],
    }


@app.post("/api/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    """Run the full ensemble on a symptom selection and return the result."""

    training_file = request.training_file
    available = _datasets()
    if training_file not in available:
        raise HTTPException(status_code=400, detail=f"Unknown dataset '{training_file}'. Available: {available}")

    try:
        bundle = predict_from_symptoms(request.symptoms, training_file)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f"Model artifact missing: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    per_model = {
        "decision_tree": bundle.decision_tree_prediction,
        "naive_bayes": bundle.naive_bayes_prediction,
        "random_forest": bundle.random_forest_prediction,
        "logistic_regression": bundle.logistic_regression_prediction,
        "svm": bundle.svm_prediction,
        "xgboost": bundle.xgboost_prediction,
        "lightgbm": bundle.lightgbm_prediction,
    }
    predictions = [per_model[name] for name in MODEL_ORDER]
    final_prediction = bundle.ensemble_prediction
    confidence_level = _confidence_level(bundle.naive_bayes_probabilities)
    agreement = model_agreement(predictions)

    log_prediction(
        training_file=training_file,
        symptoms=request.symptoms,
        model_predictions=per_model,
        final_prediction=final_prediction,
        confidence_level=confidence_level,
        ood=bundle.ood,
        agreement=agreement,
    )

    return {
        "training_file": training_file,
        "requested_symptoms": request.symptoms,
        "final_prediction": final_prediction,
        "icd10": get_icd10_code(final_prediction),
        "confidence": confidence_level,
        "per_model_predictions": per_model,
        "agreement": agreement,
        "top3_diseases": [
            {"disease": str(row["prognosis"]), "probability_pct": float(row["probability_pct"])}
            for _, row in bundle.naive_bayes_probabilities.head(3).iterrows()
        ],
        "differential_diagnosis": [
            {
                "disease": entry["disease"],
                "icd10": get_icd10_code(entry["disease"]),
                "score_pct": entry["score_pct"],
                "support_count": entry["support_count"],
                "supporting_models": entry["supporting_models"],
            }
            for entry in (bundle.differential_diagnosis or [])
        ],
        "red_flags": check_red_flags(request.symptoms),
        "naive_bayes_probabilities": [
            {"disease": str(row["prognosis"]), "probability_pct": float(row["probability_pct"])}
            for _, row in bundle.naive_bayes_probabilities.iterrows()
        ],
        "out_of_distribution": bundle.ood,
        "apriori_recommendations": _apriori_rows(bundle.apriori_rules),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)