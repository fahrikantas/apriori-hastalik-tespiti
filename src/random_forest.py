"""Random Forest model training and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.preprocess import preprocess_training_data
from src.utils import TARGET_COLUMN, resolve_model_path

RANDOM_FOREST_MODEL_NAME = "random_forest.pkl"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_ESTIMATORS = 100


@dataclass(frozen=True)
class RandomForestResult:
    """Training result bundle for Random Forest."""

    model: Any
    label_encoder: LabelEncoder
    accuracy: float
    classification_report_text: str
    feature_importance: pd.DataFrame
    feature_columns: list[str]
    model_path: str


def _prepare_training_split(file_name: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load the cleaned dataset and return the supervised learning split."""

    preprocessed = preprocess_training_data(file_name)
    frame = preprocessed.frame
    feature_columns = preprocessed.symptom_columns
    features = frame[feature_columns]
    target = frame[TARGET_COLUMN]
    return features, target, feature_columns


def train_random_forest(
    file_name: str = "Training.csv",
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_estimators: int = DEFAULT_ESTIMATORS,
) -> RandomForestResult:
    """Train a Random Forest classifier and persist the fitted model."""

    features, target, feature_columns = _prepare_training_split(file_name)
    label_encoder = LabelEncoder()
    encoded_target = label_encoder.fit_transform(target)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        encoded_target,
        test_size=test_size,
        random_state=random_state,
        stratify=encoded_target,
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = float(accuracy_score(y_test, predictions))
    report_text = classification_report(
        y_test,
        predictions,
        target_names=label_encoder.classes_,
        zero_division=0,
    )

    feature_importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values(by="importance", ascending=False).reset_index(drop=True)

    model_path = resolve_model_path(RANDOM_FOREST_MODEL_NAME)
    joblib.dump(
        {
            "model": model,
            "label_encoder": label_encoder,
            "feature_columns": feature_columns,
            "accuracy": accuracy,
            "classification_report": report_text,
            "feature_importance": feature_importance,
        },
        model_path,
    )

    return RandomForestResult(
        model=model,
        label_encoder=label_encoder,
        accuracy=accuracy,
        classification_report_text=report_text,
        feature_importance=feature_importance,
        feature_columns=feature_columns,
        model_path=str(model_path),
    )
