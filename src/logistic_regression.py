"""Logistic Regression model training and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

from src.model_metadata import build_bundle_metadata
from src.preprocess import preprocess_training_data
from src.split import split_train_test
from src.utils import TARGET_COLUMN, resolve_model_path

LOGISTIC_REGRESSION_MODEL_NAME = "logistic_regression.pkl"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42


@dataclass(frozen=True)
class LogisticRegressionResult:
    """Training result bundle for Logistic Regression."""

    model: Any
    label_encoder: LabelEncoder
    accuracy: float
    classification_report_text: str
    feature_columns: list[str]
    model_path: str


def _prepare_training_split(file_name: str) -> tuple[pd.DataFrame, pd.Series, list[str], Any]:
    """Load the cleaned training data and split it into features and target."""

    preprocessed = preprocess_training_data(file_name)
    frame = preprocessed.frame
    feature_columns = preprocessed.symptom_columns
    features = frame[feature_columns]
    target = frame[TARGET_COLUMN]
    return features, target, feature_columns, preprocessed


def _make_estimator(random_state: int = DEFAULT_RANDOM_STATE) -> LogisticRegression:
    return LogisticRegression(
        max_iter=1500,
        random_state=random_state,
        class_weight="balanced",
    )


def train_logistic_regression(
    file_name: str = "Training.csv",
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> LogisticRegressionResult:
    """Train a Logistic Regression classifier and persist the artifact.

    The serving model is fit on the full dataset; the reported accuracy is
    measured with an independently fitted, leak-aware fold model.
    """

    features, target, feature_columns, preprocessed = _prepare_training_split(file_name)
    label_encoder = LabelEncoder()
    encoded_target = label_encoder.fit_transform(target)

    x_train, x_test, y_train, y_test = split_train_test(
        features,
        encoded_target,
        test_size=test_size,
        random_state=random_state,
    )

    serving_model = _make_estimator(random_state)
    serving_model.fit(features, encoded_target)

    eval_model = _make_estimator(random_state)
    eval_model.fit(x_train, y_train)

    predictions = eval_model.predict(x_test)
    accuracy = float(accuracy_score(y_test, predictions))

    # Ensure classification_report receives matching labels and target_names for
    # the subset of classes present in this holdout fold.
    import numpy as _np

    present_labels = _np.unique(_np.concatenate([_np.asarray(y_test), _np.asarray(predictions)]))
    try:
        present_names = label_encoder.inverse_transform(present_labels)
        report_text = classification_report(
            y_test,
            predictions,
            labels=present_labels.tolist(),
            target_names=present_names.tolist(),
            zero_division=0,
        )
    except Exception:
        report_text = classification_report(y_test, predictions, zero_division=0)

    model_path = resolve_model_path(LOGISTIC_REGRESSION_MODEL_NAME)
    joblib.dump(
        {
            "model": serving_model,
            "label_encoder": label_encoder,
            "feature_columns": feature_columns,
            "accuracy": accuracy,
            "classification_report": report_text,
            "metadata": build_bundle_metadata(
                preprocessed.frame,
                feature_columns,
                file_name,
                test_size=test_size,
                random_state=random_state,
            ),
        },
        model_path,
    )

    return LogisticRegressionResult(
        model=serving_model,
        label_encoder=label_encoder,
        accuracy=accuracy,
        classification_report_text=report_text,
        feature_columns=feature_columns,
        model_path=str(model_path),
    )