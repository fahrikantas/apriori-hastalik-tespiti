"""Decision Tree model training and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from src.preprocess import preprocess_training_data
from src.utils import TARGET_COLUMN, resolve_model_path

DECISION_TREE_MODEL_NAME = "decision_tree.pkl"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42


@dataclass(frozen=True)
class SupervisedModelResult:
    """Training artifacts produced by a supervised learning experiment."""

    model: Any
    label_encoder: LabelEncoder
    accuracy: float
    confusion_matrix: pd.DataFrame
    classification_report_text: str
    feature_columns: list[str]
    model_path: str


def _prepare_training_split(file_name: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load the cleaned training data and split it into features and target."""

    preprocessed = preprocess_training_data(file_name)
    frame = preprocessed.frame
    feature_columns = preprocessed.symptom_columns
    features = frame[feature_columns]
    target = frame[TARGET_COLUMN]
    return features, target, feature_columns


def train_decision_tree(
    file_name: str = "Training.csv",
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> SupervisedModelResult:
    """Train a Decision Tree classifier and persist the fitted artifact."""

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

    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = float(accuracy_score(y_test, predictions))
    report_text = classification_report(
        y_test,
        predictions,
        target_names=label_encoder.classes_,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, predictions)
    confusion_df = pd.DataFrame(
        cm,
        index=label_encoder.classes_,
        columns=label_encoder.classes_,
    )

    model_path = resolve_model_path(DECISION_TREE_MODEL_NAME)
    joblib.dump(
        {
            "model": model,
            "label_encoder": label_encoder,
            "feature_columns": feature_columns,
            "accuracy": accuracy,
            "classification_report": report_text,
            "confusion_matrix": confusion_df,
        },
        model_path,
    )

    return SupervisedModelResult(
        model=model,
        label_encoder=label_encoder,
        accuracy=accuracy,
        confusion_matrix=confusion_df,
        classification_report_text=report_text,
        feature_columns=feature_columns,
        model_path=str(model_path),
    )
