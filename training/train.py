"""Train and compare churn models, log to MLflow, and persist the best pipeline.

Run with:
    python -m training.train
"""

from __future__ import annotations

import joblib
import mlflow
import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.config import COLUMNS_PATH, MLFLOW_EXPERIMENT, MODEL_PATH
from src.data import get_X_y, load_data
from src.pipeline import create_pipeline

RANDOM_STATE = 42
TEST_SIZE = 0.2
BACKGROUND_SIZE = 100
BACKGROUND_PATH = "models/background_data.pkl"
FEATURE_NAMES_PATH = "models/feature_names.pkl"


def build_candidate_models() -> dict[str, BaseEstimator]:
    """Return the candidate classifiers keyed by their ``model_type`` tag."""
    return {
        'logistic_regression': LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE,
        ),
        'random_forest': RandomForestClassifier(random_state=RANDOM_STATE),
        'gradient_boosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def evaluate(pipeline, X_test, y_test) -> dict[str, float]:
    """Score ``pipeline`` on the held-out set and return f1, accuracy, auc_roc."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        'f1': f1_score(y_test, y_pred),
        'accuracy': accuracy_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_proba),
    }


def train_and_log(
    model_type: str,
    model: BaseEstimator,
    X_train, X_test, y_train, y_test,
):
    """Fit a pipeline for ``model``, log its metrics to MLflow, and return it.

    Returns the fitted pipeline and its evaluation metrics dict.
    """
    pipeline = create_pipeline(model)
    with mlflow.start_run(run_name=model_type):
        mlflow.set_tag('model_type', model_type)
        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test)
        mlflow.log_metrics(metrics)
    return pipeline, metrics


def pick_best_by_f1(results):
    """Return the ``(name, pipeline, metrics)`` entry with the highest F1 score."""
    return max(results, key=lambda entry: entry[2]['f1'])


def save_artifacts(pipeline, feature_columns: list[str]) -> None:
    """Persist the fitted pipeline and its expected input column order to disk."""
    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(feature_columns, COLUMNS_PATH)


def save_shap_background(pipeline, X_train: pd.DataFrame) -> None:
    """Persist a SHAP background sample and transformed feature names.

    LinearExplainer needs a representative background distribution to produce
    non-degenerate SHAP values; a single-row masker collapses every
    contribution to zero. We sample ``BACKGROUND_SIZE`` rows from the
    transformed training matrix and store them alongside the feature names.
    """
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']

    X_train_transformed = preprocessor.transform(X_train)
    if hasattr(X_train_transformed, 'toarray'):
        X_train_transformed = X_train_transformed.toarray()

    feature_names = list(preprocessor.get_feature_names_out())
    X_train_transformed_df = pd.DataFrame(
        X_train_transformed, columns=feature_names,
    )

    background = shap.utils.sample(
        X_train_transformed_df, BACKGROUND_SIZE, random_state=RANDOM_STATE,
    )

    joblib.dump(background, BACKGROUND_PATH)
    joblib.dump(feature_names, FEATURE_NAMES_PATH)

    expected_value = (
        classifier.intercept_[0]
        + np.dot(classifier.coef_[0], background.values.mean(axis=0))
    )
    print(
        f"SHAP background saved ({len(background)} rows, "
        f"{len(feature_names)} features); "
        f"expected_value (logit) = {expected_value:.6f}"
    )


def print_result_row(model_type: str, metrics: dict[str, float]) -> None:
    """Print one results table row in the ``Accuracy | F1 | AUC`` format."""
    print(
        f"{model_type:>20} | "
        f"Accuracy: {metrics['accuracy']:.4f} | "
        f"F1: {metrics['f1']:.4f} | "
        f"AUC: {metrics['auc_roc']:.4f}"
    )


def main() -> None:
    """Orchestrate loading, training, logging, selection, and persistence."""
    df = load_data()
    X, y = get_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    results = []
    for model_type, model in build_candidate_models().items():
        pipeline, metrics = train_and_log(
            model_type, model, X_train, X_test, y_train, y_test,
        )
        results.append((model_type, pipeline, metrics))
        print_result_row(model_type, metrics)

    best_name, best_pipeline, best_metrics = pick_best_by_f1(results)
    print(f"\nBest model: {best_name} (F1={best_metrics['f1']:.4f})")

    print("\nClassification report:")
    print(classification_report(y_test, best_pipeline.predict(X_test)))

    save_artifacts(best_pipeline, list(X.columns))
    print(f"Saved pipeline to {MODEL_PATH}")
    print(f"Saved input columns to {COLUMNS_PATH}")

    if best_name == 'logistic_regression':
        save_shap_background(best_pipeline, X_train)
        print(f"Saved SHAP background to {BACKGROUND_PATH}")
        print(f"Saved feature names to {FEATURE_NAMES_PATH}")


if __name__ == '__main__':
    main()
