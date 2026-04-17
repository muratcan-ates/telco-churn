"""Factories for the sklearn preprocessing and modelling pipeline."""

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CAT_COLS, NUM_COLS


def create_preprocessor() -> ColumnTransformer:
    """Build the ColumnTransformer that scales numeric and one-hot encodes categorical columns.

    Unseen categories at inference time are ignored (``handle_unknown='ignore'``)
    so the pipeline never raises on new category values.
    """
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUM_COLS),
            ('cat', OneHotEncoder(handle_unknown='ignore'), CAT_COLS),
        ]
    )


def create_pipeline(model: BaseEstimator) -> Pipeline:
    """Wrap the given classifier in the full preprocessing + classification pipeline."""
    return Pipeline(steps=[
        ('preprocessor', create_preprocessor()),
        ('classifier', model),
    ])
