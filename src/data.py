"""Data loading and cleaning utilities for the Telco churn dataset."""

import pandas as pd

from src.config import DATA_PATH, FEATURE_COLS, TARGET


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and return a cleaned, modelling-ready DataFrame.

    Steps:
        1. Read the CSV from ``path``.
        2. Coerce ``TotalCharges`` to numeric (blank strings become NaN).
        3. Fill missing ``TotalCharges`` with the column median.
        4. Drop the ``customerID`` identifier column.
        5. Convert ``Churn`` from ``Yes``/``No`` to a binary 1/0 target.
    """
    df = pd.read_csv(path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    df = df.drop(columns=['customerID'])
    df[TARGET] = (df[TARGET] == 'Yes').astype(int)
    return df


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned DataFrame into the feature matrix ``X`` and target vector ``y``."""
    X = df[FEATURE_COLS]
    y = df[TARGET]
    return X, y
