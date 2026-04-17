"""FastAPI service for Telco churn prediction with SHAP explainability.

Exposes a JSON ``/predict`` endpoint backed by the fitted pipeline persisted
by ``training/train.py``.
"""

from __future__ import annotations

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import CustomerInput, PredictionOutput
from src.config import COLUMNS_PATH, MODEL_PATH
from src.explain import get_top_factors


def _load_artifacts() -> tuple[object, list[str]]:
    """Load the fitted pipeline and its expected column order from disk."""
    try:
        pipeline = joblib.load(MODEL_PATH)
        input_columns = joblib.load(COLUMNS_PATH)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Model artifacts not found at {MODEL_PATH} / {COLUMNS_PATH}. "
            "Run training/train.py first."
        ) from exc
    return pipeline, input_columns


pipeline, input_columns = _load_artifacts()


app = FastAPI(
    title="Telco Churn API",
    version="1.0.0",
    description="ML-powered churn prediction with SHAP explainability",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict:
    """Return a short index pointing to the API docs."""
    return {"message": "Telco Churn API", "docs": "/docs"}


@app.get("/health")
def health() -> dict:
    """Return a simple liveness probe confirming the model is loaded."""
    return {"status": "healthy", "model_loaded": True}


def _customer_to_frame(customer: CustomerInput) -> pd.DataFrame:
    """Convert a validated ``CustomerInput`` into a single-row DataFrame.

    The returned DataFrame's columns are ordered exactly as the fitted
    pipeline expects them.
    """
    return pd.DataFrame([customer.model_dump()])[input_columns]


@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput) -> PredictionOutput:
    """Score a single customer and return probability, label, and top drivers."""
    X = _customer_to_frame(customer)
    probability = float(pipeline.predict_proba(X)[0][1])
    prediction = int(pipeline.predict(X)[0])
    top_factors = get_top_factors(pipeline, X, top_n=5)
    return PredictionOutput(
        churn_probability=probability,
        churn_prediction=prediction,
        top_factors=top_factors,
    )


"""Run: uvicorn api.main:app --reload --port 8000"""
