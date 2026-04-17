"""SHAP-based explainability utilities for single churn predictions.

The :class:`ChurnExplainer` loads the fitted pipeline, a background sample
drawn at training time, and the transformed feature names, then builds a
LinearExplainer backed by a :class:`shap.maskers.Independent` masker. Using a
proper background distribution is what keeps SHAP values from collapsing to
zero -- a single-row background would mask every contribution.
"""

from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import shap

from src.config import MODEL_PATH

BACKGROUND_PATH = "models/background_data.pkl"
FEATURE_NAMES_PATH = "models/feature_names.pkl"


class ChurnExplainer:
    """Wrap a fitted churn pipeline with a background-backed LinearExplainer."""

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        background_path: str = BACKGROUND_PATH,
        feature_names_path: str = FEATURE_NAMES_PATH,
    ) -> None:
        self.pipeline = joblib.load(model_path)
        self.background: pd.DataFrame = joblib.load(background_path)
        self.feature_names: list[str] = joblib.load(feature_names_path)

        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.classifier = self.pipeline.named_steps["classifier"]

        masker = shap.maskers.Independent(self.background, max_samples=100)
        self.explainer = shap.LinearExplainer(self.classifier, masker)
        self.expected_value = float(
            np.ravel(self.explainer.expected_value)[0]
        )

    def predict_and_explain(
        self,
        row: dict | pd.DataFrame,
        top_n: int = 5,
    ) -> dict:
        """Score a single customer and return the top SHAP drivers.

        Returns a dict with ``prediction``, ``probability``, ``expected_value``,
        and ``top_factors``. Factors with exactly zero impact are skipped.
        """
        frame = pd.DataFrame([row]) if isinstance(row, dict) else row

        proba = float(self.pipeline.predict_proba(frame)[0, 1])
        pred = int(proba >= 0.5)

        x_trans = self.preprocessor.transform(frame)
        if hasattr(x_trans, "toarray"):
            x_trans = x_trans.toarray()
        x_trans_df = pd.DataFrame(x_trans, columns=self.feature_names)

        explanation = self.explainer(x_trans_df)
        shap_vals = np.asarray(explanation.values).reshape(-1)
        feature_values = np.asarray(x_trans_df.iloc[0])

        order = np.argsort(np.abs(shap_vals))[::-1]

        top_factors: list[dict] = []
        for idx in order:
            impact = float(shap_vals[idx])
            if impact == 0.0:
                continue
            top_factors.append({
                "feature": _clean_name(self.feature_names[idx]),
                "value": float(feature_values[idx]),
                "impact": impact,
                "direction": (
                    "increases_churn" if impact > 0 else "retains"
                ),
            })
            if len(top_factors) >= top_n:
                break

        return {
            "prediction": pred,
            "probability": proba,
            "expected_value": self.expected_value,
            "top_factors": top_factors,
        }


def _clean_name(raw: str) -> str:
    """Strip the ColumnTransformer ``num__`` / ``cat__`` prefixes."""
    return raw.replace("num__", "").replace("cat__", "")


@lru_cache(maxsize=1)
def get_explainer() -> ChurnExplainer:
    """Return a process-wide singleton ChurnExplainer."""
    return ChurnExplainer()


def get_top_factors(
    pipeline,  # noqa: ARG001 - kept for backward-compat with the old signature
    X_single: pd.DataFrame,
    top_n: int = 5,
) -> list[dict]:
    """Return the top churn drivers in the legacy (absolute-impact) format.

    Preserved so ``api.main`` keeps working while we migrate callers to
    :class:`ChurnExplainer` directly.
    """
    result = get_explainer().predict_and_explain(X_single, top_n=top_n)
    return [
        {
            "feature": factor["feature"],
            "impact": abs(factor["impact"]),
            "direction": (
                "increases_churn" if factor["impact"] > 0
                else "decreases_churn"
            ),
        }
        for factor in result["top_factors"]
    ]
