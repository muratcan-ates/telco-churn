"""Pydantic v2 request and response schemas for the churn prediction API.

The ``CustomerInput`` schema mirrors the feature set declared in
:mod:`src.config`, and a module-level check guards against silent drift
between the two sources of truth.
"""

from pydantic import BaseModel, Field

from src.config import CAT_COLS, NUM_COLS


class CustomerInput(BaseModel):
    """Customer attributes accepted by the churn prediction endpoint."""

    tenure: float = Field(
        ge=0,
        description="Number of months the customer has stayed with the company.",
    )
    MonthlyCharges: float = Field(
        ge=0,
        description="The amount charged to the customer monthly.",
    )
    TotalCharges: float = Field(
        ge=0,
        description="The total amount charged to the customer.",
    )
    SeniorCitizen: int = Field(
        ge=0, le=1,
        description="Whether the customer is a senior citizen (1) or not (0).",
    )

    gender: str = Field(
        description="Customer gender ('Male' or 'Female').",
    )
    Partner: str = Field(
        description="Whether the customer has a partner ('Yes' or 'No').",
    )
    Dependents: str = Field(
        description="Whether the customer has dependents ('Yes' or 'No').",
    )
    PhoneService: str = Field(
        description="Whether the customer has phone service ('Yes' or 'No').",
    )
    MultipleLines: str = Field(
        description="Multiple lines status ('Yes', 'No', or 'No phone service').",
    )
    InternetService: str = Field(
        description="Internet service type ('DSL', 'Fiber optic', or 'No').",
    )
    OnlineSecurity: str = Field(
        description="Online security add-on ('Yes', 'No', or 'No internet service').",
    )
    OnlineBackup: str = Field(
        description="Online backup add-on ('Yes', 'No', or 'No internet service').",
    )
    DeviceProtection: str = Field(
        description="Device protection add-on ('Yes', 'No', or 'No internet service').",
    )
    TechSupport: str = Field(
        description="Tech support add-on ('Yes', 'No', or 'No internet service').",
    )
    StreamingTV: str = Field(
        description="Streaming TV add-on ('Yes', 'No', or 'No internet service').",
    )
    StreamingMovies: str = Field(
        description="Streaming movies add-on ('Yes', 'No', or 'No internet service').",
    )
    Contract: str = Field(
        description="Contract term ('Month-to-month', 'One year', or 'Two year').",
    )
    PaperlessBilling: str = Field(
        description="Whether the customer uses paperless billing ('Yes' or 'No').",
    )
    PaymentMethod: str = Field(
        description=(
            "Payment method ('Electronic check', 'Mailed check', "
            "'Bank transfer (automatic)', or 'Credit card (automatic)')."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "tenure": 12,
                "MonthlyCharges": 70.35,
                "TotalCharges": 845.50,
                "SeniorCitizen": 0,
                "gender": "Female",
                "Partner": "Yes",
                "Dependents": "No",
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
            }
        }
    }


class PredictionOutput(BaseModel):
    """Churn prediction response."""

    churn_probability: float = Field(
        ge=0, le=1,
        description="Probability of customer churning.",
    )
    churn_prediction: int = Field(
        ge=0, le=1,
        description="0 = will stay, 1 = will churn.",
    )
    top_factors: list[dict] = Field(
        description=(
            "Top SHAP-derived drivers of the prediction, each a dict with "
            "keys 'feature', 'impact', and 'direction'."
        ),
    )
    model_version: str = "1.0.0"


_EXPECTED_FIELDS = set(NUM_COLS + CAT_COLS)
_DECLARED_FIELDS = set(CustomerInput.model_fields.keys())
assert _DECLARED_FIELDS == _EXPECTED_FIELDS, (
    "CustomerInput fields drift from src.config: "
    f"missing={_EXPECTED_FIELDS - _DECLARED_FIELDS}, "
    f"extra={_DECLARED_FIELDS - _EXPECTED_FIELDS}"
)
