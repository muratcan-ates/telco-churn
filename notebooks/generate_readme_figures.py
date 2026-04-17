"""Generate README figures: SHAP summary + ROC curve + confusion matrix."""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import train_test_split

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

pipeline = joblib.load("models/pipeline.pkl")
background = joblib.load("models/background_data.pkl")
feature_names = joblib.load("models/feature_names.pkl")

df = pd.read_csv("data/telco_customer_churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
if df["Churn"].dtype == object:
    y = (df.pop("Churn") == "Yes").astype(int)
else:
    y = df.pop("Churn").astype(int)
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])
X = df

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42,
)

# ---------- Figure 1: ROC curve + confusion matrix --------------------------
y_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(
    fpr, tpr,
    label=f"Logistic Regression (AUC = {roc_auc:.3f})",
    linewidth=2,
)
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random classifier")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve — Test Set")
axes[0].legend(loc="lower right")
axes[0].grid(alpha=0.2)

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"]).plot(
    ax=axes[1], cmap="Blues", colorbar=False,
)
axes[1].set_title("Confusion Matrix @ threshold=0.5")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "model_performance.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"saved {FIGURES_DIR / 'model_performance.png'}")

# ---------- Figure 2: SHAP summary plot -------------------------------------
preprocessor = pipeline.named_steps["preprocessor"]
classifier = pipeline.named_steps["classifier"]

X_test_transformed = preprocessor.transform(X_test)
if hasattr(X_test_transformed, "toarray"):
    X_test_transformed = X_test_transformed.toarray()

rng = np.random.default_rng(42)
sample_size = min(500, X_test_transformed.shape[0])
sample_idx = rng.choice(X_test_transformed.shape[0], size=sample_size, replace=False)
X_test_sample = pd.DataFrame(
    X_test_transformed[sample_idx], columns=feature_names,
)

masker = shap.maskers.Independent(background, max_samples=100)
explainer = shap.LinearExplainer(classifier, masker)
explanation = explainer(X_test_sample)

plt.figure(figsize=(10, 8))
shap.summary_plot(
    explanation.values, X_test_sample,
    feature_names=feature_names, max_display=15, show=False,
)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"saved {FIGURES_DIR / 'shap_summary.png'}")

print("All figures generated.")
