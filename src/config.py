"""Project-wide configuration constants for the Telco churn pipeline."""

NUM_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']

CAT_COLS = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod',
]

FEATURE_COLS = NUM_COLS + CAT_COLS
TARGET = 'Churn'

DATA_PATH = "data/telco_customer_churn.csv"
MODEL_PATH = "models/pipeline.pkl"
COLUMNS_PATH = "models/input_columns.pkl"
MLFLOW_EXPERIMENT = "telco-churn"
