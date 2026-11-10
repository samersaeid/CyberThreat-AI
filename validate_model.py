
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "cleaned_network_data.csv"
MODEL_PATH = BASE_DIR / "models" / "cyberthreat_model.pkl"
FEATURES_PATH = BASE_DIR / "models" / "feature_names.pkl"

print("=== CyberThreat AI: Validation Audit ===")

# Load model and data
model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)

df = pd.read_csv(DATA_PATH, low_memory=False)
df.columns = df.columns.str.strip()

print("Original records:", len(df))

# Match training preprocessing
df = df.drop_duplicates().copy()
df["Label"] = df["Label"].str.strip()
df = df[df["Label"].isin(["BENIGN", "DDoS"])]

X = df[features].copy()

for column in features:
    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )

X = X.replace([np.inf, -np.inf], np.nan)
valid = X.notna().all(axis=1)

X = X.loc[valid]
y = df.loc[valid, "Label"].map({
    "BENIGN": 0,
    "DDoS": 1
})

print("Valid records:", len(X))
print("Model features:", len(features))

# IMPORTANT:
# This evaluates the saved model on the full dataset.
# The dataset includes records used during training.
# Therefore these metrics are NOT independent test results.

predictions = model.predict(X)

print("\n=== Full-Dataset Diagnostic ===")
print("Accuracy:", round(accuracy_score(y, predictions), 6))
print("Precision:", round(precision_score(y, predictions), 6))
print("Recall:", round(recall_score(y, predictions), 6))
print("F1-score:", round(f1_score(y, predictions), 6))

print("\nClassification Report:")
print(
    classification_report(
        y,
        predictions,
        target_names=["BENIGN", "DDoS"],
        zero_division=0
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y, predictions))

print("\n=== Feature Audit ===")

suspicious_features = [
    "Destination Port",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward"
]

for feature in suspicious_features:
    if feature in features:
        print("Review feature:", feature)

print("\nWARNING:")
print("This is a diagnostic, not independent validation.")
print("The model has already seen training records.")
print("A separate holdout or external dataset is required.")
