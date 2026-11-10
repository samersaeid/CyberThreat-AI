
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

BASE = Path(__file__).resolve().parent

DATA = BASE / "data/cleaned_network_data.csv"
MODEL_OUT = BASE / "models/robust_model.pkl"
FEATURES_OUT = BASE / "models/robust_features.pkl"

print("=== CyberThreat AI: Robust Model Experiment ===")

df = pd.read_csv(DATA, low_memory=False)
df.columns = df.columns.str.strip()

df = df.drop_duplicates()

y = df["Label"].map({
    "BENIGN": 0,
    "DDoS": 1
})

if y.isna().any():
    raise ValueError("Unexpected labels found.")

X = df.drop(columns=["Label"])
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)

valid = X.notna().all(axis=1)
X = X.loc[valid]
y = y.loc[valid].astype(int)

# Remove constant features
constant_features = [
    col for col in X.columns
    if X[col].nunique() <= 1
]

X = X.drop(columns=constant_features)

# Remove verified duplicate feature
X = X.drop(
    columns=["Fwd Header Length.1"],
    errors="ignore"
)

# Experimental exclusion of environment-sensitive
# features; this is a hypothesis, not a proven fix.
excluded_features = [
    "Destination Port",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward"
]

X = X.drop(
    columns=excluded_features,
    errors="ignore"
)

print("Records:", len(X))
print("Features:", X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

print("\nTraining experimental model...")

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\n=== Internal Evaluation ===")
print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))
print("F1:", f1_score(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_OUT)
joblib.dump(list(X.columns), FEATURES_OUT)

print("\nSaved model:", MODEL_OUT)
print("Saved features:", FEATURES_OUT)
print("\nOriginal model remains unchanged.")
