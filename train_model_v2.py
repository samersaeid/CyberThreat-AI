from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data/cleaned_network_data.csv"
MODEL_DIR = BASE / "models/v2"
REPORT_DIR = BASE / "reports/v2"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=== CyberThreat AI v2.0 - Baseline Training ===")

df = pd.read_csv(DATA, low_memory=False)
df.columns = df.columns.str.strip()
df = df.drop_duplicates().copy()

df["Label"] = df["Label"].astype(str).str.strip()
df = df[df["Label"].isin(["BENIGN", "DDoS"])]

y = df["Label"].map({"BENIGN": 0, "DDoS": 1})
X = df.drop(columns=["Label"])
X = X.select_dtypes(include=[np.number])
X = X.replace([np.inf, -np.inf], np.nan)

valid = X.notna().all(axis=1)
X = X.loc[valid]
y = y.loc[valid]

if X.empty or y.nunique() != 2:
    raise RuntimeError("Insufficient valid data or classes.")

print("Valid records:", len(X))
print("Features:", X.shape[1])

# Internal development split: 60% train, 20% validation,
# 20% test. This is not an independent cross-dataset test.
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.20,
    stratify=y, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=0.25,
    stratify=y_dev, random_state=42
)

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

print("Training...")
model.fit(X_train, y_train)

def evaluate(name, x, labels):
    pred = model.predict(x)
    metrics = {
        "accuracy": float(accuracy_score(labels, pred)),
        "precision": float(precision_score(
            labels, pred, zero_division=0
        )),
        "recall": float(recall_score(
            labels, pred, zero_division=0
        )),
        "f1": float(f1_score(
            labels, pred, zero_division=0
        )),
        "confusion_matrix": confusion_matrix(
            labels, pred, labels=[0, 1]
        ).tolist()
    }
    print("\n", name, json.dumps(metrics, indent=2))
    return metrics

results = {
    "validation": evaluate("VALIDATION", X_val, y_val),
    "internal_test": evaluate("INTERNAL TEST", X_test, y_test)
}

joblib.dump(model, MODEL_DIR / "cyberthreat_model_v2.pkl")
joblib.dump(
    X.columns.tolist(),
    MODEL_DIR / "feature_names_v2.pkl"
)

with open(REPORT_DIR / "baseline_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nModel saved in:", MODEL_DIR)
print("Reports saved in:", REPORT_DIR)
print("Original model remains unchanged.")
