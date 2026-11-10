
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

# ==========================================
# CyberThreat AI - External Validation
# Train: CIC-IDS2017
# Test:  CSE-CIC-IDS2018
# ==========================================

BASE = Path(__file__).resolve().parent

DATA_PATH = BASE / "data/external/ddos_2018_prepared.csv"
MODEL_PATH = BASE / "models/cyberthreat_model.pkl"
FEATURES_PATH = BASE / "models/feature_names.pkl"
REPORT_PATH = BASE / "reports/external_validation_results.csv"

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

print("=" * 55)
print("CyberThreat AI - External Validation")
print("=" * 55)

# 1. Load trained model
print("\nLoading trained model...")

model = joblib.load(MODEL_PATH)
features = list(joblib.load(FEATURES_PATH))

print("Model loaded.")
print("Expected features:", len(features))

# 2. Check external data
if not DATA_PATH.exists():
    raise FileNotFoundError(DATA_PATH)

header = pd.read_csv(DATA_PATH, nrows=0)
header.columns = header.columns.str.strip()

missing = [
    feature for feature in features
    if feature not in header.columns
]

if missing:
    raise ValueError(f"Missing features: {missing}")

if "Label" not in header.columns:
    raise ValueError("Label column missing.")

# 3. Evaluate in chunks to limit memory use
print("\nEvaluating external data...")

CHUNK_SIZE = 50000

all_true = []
all_pred = []

for chunk_number, chunk in enumerate(
    pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE),
    start=1
):
    chunk.columns = chunk.columns.str.strip()

    X = chunk[features].apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.replace([np.inf, -np.inf], np.nan)

    y = chunk["Label"].astype(str).str.strip().map({
        "BENIGN": 0,
        "DDoS": 1
    })

    valid = X.notna().all(axis=1) & y.notna()

    X = X.loc[valid]
    y = y.loc[valid].astype(int)

    if X.empty:
        print(f"Chunk {chunk_number}: no valid rows")
        continue

    predictions = model.predict(X)

    all_true.append(y.to_numpy())
    all_pred.append(predictions)

    print(
        f"Chunk {chunk_number}: "
        f"{len(X):,} records evaluated"
    )

if not all_true:
    raise RuntimeError("No valid records evaluated.")

y_true = np.concatenate(all_true)
y_pred = np.concatenate(all_pred)

# 4. Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(
    y_true, y_pred, zero_division=0
)
recall = recall_score(
    y_true, y_pred, zero_division=0
)
f1 = f1_score(
    y_true, y_pred, zero_division=0
)

cm = confusion_matrix(
    y_true, y_pred, labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

print("\n" + "=" * 55)
print("EXTERNAL VALIDATION RESULTS")
print("=" * 55)

print("Total evaluated:", f"{len(y_true):,}")
print("Accuracy:", round(accuracy, 6))
print("Precision:", round(precision, 6))
print("Recall:", round(recall, 6))
print("F1-score:", round(f1, 6))

print("\nConfusion Matrix:")
print(cm)

print("\nTrue Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["BENIGN", "DDoS"],
        zero_division=0
    )
)

# 5. Save summary
results = pd.DataFrame([{
    "Dataset": "CSE-CIC-IDS2018",
    "Total Records": len(y_true),
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "TN": tn,
    "FP": fp,
    "FN": fn,
    "TP": tp
}])

results.to_csv(REPORT_PATH, index=False)

print("\nReport saved:")
print(REPORT_PATH)

print("\nIMPORTANT:")
print("No retraining was performed.")
print("Feature extraction compatibility needs review.")
print("External validation completed.")
