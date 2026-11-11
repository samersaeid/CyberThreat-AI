import json
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import joblib

OUT = Path("results/v2_1")
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = "data/cleaned_network_data.csv"
EXTERNAL_PATH = "data/external/ddos_2018_prepared.csv"

excluded = [
    "Fwd Packet Length Max",
    "Fwd Packet Length Mean",
    "Flow Duration",
    "Flow Bytes/s",
    "Fwd IAT Std",
    "Average Packet Size"
]

train = pd.read_csv(TRAIN_PATH)
external = pd.read_csv(EXTERNAL_PATH)

train.columns = train.columns.str.strip()
external.columns = external.columns.str.strip()

train = train[train["Label"].isin(["BENIGN", "DDoS"])].copy()
external = external[external["Label"].isin(["BENIGN", "DDoS"])].copy()

common = [
    c for c in train.columns
    if c != "Label"
    and c in external.columns
    and c not in excluded
    and pd.api.types.is_numeric_dtype(train[c])
    and pd.api.types.is_numeric_dtype(external[c])
]

print("Common features:", len(common))
print("Excluded features:", excluded)

if not common:
    raise ValueError("No common numeric features found.")

train = train.replace([np.inf, -np.inf], np.nan)
external = external.replace([np.inf, -np.inf], np.nan)

train = train.dropna(subset=common)
external = external.dropna(subset=common)

# Remove exact duplicates within each dataset.
train = train.drop_duplicates(subset=common + ["Label"])
external = external.drop_duplicates(subset=common + ["Label"])

X = train[common]
y = (train["Label"] == "DDoS").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

def evaluate(name, X_eval, y_eval):
    predictions = model.predict(X_eval)

    result = {
        "accuracy": float(accuracy_score(y_eval, predictions)),
        "precision": float(precision_score(y_eval, predictions, zero_division=0)),
        "recall": float(recall_score(y_eval, predictions, zero_division=0)),
        "f1": float(f1_score(y_eval, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(
            y_eval, predictions, labels=[0, 1]
        ).tolist()
    }

    print(f"\n=== {name} ===")
    for key, value in result.items():
        print(f"{key}: {value}")

    return result

internal = evaluate("2017 INTERNAL", X_test, y_test)

external_result = evaluate(
    "2018 EXTERNAL",
    external[common],
    (external["Label"] == "DDoS").astype(int)
)

joblib.dump(
    {"model": model, "features": common},
    OUT / "ablation_model.joblib"
)

with open(OUT / "ablation_report.json", "w") as f:
    json.dump({
        "experiment": "v2.1 feature ablation",
        "excluded_features": excluded,
        "features_used": common,
        "internal": internal,
        "external": external_result
    }, f, indent=2)

print("\nSaved results to:", OUT)
