
from pathlib import Path

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

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "cleaned_network_data.csv"
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

print("=== CyberThreat AI: Feature Ablation Test ===")

# Load and clean data
df = pd.read_csv(DATA_PATH, low_memory=False)
df.columns = df.columns.str.strip()

df = df.drop_duplicates().copy()
df["Label"] = df["Label"].str.strip()
df = df[df["Label"].isin(["BENIGN", "DDoS"])]

y = df["Label"].map({"BENIGN": 0, "DDoS": 1})

X = df.drop(columns=["Label"])
X = X.select_dtypes(include=[np.number])
X = X.replace([np.inf, -np.inf], np.nan)

valid = X.notna().all(axis=1)
X = X.loc[valid]
y = y.loc[valid]

# Same split for both experiments
train_idx, test_idx = train_test_split(
    np.arange(len(X)),
    test_size=0.2,
    random_state=42,
    stratify=y
)

removed_features = [
    "Destination Port",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward"
]

experiments = {
    "All Features": X,
    "Without Selected Features": X.drop(
        columns=removed_features,
        errors="ignore"
    )
}

results = []

for name, features in experiments.items():
    print(f"\nTraining: {name}")
    print("Feature count:", features.shape[1])

    X_train = features.iloc[train_idx]
    X_test = features.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "Experiment": name,
        "Features": features.shape[1],
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1": f1_score(y_test, predictions)
    }

    results.append(metrics)

    print("Accuracy:", round(metrics["Accuracy"], 6))
    print("Precision:", round(metrics["Precision"], 6))
    print("Recall:", round(metrics["Recall"], 6))
    print("F1:", round(metrics["F1"], 6))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

report = pd.DataFrame(results)

report.to_csv(
    REPORT_DIR / "ablation_results.csv",
    index=False
)

print("\n=== Comparison ===")
print(report.to_string(index=False))

print("\nSaved: reports/ablation_results.csv")
print(
    "Note: Both experiments use a random split "
    "of the same capture dataset."
)
