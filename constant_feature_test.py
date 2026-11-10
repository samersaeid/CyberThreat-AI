
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

DATA_PATH = Path("data/cleaned_network_data.csv")

print("=== CyberThreat AI: Constant Feature Test ===")

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

train_idx, test_idx = train_test_split(
    np.arange(len(X)),
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

# Detect constants from TRAINING data only
constant_columns = [
    col for col in X_train.columns
    if X_train[col].nunique(dropna=False) <= 1
]

print("Constant features:", len(constant_columns))
print(constant_columns)

experiments = {
    "All Features": X.columns.tolist(),
    "Without Constants": [
        col for col in X.columns
        if col not in constant_columns
    ]
}

for name, columns in experiments.items():
    print(f"\nTraining: {name}")
    print("Feature count:", len(columns))

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train[columns], y_train)

    predictions = model.predict(X_test[columns])

    print(
        "Accuracy:",
        round(accuracy_score(y_test, predictions), 6)
    )

    print(
        "Precision:",
        round(precision_score(y_test, predictions), 6)
    )

    print(
        "Recall:",
        round(recall_score(y_test, predictions), 6)
    )

    print(
        "F1:",
        round(f1_score(y_test, predictions), 6)
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

print("\nTest completed.")
