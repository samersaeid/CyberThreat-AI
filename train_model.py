
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Paths
DATA_PATH = Path("data/cleaned_network_data.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

print("=== CyberThreat AI: Model Training ===")

# Load dataset
df = pd.read_csv(DATA_PATH, low_memory=False)
df.columns = df.columns.str.strip()

# Remove duplicate rows
before = len(df)
df = df.drop_duplicates().copy()

print("Duplicates removed:", before - len(df))

# Binary labels: BENIGN = 0, DDoS = 1
df["Label"] = df["Label"].str.strip()
df = df[df["Label"].isin(["BENIGN", "DDoS"])]

y = df["Label"].map({"BENIGN": 0, "DDoS": 1})

# Keep numeric features only
X = df.drop(columns=["Label"])
X = X.select_dtypes(include=[np.number])

# Clean remaining invalid values
X = X.replace([np.inf, -np.inf], np.nan)
valid = X.notna().all(axis=1)

X = X.loc[valid]
y = y.loc[valid]

print("Samples:", len(X))
print("Features:", X.shape[1])

# Split data BEFORE training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

print("\nTraining model...")
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, predictions)

print("\n=== Model Evaluation ===")
print(f"Accuracy: {accuracy:.4f}")

print(
    classification_report(
        y_test,
        predictions,
        target_names=["BENIGN", "DDoS"],
        zero_division=0
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Save model
joblib.dump(
    model,
    MODEL_DIR / "cyberthreat_model.pkl"
)

# Save feature names
joblib.dump(
    X.columns.tolist(),
    MODEL_DIR / "feature_names.pkl"
)

print("\nModel saved successfully!")
