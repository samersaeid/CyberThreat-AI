
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent

MODEL = BASE / "models/cyberthreat_model.pkl"
FEATURES = BASE / "models/feature_names.pkl"
DATA = BASE / "data/external/ddos_2018_prepared.csv"

model = joblib.load(MODEL)
features = list(joblib.load(FEATURES))

df = pd.read_csv(DATA, low_memory=False)
df.columns = df.columns.str.strip()

# Balanced random sample for diagnosis
benign = df[df["Label"] == "BENIGN"].sample(
    n=1000, random_state=42
)

ddos = df[df["Label"] == "DDoS"].sample(
    n=1000, random_state=42
)

sample = pd.concat(
    [benign, ddos],
    ignore_index=True
)

X = sample[features].copy()
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)

if X.isna().any().any():
    raise ValueError("Invalid feature values found.")

ddos_index = list(model.classes_).index(1)

def get_probabilities(data):
    return model.predict_proba(data)[:, ddos_index]

baseline = get_probabilities(X)

print("=== CyberThreat AI: Feature Shift Probe ===")
print("Sample records:", len(X))
print("Baseline average DDoS probability:", baseline.mean())
print("Baseline predicted DDoS:", (baseline >= 0.5).sum())

features_to_test = [
    "Init_Win_bytes_forward",
    "Flow Duration",
    "Fwd Packet Length Max",
    "Avg Fwd Segment Size",
    "Flow Packets/s",
    "Destination Port"
]

results = []

for feature in features_to_test:

    modified = X.copy()

    # Replace feature with its median from the
    # current sample, keeping other features unchanged.
    modified[feature] = X[feature].median()

    new_probabilities = get_probabilities(modified)

    change = np.abs(
        new_probabilities - baseline
    ).mean()

    results.append({
        "Feature": feature,
        "Mean Absolute Probability Change": change,
        "New Average Probability": new_probabilities.mean(),
        "Predicted DDoS": int(
            (new_probabilities >= 0.5).sum()
        )
    })

report = pd.DataFrame(results)
report = report.sort_values(
    "Mean Absolute Probability Change",
    ascending=False
)

print("\n=== Results ===")
print(report.to_string(index=False))

output = BASE / "reports/feature_shift_probe.csv"
output.parent.mkdir(parents=True, exist_ok=True)
report.to_csv(output, index=False)

print("\nSaved:", output)
