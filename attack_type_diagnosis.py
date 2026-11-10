
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent

RAW = BASE / "data/external/ddos_2018.csv"
PREPARED = BASE / "data/external/ddos_2018_prepared.csv"
MODEL = BASE / "models/cyberthreat_model.pkl"
FEATURES = BASE / "models/feature_names.pkl"

model = joblib.load(MODEL)
features = list(joblib.load(FEATURES))

print("=== CyberThreat AI: Attack Type Diagnosis ===")

raw_labels = pd.read_csv(RAW, usecols=["Label"])
raw_labels["Label"] = raw_labels["Label"].astype(str).str.strip()

prepared = pd.read_csv(PREPARED)

# Preparation preserved every record in the same order.
if len(raw_labels) != len(prepared):
    raise ValueError("Row count mismatch.")

X = prepared[features]
probability_index = list(model.classes_).index(1)

results = []

for attack_type in raw_labels["Label"].unique():

    indices = raw_labels.index[
        raw_labels["Label"] == attack_type
    ]

    print(f"\nEvaluating: {attack_type}")

    count = 0
    detected = 0
    probability_sum = 0.0

    for start in range(0, len(indices), 50000):
        batch_indices = indices[start:start + 50000]
        batch = X.iloc[batch_indices]

        probabilities = model.predict_proba(
            batch
        )[:, probability_index]

        count += len(probabilities)
        detected += int((probabilities >= 0.5).sum())
        probability_sum += float(probabilities.sum())

    detection_rate = detected / count if count else 0

    results.append({
        "Attack Type": attack_type,
        "Records": count,
        "Predicted DDoS": detected,
        "Detection Rate": detection_rate,
        "Average DDoS Probability": (
            probability_sum / count if count else 0
        )
    })

report = pd.DataFrame(results)

print("\n=== RESULTS ===")
print(report.to_string(index=False))

output = BASE / "reports/attack_type_diagnosis.csv"
output.parent.mkdir(parents=True, exist_ok=True)
report.to_csv(output, index=False)

print("\nSaved:", output)
