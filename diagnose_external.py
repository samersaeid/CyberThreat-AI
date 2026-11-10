
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent

TRAIN_PATH = BASE / "data/cleaned_network_data.csv"
TEST_PATH = BASE / "data/external/ddos_2018_prepared.csv"
MODEL_PATH = BASE / "models/cyberthreat_model.pkl"
FEATURES_PATH = BASE / "models/feature_names.pkl"

print("=== CyberThreat AI: External Diagnosis ===")

model = joblib.load(MODEL_PATH)
features = list(joblib.load(FEATURES_PATH))

# Read a sample from both datasets.
# This is diagnostic sampling, not final evaluation.
train = pd.read_csv(TRAIN_PATH, nrows=10000)
test = pd.read_csv(TEST_PATH, nrows=10000)

train.columns = train.columns.str.strip()
test.columns = test.columns.str.strip()

important_features = [
    "Destination Port",
    "Flow Duration",
    "Fwd Packet Length Max",
    "Init_Win_bytes_forward",
    "Avg Fwd Segment Size",
    "Flow Packets/s"
]

print("\n=== Feature Distribution Comparison ===")

for feature in important_features:
    if feature not in features:
        continue

    a = pd.to_numeric(train[feature], errors="coerce")
    b = pd.to_numeric(test[feature], errors="coerce")

    print(f"\nFeature: {feature}")
    print("2017 median:", a.median())
    print("2018 median:", b.median())
    print("2017 min/max:", a.min(), a.max())
    print("2018 min/max:", b.min(), b.max())

print("\n=== Prediction Probability Diagnosis ===")

X = test[features].apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)

valid = X.notna().all(axis=1)
X = X.loc[valid]

probabilities = model.predict_proba(X)

ddos_index = list(model.classes_).index(1)
ddos_prob = probabilities[:, ddos_index]

print("Records checked:", len(X))
print("Minimum DDoS probability:", ddos_prob.min())
print("Maximum DDoS probability:", ddos_prob.max())
print("Average DDoS probability:", ddos_prob.mean())

print("\nProbability percentiles:")
for percentile in [0, 25, 50, 75, 90, 99, 100]:
    value = np.percentile(ddos_prob, percentile)
    print(f"{percentile}%: {value:.6f}")

print("\nPredicted DDoS:", int((ddos_prob >= 0.5).sum()))
print("Predicted BENIGN:", int((ddos_prob < 0.5).sum()))

print("\nDiagnosis completed.")
