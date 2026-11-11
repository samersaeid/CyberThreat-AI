import pandas as pd
import numpy as np
import joblib

bundle = joblib.load("results/v2_1/ablation_model.joblib")
model = bundle["model"]
features = bundle["features"]

raw_path = "data/external/ddos_2018.csv"
prepared_path = "data/external/ddos_2018_prepared.csv"

raw_header = pd.read_csv(raw_path, nrows=0)
prepared_header = pd.read_csv(prepared_path, nrows=0)

raw_columns = {
    c.strip(): c for c in raw_header.columns
}
prepared_columns = {
    c.strip(): c for c in prepared_header.columns
}

# Verify alignment using unchanged shared features.
candidates = [
    "Flow Duration",
    "Total Fwd Packets",
    "Fwd Packet Length Max",
    "Flow IAT Mean"
]

shared = [
    c for c in candidates
    if c in raw_columns and c in prepared_columns
]

if len(shared) < 2:
    raise RuntimeError(
        "Not enough shared columns to verify row alignment."
    )

raw_use = [raw_columns[c] for c in shared]
raw_use.append(raw_columns["Label"])

raw = pd.read_csv(raw_path, usecols=raw_use)
prepared = pd.read_csv(
    prepared_path,
    usecols=[prepared_columns[c] for c in shared]
)

raw.columns = raw.columns.str.strip()
prepared.columns = prepared.columns.str.strip()

if len(raw) != len(prepared):
    raise RuntimeError("Row counts do not match.")

print("=== ROW ALIGNMENT CHECK ===")

for feature in shared:
    a = pd.to_numeric(raw[feature], errors="coerce")
    b = pd.to_numeric(prepared[feature], errors="coerce")

    equal = np.isclose(
        a.to_numpy(),
        b.to_numpy(),
        rtol=1e-7,
        atol=1e-8,
        equal_nan=True
    )

    print(feature, "matches:", int(equal.sum()), "/", len(equal))

    if not equal.all():
        raise RuntimeError(
            f"Row alignment not verified for {feature}."
        )

print("\nRow alignment verified for selected features.")

# Load the same prepared records in their original order.
df = pd.read_csv(
    prepared_path,
    usecols=features
)

df = df.replace([np.inf, -np.inf], np.nan)

valid = df[features].notna().all(axis=1)

df = df.loc[valid]
attack_types = raw.loc[valid, "Label"].str.strip()

probs = model.predict_proba(df[features])[:, 1]

print("\n=== ATTACK TYPE ANALYSIS ===")

for attack_type in attack_types.unique():
    mask = attack_types.to_numpy() == attack_type
    scores = probs[mask]

    print(f"\n{attack_type}")
    print("Records:", len(scores))
    print("Mean score:", round(float(scores.mean()), 6))
    print("Max score:", round(float(scores.max()), 6))

    for threshold in [0.5, 0.01]:
        detected = int((scores >= threshold).sum())
        rate = detected / len(scores)

        metric = (
            "False positive rate"
            if attack_type.lower() == "benign"
            else "Recall"
        )

        print(
            f"Threshold {threshold}: "
            f"Detected={detected}, "
            f"{metric}={rate:.4%}"
        )
