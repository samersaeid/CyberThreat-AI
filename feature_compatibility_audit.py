from pathlib import Path
import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
TRAIN = BASE / "data/cleaned_network_data.csv"
EXTERNAL = BASE / "data/external/ddos_2018_prepared.csv"
FEATURES = BASE / "models/feature_names.pkl"
OUTPUT = BASE / "reports/feature_compatibility_audit.csv"

features = list(joblib.load(FEATURES))

def load_sample(path, sample_size=100000):
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    missing = set(features) - set(df.columns)
    if missing:
        raise ValueError(f"Missing features: {missing}")

    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)

    return df

train = load_sample(TRAIN)
external = load_sample(EXTERNAL)

results = []

for feature in features:
    a = pd.to_numeric(train[feature], errors="coerce")
    b = pd.to_numeric(external[feature], errors="coerce")

    a = a.replace([np.inf, -np.inf], np.nan).dropna()
    b = b.replace([np.inf, -np.inf], np.nan).dropna()

    if a.empty or b.empty:
        continue

    train_median = float(a.median())
    external_median = float(b.median())

    results.append({
        "Feature": feature,
        "Train Median": train_median,
        "External Median": external_median,
        "Train Zero %": round(float((a == 0).mean() * 100), 2),
        "External Zero %": round(float((b == 0).mean() * 100), 2),
        "Median Difference": abs(train_median - external_median),
        "Zero Rate Difference": round(
            abs(float((a == 0).mean() - (b == 0).mean()) * 100), 2
        )
    })

report = pd.DataFrame(results)
report = report.sort_values(
    "Zero Rate Difference", ascending=False
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
report.to_csv(OUTPUT, index=False)

print("\n=== FEATURE COMPATIBILITY AUDIT ===")
print("Features checked:", len(report))
print("\nTop 15 zero-rate differences:")
print(report.head(15).to_string(index=False))
print("\nReport saved:", OUTPUT)
