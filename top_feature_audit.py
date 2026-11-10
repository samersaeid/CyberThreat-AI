
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent

model = joblib.load(
    BASE / "models/cyberthreat_model.pkl"
)

features = list(joblib.load(
    BASE / "models/feature_names.pkl"
))

train_path = BASE / "data/cleaned_network_data.csv"
test_path = BASE / "data/external/ddos_2018_prepared.csv"

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

top_features = importance.head(15).copy()
columns = top_features["Feature"].tolist()

def get_ddos_medians(path):
    df = pd.read_csv(
        path,
        usecols=columns + ["Label"],
        low_memory=False
    )

    df["Label"] = df["Label"].astype(str).str.strip()
    df = df[df["Label"] == "DDoS"]

    numeric = df[columns].apply(
        pd.to_numeric, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)

    return numeric.median()

median_2017 = get_ddos_medians(train_path)
median_2018 = get_ddos_medians(test_path)

top_features["DDoS Median 2017"] = (
    top_features["Feature"].map(median_2017)
)

top_features["DDoS Median 2018"] = (
    top_features["Feature"].map(median_2018)
)

print("\n=== TOP 15 FEATURE AUDIT ===")
print(top_features.to_string(index=False))

output = BASE / "reports/top_feature_audit.csv"
output.parent.mkdir(parents=True, exist_ok=True)
top_features.to_csv(output, index=False)

print("\nSaved:", output)
