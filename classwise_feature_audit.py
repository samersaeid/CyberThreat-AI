from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parent

paths = {
    "CIC-IDS2017": BASE / "data/cleaned_network_data.csv",
    "CSE-CIC-IDS2018": BASE / "data/external/ddos_2018_prepared.csv"
}

features = [
    "Flow Bytes/s",
    "Max Packet Length",
    "Fwd Packet Length Min",
    "Flow IAT Std",
    "RST Flag Count",
    "ECE Flag Count"
]

results = []

for dataset, path in paths.items():
    df = pd.read_csv(path, usecols=lambda c: c.strip() in features + ["Label"])
    df.columns = df.columns.str.strip()
    df["Label"] = df["Label"].astype(str).str.strip()

    for label in ["BENIGN", "DDoS"]:
        subset = df[df["Label"] == label]

        for feature in features:
            values = pd.to_numeric(
                subset[feature], errors="coerce"
            ).replace([np.inf, -np.inf], np.nan).dropna()

            results.append({
                "Dataset": dataset,
                "Label": label,
                "Feature": feature,
                "Records": len(values),
                "Median": values.median(),
                "Zero Percent": round(
                    (values == 0).mean() * 100, 2
                ) if len(values) else np.nan
            })

report = pd.DataFrame(results)

output = BASE / "reports/classwise_feature_audit.csv"
report.to_csv(output, index=False)

print("\n=== CLASSWISE FEATURE AUDIT ===")
print(report.to_string(index=False))
print("\nSaved:", output)
