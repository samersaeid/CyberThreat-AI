
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent

DATASETS = {
    "2017": BASE / "data/cleaned_network_data.csv",
    "2018": BASE / "data/external/ddos_2018_prepared.csv"
}

FEATURES = [
    "Destination Port",
    "Flow Duration",
    "Fwd Packet Length Max",
    "Init_Win_bytes_forward",
    "Avg Fwd Segment Size",
    "Flow Packets/s"
]

REPORT = BASE / "reports/class_distribution_audit.csv"
REPORT.parent.mkdir(parents=True, exist_ok=True)

results = []

print("=== CyberThreat AI: Class Distribution Audit ===")

for year, path in DATASETS.items():
    print(f"\nProcessing {year}...")

    df = pd.read_csv(
        path,
        usecols=FEATURES + ["Label"],
        low_memory=False
    )

    df.columns = df.columns.str.strip()
    df["Label"] = df["Label"].astype(str).str.strip()

    for label in ["BENIGN", "DDoS"]:
        subset = df[df["Label"] == label]

        print(f"\n{year} - {label}")
        print("Records:", len(subset))

        if subset.empty:
            continue

        for feature in FEATURES:
            values = pd.to_numeric(
                subset[feature],
                errors="coerce"
            ).replace([np.inf, -np.inf], np.nan)

            median = values.median()
            q25 = values.quantile(0.25)
            q75 = values.quantile(0.75)

            results.append({
                "Year": year,
                "Class": label,
                "Feature": feature,
                "Count": values.notna().sum(),
                "Median": median,
                "Q25": q25,
                "Q75": q75
            })

            print(
                f"{feature}: "
                f"median={median:.2f}, "
                f"Q25={q25:.2f}, "
                f"Q75={q75:.2f}"
            )

report = pd.DataFrame(results)
report.to_csv(REPORT, index=False)

print("\nReport saved:", REPORT)
print("Audit completed.")
