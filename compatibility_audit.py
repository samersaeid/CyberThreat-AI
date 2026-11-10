
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent

DATASETS = {
    "2017": BASE / "data/cleaned_network_data.csv",
    "2018": BASE / "data/external/ddos_2018_prepared.csv"
}

FEATURES = [
    "Fwd Header Length",
    "Fwd Header Length.1",
    "Total Fwd Packets",
    "Total Length of Fwd Packets",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Init_Win_bytes_forward",
    "Fwd Packet Length Max",
    "Avg Fwd Segment Size"
]

print("=== CyberThreat AI: Compatibility Audit ===")

for year, path in DATASETS.items():

    print(f"\n{'=' * 50}")
    print(f"DATASET: {year}")
    print("=" * 50)

    df = pd.read_csv(
        path,
        usecols=FEATURES + ["Label"],
        low_memory=False
    )

    df["Label"] = df["Label"].astype(str).str.strip()

    for label in ["BENIGN", "DDoS"]:
        subset = df[df["Label"] == label]

        print(f"\nClass: {label}")
        print("Records:", len(subset))

        if subset.empty:
            continue

        for feature in FEATURES:
            values = pd.to_numeric(
                subset[feature], errors="coerce"
            ).replace([np.inf, -np.inf], np.nan)

            print(
                f"{feature}: "
                f"median={values.median():.2f}, "
                f"zero%={(values == 0).mean() * 100:.2f}, "
                f"negative%={(values < 0).mean() * 100:.2f}"
            )

        # Verify duplicate header length
        equal_headers = (
            subset["Fwd Header Length"]
            == subset["Fwd Header Length.1"]
        ).mean() * 100

        print(
            "Header duplicate equality:",
            f"{equal_headers:.2f}%"
        )

        # Check whether subflow values match totals.
        # Differences are diagnostic, not automatically errors.
        for a, b in [
            ("Subflow Fwd Packets", "Total Fwd Packets"),
            ("Subflow Fwd Bytes", "Total Length of Fwd Packets")
        ]:
            equal = (subset[a] == subset[b]).mean() * 100
            print(f"{a} == {b}: {equal:.2f}%")

print("\nCompatibility audit completed.")
