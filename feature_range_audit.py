import pandas as pd
import numpy as np

features = [
    "Fwd Packet Length Max",
    "Fwd Packet Length Mean",
    "Flow Duration",
    "Flow Bytes/s",
    "Fwd IAT Std",
    "Average Packet Size"
]

datasets = {
    "2017": "data/cleaned_network_data.csv",
    "2018": "data/external/ddos_2018_prepared.csv"
}

for year, path in datasets.items():
    df = pd.read_csv(path, usecols=features + ["Label"])
    df = df[df["Label"].str.strip().str.upper() == "BENIGN"]

    print(f"\n=== {year} BENIGN ===")
    print("Records:", len(df))

    for feature in features:
        values = pd.to_numeric(df[feature], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan).dropna()

        print(
            f"{feature}: "
            f"min={values.min():.2f}, "
            f"median={values.median():.2f}, "
            f"p95={values.quantile(0.95):.2f}, "
            f"max={values.max():.2f}"
        )
