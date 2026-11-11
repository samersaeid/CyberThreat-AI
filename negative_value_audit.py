import pandas as pd

features = [
    "Flow Duration",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd Packet Length Max",
    "Fwd Packet Length Mean",
    "Average Packet Size"
]

for year, path in [
    ("2017", "data/cleaned_network_data.csv"),
    ("2018", "data/external/ddos_2018_prepared.csv")
]:
    df = pd.read_csv(path, usecols=features + ["Label"])

    print(f"\n=== {year} NEGATIVE VALUE AUDIT ===")

    for feature in features:
        negative = df[feature] < 0

        print(
            f"{feature}: "
            f"{negative.sum()} negative records "
            f"({negative.mean() * 100:.4f}%)"
        )

    invalid = (df[features] < 0).any(axis=1)

    print("\nTotal affected rows:", invalid.sum())
    print("\nAffected rows by label:")
    print(df.loc[invalid, "Label"].value_counts().to_string())
