import pandas as pd

features = [
    "Fwd Packet Length Max",
    "Fwd Packet Length Mean",
    "Flow Duration"
]

datasets = [
    ("2017", "data/cleaned_network_data.csv"),
    ("2018", "data/external/ddos_2018_prepared.csv")
]

for year, path in datasets:
    df = pd.read_csv(path, usecols=features + ["Label"])

    print(f"\n========== {year} ==========")

    for label, group in df.groupby("Label"):
        print(f"\nLabel: {label}")
        print("Records:", len(group))

        for feature in features:
            counts = group[feature].value_counts()
            top_value = counts.index[0]
            top_count = counts.iloc[0]

            print(
                f"{feature}: "
                f"Most common={top_value}, "
                f"Count={top_count}, "
                f"Percentage={top_count / len(group) * 100:.2f}%"
            )
