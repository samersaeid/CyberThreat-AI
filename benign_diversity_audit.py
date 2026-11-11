import pandas as pd

path = "data/external/ddos_2018_prepared.csv"

df = pd.read_csv(path, low_memory=False)
benign = df[df["Label"] == "BENIGN"].copy()

features = [
    "Fwd Packet Length Max",
    "Fwd Packet Length Mean",
    "Average Packet Size",
    "Flow Duration",
    "Flow Bytes/s"
]

print("=== 2018 BENIGN DIVERSITY AUDIT ===")
print("Total BENIGN:", len(benign))
print("Unique full rows:", len(benign.drop_duplicates()))
print("Unique feature combinations:", len(benign[features].drop_duplicates()))

print("\nUnique values per feature:")
print(benign[features].nunique().to_string())

print("\nMost common feature combinations:")
print(
    benign[features]
    .value_counts()
    .head(5)
    .to_string()
)
