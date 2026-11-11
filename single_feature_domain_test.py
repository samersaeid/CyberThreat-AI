import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

features = [
    "Fwd Packet Length Max",
    "Fwd Packet Length Mean",
    "Flow Duration",
    "Flow Bytes/s",
    "Fwd IAT Std",
    "Average Packet Size"
]

frames = []

for path, domain in [
    ("data/cleaned_network_data.csv", 0),
    ("data/external/ddos_2018_prepared.csv", 1)
]:
    df = pd.read_csv(path, usecols=features + ["Label"])
    df = df[df["Label"].str.strip().str.upper() == "BENIGN"]
    df = df[features].replace([np.inf, -np.inf], np.nan)
    df = df.dropna().drop_duplicates()
    df = df.sample(n=min(30000, len(df)), random_state=42)
    df["Domain"] = domain
    frames.append(df)

data = pd.concat(frames, ignore_index=True)

train, test = train_test_split(
    data,
    test_size=0.25,
    random_state=42,
    stratify=data["Domain"]
)

print("=== SINGLE FEATURE DOMAIN TEST ===")

for feature in features:
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(train[[feature]], train["Domain"])
    probabilities = model.predict_proba(test[[feature]])[:, 1]

    auc = roc_auc_score(test["Domain"], probabilities)

    print(f"{feature}: {auc:.4f}")
