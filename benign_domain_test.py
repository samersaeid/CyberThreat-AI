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
    print("Domain", domain, "BENIGN samples:", len(df))

data = pd.concat(frames, ignore_index=True)

X = data[features]
y = data["Domain"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

prob = model.predict_proba(X_test)[:, 1]

print("\nBENIGN-only Domain ROC-AUC:",
      round(roc_auc_score(y_test, prob), 4))
