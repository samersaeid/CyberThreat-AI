import pandas as pd
import numpy as np
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
    df = pd.read_csv(path, usecols=features)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.sample(n=min(50000, len(df)), random_state=42)
    df["Domain"] = domain
    frames.append(df)

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

print("Domain ROC-AUC:", round(roc_auc_score(y_test, prob), 4))
print("\nFeature importance:")
for name, value in sorted(
    zip(features, model.feature_importances_),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{name}: {value:.4f}")
