from pathlib import Path
import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
MODEL = joblib.load(BASE / "models/cyberthreat_model.pkl")
FEATURES = list(joblib.load(BASE / "models/feature_names.pkl"))

DATA = BASE / "data/external/ddos_2018_prepared.csv"

results = []

for chunk in pd.read_csv(DATA, chunksize=50000):
    chunk.columns = chunk.columns.str.strip()

    X = chunk[FEATURES].apply(
        pd.to_numeric, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)

    valid = X.notna().all(axis=1)
    X = X.loc[valid]

    if X.empty:
        continue

    probabilities = MODEL.predict_proba(X)
    attack_index = list(MODEL.classes_).index(1)
    attack_probabilities = probabilities[:, attack_index]

    labels = chunk.loc[valid, "Label"].astype(str).str.strip()

    for label in ["BENIGN", "DDoS"]:
        values = attack_probabilities[labels.to_numpy() == label]

        if len(values):
            results.append(pd.DataFrame({
                "Label": label,
                "Probability": values
            }))

report = pd.concat(results, ignore_index=True)

print("\n=== MODEL PROBABILITY DIAGNOSIS ===")

for label, group in report.groupby("Label"):
    p = group["Probability"]

    print("\nClass:", label)
    print("Records:", len(p))
    print("Mean probability:", round(p.mean(), 6))
    print("Median probability:", round(p.median(), 6))
    print("Maximum probability:", round(p.max(), 6))
    print("Predicted DDoS at 0.5:", int((p >= 0.5).sum()))
    print("Predicted DDoS at 0.3:", int((p >= 0.3).sum()))
    print("Predicted DDoS at 0.1:", int((p >= 0.1).sum()))

output = BASE / "reports/probability_diagnosis.csv"

report.groupby("Label")["Probability"].agg(
    ["count", "mean", "median", "max"]
).to_csv(output)

print("\nSaved:", output)
