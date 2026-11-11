import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score

bundle = joblib.load("results/v2_1/ablation_model.joblib")

model = bundle["model"]
features = bundle["features"]

df = pd.read_csv(
    "data/external/ddos_2018_prepared.csv",
    usecols=features + ["Label"]
)

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=features)

y = (df["Label"] == "DDoS").astype(int)

probs = model.predict_proba(df[features])[:, 1]

print("=== V2.1 EXTERNAL PROBABILITY AUDIT ===")
print("Records:", len(df))

for label, value in [("BENIGN", 0), ("DDoS", 1)]:
    p = probs[y.to_numpy() == value]

    print(f"\n{label}:")
    print("Mean:", round(float(np.mean(p)), 6))
    print("Median:", round(float(np.median(p)), 6))
    print("P95:", round(float(np.quantile(p, 0.95)), 6))
    print("Max:", round(float(np.max(p)), 6))

print("\nROC-AUC:", roc_auc_score(y, probs))
print("Average Precision:", average_precision_score(y, probs))

print("\n=== THRESHOLD DIAGNOSTICS ===")

for threshold in [0.5, 0.3, 0.1, 0.05, 0.01]:
    pred = probs >= threshold

    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())

    recall = tp / int((y == 1).sum())
    precision = tp / (tp + fp) if tp + fp else 0

    print(
        f"Threshold={threshold:.2f} "
        f"TP={tp} FP={fp} "
        f"Recall={recall:.4f} "
        f"Precision={precision:.4f}"
    )
