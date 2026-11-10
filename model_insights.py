
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

MODEL_PATH = Path("models/cyberthreat_model.pkl")
FEATURES_PATH = Path("models/feature_names.pkl")
REPORTS = Path("reports")

REPORTS.mkdir(exist_ok=True)

# Load trained model and feature names
model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)

# Calculate feature importance
importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("=== Top 10 Important Features ===")
print(importance.head(10).to_string(index=False))

# Save results
importance.to_csv(
    REPORTS / "feature_importance.csv",
    index=False
)

# Create chart
top_features = importance.head(10)

plt.figure(figsize=(11, 6))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.title("CyberThreat AI - Top 10 Features")
plt.xlabel("Random Forest Feature Importance")
plt.tight_layout()

plt.savefig(
    REPORTS / "feature_importance.png",
    dpi=250
)

plt.close()

print("\nFeature importance chart saved!")
