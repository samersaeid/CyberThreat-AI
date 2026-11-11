from pathlib import Path
import pandas as pd
import joblib

base = Path(__file__).resolve().parent

model = joblib.load(base / "models/cyberthreat_model.pkl")
features = joblib.load(base / "models/feature_names.pkl")

importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

top_features = importance.head(10).index.tolist()

datasets = {
    "2017": base / "data/cleaned_network_data.csv",
    "2018": base / "data/external/ddos_2018_prepared.csv"
}

print("\n=== TOP FEATURE DISTRIBUTION AUDIT ===")

for year, path in datasets.items():
    df = pd.read_csv(
        path,
        usecols=top_features + ["Label"],
        low_memory=False
    )

    df = df[df["Label"].str.strip() == "DDoS"]

    print(f"\nDATASET: {year}")
    print("DDoS records:", len(df))

    for feature in top_features:
        values = pd.to_numeric(
            df[feature], errors="coerce"
        ).dropna()

        print(
            f"{feature}: "
            f"median={values.median():.4f}, "
            f"zero%={(values == 0).mean() * 100:.2f}, "
            f"unique={values.nunique()}"
        )
