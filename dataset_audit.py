
import pandas as pd
from pathlib import Path

data_path = Path("data/cleaned_network_data.csv")

df = pd.read_csv(data_path, low_memory=False)
df.columns = df.columns.str.strip()

print("=== CyberThreat AI: Dataset Audit ===")

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")

print("\nClass distribution:")
print(df["Label"].value_counts())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nColumns with only one unique value:")
constant_columns = [
    col for col in df.columns
    if df[col].nunique(dropna=False) <= 1
]
print(constant_columns)

print("\nAudit completed.")
