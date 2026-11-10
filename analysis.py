
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_PATH = Path("data/network_data.csv")
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH, low_memory=False)
df.columns = df.columns.str.strip()

print("=== CyberThreat AI: Data Analysis ===")
print("Original Shape:", df.shape)

# Check data quality
print("Missing Values:", df.isna().sum().sum())
print("Duplicate Rows:", df.duplicated().sum())

# Replace infinite values
df = df.replace([np.inf, -np.inf], np.nan)

# Remove rows with missing values
df = df.dropna()

print("Cleaned Shape:", df.shape)

# Class distribution
counts = df["Label"].value_counts()
print("\nClass Distribution:")
print(counts)

# Plot 1: Traffic Distribution
plt.figure(figsize=(9, 5))
sns.countplot(data=df, x="Label", hue="Label",
              palette="Set2", legend=False)

plt.title("CyberThreat AI - Traffic Distribution")
plt.xlabel("Traffic Type")
plt.ylabel("Number of Network Flows")
plt.tight_layout()
plt.savefig(REPORTS / "traffic_distribution.png", dpi=200)
plt.close()

# Plot 2: Flow Duration
plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x="Label", y="Flow Duration",
            hue="Label", legend=False,
            showfliers=False)

plt.title("Flow Duration by Traffic Type")
plt.xlabel("Traffic Type")
plt.ylabel("Flow Duration")
plt.tight_layout()
plt.savefig(REPORTS / "flow_duration.png", dpi=200)
plt.close()

# Save cleaned data
df.to_csv("data/cleaned_network_data.csv", index=False)

print("\nAnalysis completed successfully!")
print("Charts saved in reports/")
print("Cleaned dataset saved in data/")
