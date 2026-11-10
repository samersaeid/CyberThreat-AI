
from pathlib import Path
import sys

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
FEATURES_PATH = BASE_DIR / "models" / "feature_names.pkl"

print("=== CyberThreat AI: External Dataset Check ===")

if len(sys.argv) < 2:
    print("Usage: python check_external_data.py <csv_path>")
    sys.exit(1)

csv_path = Path(sys.argv[1])

if not csv_path.is_file():
    print("File not found:", csv_path)
    sys.exit(1)

features = joblib.load(FEATURES_PATH)

# Read only the header, without loading the full dataset
columns = pd.read_csv(csv_path, nrows=0).columns
columns = columns.str.strip().tolist()

missing = [f for f in features if f not in columns]
extra = [c for c in columns if c not in features]

print("Required features:", len(features))
print("Available columns:", len(columns))
print("Missing features:", len(missing))

if missing:
    print("\nMissing columns:")
    for name in missing:
        print("-", name)
else:
    print("\nAll required feature names are present.")

print("\nAdditional columns:", extra)

if "Label" in columns:
    print("Label column found.")
else:
    print("WARNING: Label column not found.")

print(
    "\nNote: Matching column names does not guarantee "
    "matching units or feature extraction methods."
)
