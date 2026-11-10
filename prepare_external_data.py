
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ==========================================
# CyberThreat AI - External Data Preparation
# CIC-IDS2018 -> CIC-IDS2017 feature format
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_PATH = BASE_DIR / "data" / "external" / "ddos_2018.csv"

OUTPUT_PATH = (
    BASE_DIR / "data" / "external" / "ddos_2018_prepared.csv"
)

FEATURES_PATH = (
    BASE_DIR / "models" / "feature_names.pkl"
)


# ==========================================
# 1. Feature name mapping
# ==========================================

rename_map = {
    "Dst Port": "Destination Port",
    "Tot Fwd Pkts": "Total Fwd Packets",
    "Tot Bwd Pkts": "Total Backward Packets",
    "TotLen Fwd Pkts": "Total Length of Fwd Packets",
    "TotLen Bwd Pkts": "Total Length of Bwd Packets",

    "Fwd Pkt Len Max": "Fwd Packet Length Max",
    "Fwd Pkt Len Min": "Fwd Packet Length Min",
    "Fwd Pkt Len Mean": "Fwd Packet Length Mean",
    "Fwd Pkt Len Std": "Fwd Packet Length Std",

    "Bwd Pkt Len Max": "Bwd Packet Length Max",
    "Bwd Pkt Len Min": "Bwd Packet Length Min",
    "Bwd Pkt Len Mean": "Bwd Packet Length Mean",
    "Bwd Pkt Len Std": "Bwd Packet Length Std",

    "Flow Byts/s": "Flow Bytes/s",
    "Flow Pkts/s": "Flow Packets/s",

    "Fwd IAT Tot": "Fwd IAT Total",
    "Bwd IAT Tot": "Bwd IAT Total",

    "Fwd Header Len": "Fwd Header Length",
    "Bwd Header Len": "Bwd Header Length",

    "Fwd Pkts/s": "Fwd Packets/s",
    "Bwd Pkts/s": "Bwd Packets/s",

    "Pkt Len Min": "Min Packet Length",
    "Pkt Len Max": "Max Packet Length",
    "Pkt Len Mean": "Packet Length Mean",
    "Pkt Len Std": "Packet Length Std",
    "Pkt Len Var": "Packet Length Variance",

    "FIN Flag Cnt": "FIN Flag Count",
    "SYN Flag Cnt": "SYN Flag Count",
    "RST Flag Cnt": "RST Flag Count",
    "PSH Flag Cnt": "PSH Flag Count",
    "ACK Flag Cnt": "ACK Flag Count",
    "URG Flag Cnt": "URG Flag Count",
    "ECE Flag Cnt": "ECE Flag Count",

    "Pkt Size Avg": "Average Packet Size",

    "Fwd Seg Size Avg": "Avg Fwd Segment Size",
    "Bwd Seg Size Avg": "Avg Bwd Segment Size",

    "Fwd Byts/b Avg": "Fwd Avg Bytes/Bulk",
    "Fwd Pkts/b Avg": "Fwd Avg Packets/Bulk",
    "Fwd Blk Rate Avg": "Fwd Avg Bulk Rate",

    "Bwd Byts/b Avg": "Bwd Avg Bytes/Bulk",
    "Bwd Pkts/b Avg": "Bwd Avg Packets/Bulk",
    "Bwd Blk Rate Avg": "Bwd Avg Bulk Rate",

    "Subflow Fwd Pkts": "Subflow Fwd Packets",
    "Subflow Fwd Byts": "Subflow Fwd Bytes",
    "Subflow Bwd Pkts": "Subflow Bwd Packets",
    "Subflow Bwd Byts": "Subflow Bwd Bytes",

    "Init Fwd Win Byts": "Init_Win_bytes_forward",
    "Init Bwd Win Byts": "Init_Win_bytes_backward",

    "Fwd Act Data Pkts": "act_data_pkt_fwd",
    "Fwd Seg Size Min": "min_seg_size_forward",
}


# ==========================================
# 2. Check required files
# ==========================================

print("=" * 55)
print("CyberThreat AI - External Dataset Preparation")
print("=" * 55)

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"External dataset not found: {INPUT_PATH}"
    )

if not FEATURES_PATH.exists():
    raise FileNotFoundError(
        f"Feature list not found: {FEATURES_PATH}"
    )

features = list(joblib.load(FEATURES_PATH))

print("\nRequired model features:", len(features))


# ==========================================
# 3. Check external dataset columns
# ==========================================

header = pd.read_csv(INPUT_PATH, nrows=0)
header.columns = header.columns.str.strip()

mapped_columns = [
    rename_map.get(column, column)
    for column in header.columns
]

# Verified identical in all 225,711 CIC-IDS2017 rows.
# This maps the corresponding 2018 header-length feature.
if (
    "Fwd Header Length.1" in features
    and "Fwd Header Length" in mapped_columns
):
    mapped_columns.append("Fwd Header Length.1")
    print("Duplicate header-length feature mapped.")

missing_features = [
    feature
    for feature in features
    if feature not in mapped_columns
]

if missing_features:
    print("\nMissing features:")
    for feature in missing_features:
        print("-", feature)

    raise SystemExit(
        "Preparation stopped: unmatched features."
    )

if "Label" not in mapped_columns:
    raise SystemExit("Preparation stopped: Label missing.")

print("All required feature names matched.")


# ==========================================
# 4. Read external dataset
# ==========================================

print("\nLoading external dataset...")

df = pd.read_csv(INPUT_PATH, low_memory=False)

df.columns = df.columns.str.strip()
df = df.rename(columns=rename_map)

print("Original records:", len(df))


# ==========================================
# 5. Handle duplicate header-length column
# ==========================================

if "Fwd Header Length.1" in features:

    if "Fwd Header Length" not in df.columns:
        raise SystemExit(
            "Missing source header-length feature."
        )

    df["Fwd Header Length.1"] = (
        df["Fwd Header Length"]
    )

    print("Fwd Header Length.1 created successfully.")


# ==========================================
# 6. Inspect labels
# ==========================================

df["Label"] = df["Label"].astype(str).str.strip()

print("\nOriginal label distribution:")
print(df["Label"].value_counts().to_string())


# ==========================================
# 7. Select BENIGN and DDoS only
# ==========================================

label_mapping = {
    "Benign": "BENIGN",
    "BENIGN": "BENIGN",
    "DDOS attack-HOIC": "DDoS",
    "DDOS attack-LOIC-UDP": "DDoS",
    "DDoS attacks-LOIC-HTTP": "DDoS",
}

# Match labels without case sensitivity.
normalized_mapping = {
    key.casefold(): value
    for key, value in label_mapping.items()
}

normalized_labels = (
    df["Label"].str.casefold().map(normalized_mapping)
)

df = df.loc[normalized_labels.notna()].copy()

df["Label"] = normalized_labels.loc[df.index]

print("\nSelected records:", len(df))

print("\nSelected label distribution:")
print(df["Label"].value_counts().to_string())

if df["Label"].nunique() != 2:
    raise SystemExit(
        "Preparation stopped: both BENIGN and DDoS "
        "must be present for binary evaluation."
    )


# ==========================================
# 8. Convert features to numeric values
# ==========================================

print("\nConverting features to numeric values...")

X = df.loc[:, features].apply(
    pd.to_numeric,
    errors="coerce"
)

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

valid_rows = X.notna().all(axis=1)

removed_rows = int((~valid_rows).sum())

X = X.loc[valid_rows].copy()

y = df.loc[valid_rows, "Label"].copy()


# ==========================================
# 9. Save prepared dataset
# ==========================================

prepared = X.copy()
prepared["Label"] = y

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

prepared.to_csv(
    OUTPUT_PATH,
    index=False
)


# ==========================================
# 10. Final report
# ==========================================

print("\n" + "=" * 55)
print("PREPARATION RESULTS")
print("=" * 55)

print("Original records:", len(pd.read_csv(
    INPUT_PATH,
    usecols=["Label"]
)))

print("Prepared records:", len(prepared))
print("Invalid numeric records removed:", removed_rows)

print("Model features:", len(features))
print("Prepared columns:", len(prepared.columns))

print("\nFinal class distribution:")
print(prepared["Label"].value_counts().to_string())

print("\nOutput file:")
print(OUTPUT_PATH)

print("\nIMPORTANT:")
print(
    "Column-name matching is not proof of identical "
    "feature extraction across dataset versions."
)

print(
    "The original trained model was not modified."
)

print("\nExternal data preparation completed.")
