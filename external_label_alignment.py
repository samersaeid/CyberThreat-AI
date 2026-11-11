import pandas as pd

raw = pd.read_csv(
    "data/external/ddos_2018.csv",
    usecols=["Label"]
)

prepared = pd.read_csv(
    "data/external/ddos_2018_prepared.csv",
    usecols=["Label"]
)

raw_labels = raw["Label"].astype(str).str.strip()
prepared_labels = prepared["Label"].astype(str).str.strip()

mapped = raw_labels.str.upper().map(
    lambda x: "BENIGN" if x == "BENIGN" else "DDoS"
)

print("Raw rows:", len(raw))
print("Prepared rows:", len(prepared))

if len(raw) == len(prepared):
    matches = mapped.eq(prepared_labels)
    print("Matching labels:", int(matches.sum()))
    print("Mismatched labels:", int((~matches).sum()))
    print("Raw attack types:")
    print(raw_labels.value_counts().to_string())
else:
    print("Row counts differ. Cannot align by row position.")
