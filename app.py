
from pathlib import Path
import io

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==========================================
# CyberThreat AI v3 - SOC Dashboard
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "cyberthreat_model.pkl"
FEATURES_PATH = BASE_DIR / "models" / "feature_names.pkl"
IMPORTANCE_PATH = BASE_DIR / "reports" / "feature_importance.csv"

st.set_page_config(
    page_title="CyberThreat AI | SOC Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# Dark Theme
# ==========================================

st.markdown("""
<style>
.stApp {
    background: #0b1220;
    color: #f8fafc;
}

h1, h2, h3 {
    color: #f8fafc !important;
}

[data-testid="stMetric"] {
    background: #172238;
    border: 1px solid #30425f;
    border-radius: 12px;
    padding: 18px;
}

[data-testid="stMetricLabel"] {
    color: #b8c6dc !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * {
    color: #ffffff !important;
}

[data-testid="stSidebar"] {
    background: #111c2e;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# Load Model
# ==========================================

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, features


# ==========================================
# Experimental Severity Calculation
# ==========================================

def calculate_severity(results):
    results = results.copy()
    results["Threat_Severity"] = "None"

    attack_mask = results["AI_Prediction"] == "DDoS"

    duration = pd.to_numeric(
        results.get(
            "Flow Duration",
            pd.Series(0, index=results.index)
        ),
        errors="coerce"
    ).fillna(0)

    packet_rate = pd.to_numeric(
        results.get(
            "Flow Packets/s",
            pd.Series(0, index=results.index)
        ),
        errors="coerce"
    ).replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    duration_seconds = duration / 1_000_000

    score = (
        (packet_rate >= 100).astype(int)
        + (packet_rate >= 1000).astype(int)
        + (duration_seconds >= 1).astype(int)
    )

    results.loc[
        attack_mask, "Threat_Severity"
    ] = "Low"

    results.loc[
        attack_mask & (score >= 1),
        "Threat_Severity"
    ] = "Medium"

    results.loc[
        attack_mask & (score >= 2),
        "Threat_Severity"
    ] = "High"

    return results


# ==========================================
# Sidebar
# ==========================================

with st.sidebar:
    st.title("🛡️ CyberThreat AI")
    st.caption("Security Operations Center")

    st.divider()

    st.success("AI Detection Engine Ready")

    st.markdown("### Model Information")
    st.write("**Algorithm:** Random Forest")
    st.write("**Dataset:** CIC-IDS2017")
    st.write("**Classification:** BENIGN / DDoS")

    st.divider()

    st.warning(
        "Research prototype. Severity scores "
        "are experimental and not validated."
    )


# ==========================================
# Main Header
# ==========================================

st.title("🛡️ CyberThreat AI")
st.caption("AI-Powered SOC Analytics & Threat Detection")

st.divider()

st.subheader("📂 Network Traffic Upload")

uploaded_file = st.file_uploader(
    "Upload a CIC-IDS2017 compatible CSV file",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Upload a CSV file to begin analysis.")
    st.stop()

if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
    st.error("Trained model files are missing.")
    st.stop()


# ==========================================
# Load Data and Predict
# ==========================================

try:
    model, features = load_model()

    with st.spinner("Reading network data..."):
        df = pd.read_csv(
            uploaded_file,
            low_memory=False
        )

        df.columns = df.columns.str.strip()

    if df.empty:
        st.error("The uploaded CSV file is empty.")
        st.stop()

    missing = [
        feature for feature in features
        if feature not in df.columns
    ]

    if missing:
        st.error("Missing required model features.")
        st.write(missing)
        st.stop()

    X = df[features].copy()

    for feature in features:
        X[feature] = pd.to_numeric(
            X[feature],
            errors="coerce"
        )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    valid_rows = X.notna().all(axis=1)

    if not valid_rows.any():
        st.error("No valid network flows found.")
        st.stop()

    skipped = int((~valid_rows).sum())

    if skipped:
        st.warning(
            f"Skipped {skipped:,} invalid network flows."
        )

    X_valid = X.loc[valid_rows]

    with st.spinner("Running AI detection..."):
        predictions = model.predict(X_valid)

    results = df.loc[valid_rows].copy()

    results["AI_Prediction"] = np.where(
        predictions == 1,
        "DDoS",
        "BENIGN"
    )

    results = calculate_severity(results)

    # ======================================
    # Detection Overview
    # ======================================

    total = len(results)

    attacks = int(
        (results["AI_Prediction"] == "DDoS").sum()
    )

    benign = total - attacks

    attack_rate = (
        attacks / total * 100
        if total > 0 else 0
    )

    st.divider()
    st.subheader("📊 Detection Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Analyzed Flows", f"{total:,}")
    c2.metric("Predicted DDoS", f"{attacks:,}")
    c3.metric("Predicted BENIGN", f"{benign:,}")
    c4.metric("DDoS Rate", f"{attack_rate:.2f}%")

    # ======================================
    # Traffic Classification
    # ======================================

    left, right = st.columns(2)

    with left:
        st.subheader("Traffic Classification")

        traffic_counts = pd.DataFrame({
            "Type": ["BENIGN", "DDoS"],
            "Flows": [benign, attacks]
        }).set_index("Type")

        st.bar_chart(
            traffic_counts,
            color="#38bdf8"
        )

    # ======================================
    # Threat Severity
    # ======================================

    with right:
        st.subheader("Threat Severity")

        severity_counts = (
            results.loc[
                results["AI_Prediction"] == "DDoS",
                "Threat_Severity"
            ]
            .value_counts()
            .reindex(
                ["High", "Medium", "Low"],
                fill_value=0
            )
        )

        st.bar_chart(
            severity_counts,
            color="#f97316"
        )

        st.caption(
            "Experimental severity rules based "
            "on packet rate and flow duration."
        )

    # ======================================
    # SOC Analytics
    # ======================================

    st.divider()
    st.subheader("🔍 SOC Analytics")

    tab1, tab2 = st.tabs([
        "Destination Ports",
        "Flow Duration"
    ])

    with tab1:
        if "Destination Port" in results.columns:

            ddos_rows = results[
                results["AI_Prediction"] == "DDoS"
            ]

            top_ports = (
                ddos_rows["Destination Port"]
                .value_counts()
                .head(10)
            )

            if not top_ports.empty:
                top_ports.index = (
                    top_ports.index.astype(str)
                )

                st.bar_chart(top_ports)
            else:
                st.info("No predicted DDoS flows.")

    with tab2:
        if "Flow Duration" in results.columns:

            duration_data = pd.to_numeric(
                results["Flow Duration"],
                errors="coerce"
            ).dropna()

            if not duration_data.empty:
                st.write("Flow Duration Summary")

                st.dataframe(
                    duration_data.describe().to_frame(
                        name="Duration (microseconds)"
                    ),
                    width="stretch"
                )

    # ======================================
    # Explainable AI
    # ======================================

    st.divider()
    st.subheader("🧠 Explainable AI")

    if IMPORTANCE_PATH.exists():

        importance_df = pd.read_csv(
            IMPORTANCE_PATH
        )

        top_features = importance_df.nlargest(
            10,
            "Importance"
        )

        st.write("Top 10 Important Network Features")

        st.bar_chart(
            top_features.set_index("Feature")[
                "Importance"
            ],
            horizontal=True,
            color="#22c55e"
        )

        with st.expander(
            "View Feature Importance Data"
        ):
            st.dataframe(
                top_features,
                hide_index=True,
                width="stretch"
            )

        st.caption(
            "Global Random Forest feature importance. "
            "This does not explain individual predictions."
        )

    else:
        st.warning(
            "Feature importance report not found. "
            "Run model_insights.py first."
        )

    # ======================================
    # AI Detection Results and Filters
    # ======================================

    st.divider()
    st.subheader("🧪 AI Detection Results")

    traffic_filter = st.selectbox(
        "Traffic Filter",
        ["All", "DDoS", "BENIGN"]
    )

    severity_filter = st.selectbox(
        "Severity Filter",
        ["All", "High", "Medium", "Low"]
    )

    # IMPORTANT: Define filtered before using it
    filtered = results.copy()

    if traffic_filter != "All":
        filtered = filtered[
            filtered["AI_Prediction"] == traffic_filter
        ]

    if severity_filter != "All":
        filtered = filtered[
            filtered["Threat_Severity"] == severity_filter
        ]

    preferred_columns = [
        "Destination Port",
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Flow Packets/s",
        "Label",
        "AI_Prediction",
        "Threat_Severity"
    ]

    visible_columns = [
        col for col in preferred_columns
        if col in filtered.columns
    ]

    st.write(
        f"Matching network flows: {len(filtered):,}"
    )

    st.dataframe(
        filtered[visible_columns].head(1000),
        width="stretch",
        hide_index=True
    )

    st.caption(
        "Showing up to 1,000 matching rows."
    )

    # ======================================
    # Download Security Report
    # ======================================

    output = io.StringIO()

    filtered.to_csv(
        output,
        index=False
    )

    st.download_button(
        label="📥 Download Security Report",
        data=output.getvalue(),
        file_name="cyberthreat_v3_report.csv",
        mime="text/csv"
    )

    # ======================================
    # Footer
    # ======================================

    st.divider()

    st.caption(
        "CyberThreat AI v3 | AI-Powered SOC Dashboard"
    )

    st.caption(
        "Research prototype. Predictions are not "
        "verified ground truth. Independent model "
        "validation is required."
    )

except Exception as error:
    st.error("Analysis failed.")
    st.exception(error)
