from pathlib import Path
import io

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# CyberThreat AI | SOC Dashboard (UI refresh)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "cyberthreat_model.pkl"
FEATURES_PATH = BASE_DIR / "models" / "feature_names.pkl"
IMPORTANCE_PATH = BASE_DIR / "reports" / "feature_importance.csv"

st.set_page_config(page_title="CyberThreat AI | SOC Dashboard", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp, [data-testid="stAppViewContainer"] {background: #0b1220; color: #e8eef8;}
[data-testid="stHeader"] {background: rgba(11,18,32,.94);}
[data-testid="stSidebar"] {background: #111d30; border-right: 1px solid #263954;}
[data-testid="stSidebar"] * {color: #e2eaf7;}
[data-testid="stSidebar"] [data-testid="stAlert"] * {color: #142033 !important;}
h1,h2,h3,h4,p,span,label {color: inherit;}
h1,h2,h3 {letter-spacing: -.025em;}
[data-testid="stCaptionContainer"], .stCaption {color: #b8c7df !important;}
[data-testid="stMetric"] {background: #17263c; border: 1px solid #32465f; border-radius: 15px; padding: 18px;}
[data-testid="stMetricLabel"] * {color: #c6d3e7 !important;}
[data-testid="stMetricValue"] * {color: #f8fafc !important;}
[data-testid="stFileUploader"] {background: #142238; border: 1px solid #354a68; border-radius: 14px; padding: 14px;}
[data-testid="stFileUploader"] * {color: #e8eef8;}
[data-testid="stFileUploader"] button {background: #2563eb !important; color: white !important; border: none;}
[data-testid="stDataFrame"] {border: 1px solid #354a68; border-radius: 12px;}
.stTabs [data-baseweb="tab-list"] {gap: 10px;}
.stTabs [data-baseweb="tab"] {background: #182941; border-radius: 9px; padding: 8px 14px; color: #e5edf8;}
.stTabs [aria-selected="true"] {background: #2453a5 !important; color: white !important;}
div[data-testid="stAlert"] {border-radius: 12px;}
div.stButton > button[kind="primary"], div.stDownloadButton > button {background: #2563eb; border: 1px solid #3b82f6; color: white; border-radius: 10px;}
.ct-hero {padding: 20px 24px; background: linear-gradient(115deg,#172b48,#111c30); border:1px solid #345072; border-radius: 18px; margin-bottom: 18px;}
.ct-hero h1 {font-size: 2.15rem; margin: 0 0 4px 0; color: #f8fafc;}
.ct-hero p {margin: 0; color: #c5d4e9;}
.ct-kicker {font-size: .8rem; letter-spacing: .14em; font-weight: 700; color: #7dd3fc; margin-bottom: 8px;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH), joblib.load(FEATURES_PATH)


def calculate_severity(results):
    """Preserve original experimental severity rules."""
    results = results.copy()
    results["Threat_Severity"] = "None"
    attack_mask = results["AI_Prediction"] == "DDoS"
    duration = pd.to_numeric(results.get("Flow Duration", pd.Series(0, index=results.index)), errors="coerce").fillna(0)
    packet_rate = pd.to_numeric(results.get("Flow Packets/s", pd.Series(0, index=results.index)), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    duration_seconds = duration / 1_000_000
    score = ((packet_rate >= 100).astype(int) + (packet_rate >= 1000).astype(int) + (duration_seconds >= 1).astype(int))
    results.loc[attack_mask, "Threat_Severity"] = "Low"
    results.loc[attack_mask & (score >= 1), "Threat_Severity"] = "Medium"
    results.loc[attack_mask & (score >= 2), "Threat_Severity"] = "High"
    return results


with st.sidebar:
    st.title("🛡️ CyberThreat AI")
    st.caption("SOC Analytics • Research Edition")
    st.divider()
    if MODEL_PATH.is_file() and FEATURES_PATH.is_file():
        st.success("Model files available")
    else:
        st.error("Model files missing")
    st.subheader("Model Information")
    st.markdown("**Algorithm:** Random Forest")
    st.markdown("**Training dataset:** CIC-IDS2017")
    st.markdown("**Predicted classes:** BENIGN / DDoS")
    st.divider()
    st.warning("Research prototype only. Independent CSE-CIC-IDS2018 testing showed 0% DDoS recall; do not rely on predictions for security decisions.")
    st.caption("Severity categories are experimental, not validated risk scores.")

st.markdown('''<div class="ct-hero"><div class="ct-kicker">SECURITY ANALYTICS / RESEARCH PROTOTYPE</div><h1>🛡️ CyberThreat AI</h1><p>Network traffic classification, model insights and downloadable analysis reports</p></div>''', unsafe_allow_html=True)
st.subheader("📂 Upload Network Traffic")
st.caption("Select a CIC-IDS2017-compatible CSV containing the model's expected network-flow features. Maximum file size depends on Streamlit configuration.")
uploaded_file = st.file_uploader("Network traffic CSV", type=["csv"], label_visibility="collapsed")

if uploaded_file is None:
    st.info("Upload a CSV file to display detection statistics, traffic charts and model insights.")
    st.markdown(
        '<p style="text-align:center;color:#94a3b8;font-size:14px;margin-top:18px;">'
        'Developed by <strong>Samer Saed Aljohani</strong></p>',
        unsafe_allow_html=True,
    )
    st.stop()

if not MODEL_PATH.is_file() or not FEATURES_PATH.is_file():
    st.error("Trained model files are missing. Restore models/cyberthreat_model.pkl and models/feature_names.pkl before analysis.")
    st.stop()

try:
    model, features = load_model()
    features = list(features)
    if not features or len(features) != len(set(features)):
        st.error("Model feature list is empty or contains duplicates.")
        st.stop()

    with st.spinner("Reading network traffic..."):
        df = pd.read_csv(uploaded_file, low_memory=False)
        df.columns = df.columns.str.strip()

    if df.empty:
        st.error("The uploaded CSV contains no rows.")
        st.stop()
    if df.columns.duplicated().any():
        st.error("The CSV contains duplicate column names after whitespace cleanup.")
        st.stop()

    missing = [feature for feature in features if feature not in df.columns]
    if missing:
        st.error(f"Missing {len(missing)} required model feature(s).")
        with st.expander("View missing features"):
            st.write(missing)
        st.stop()

    X = df.loc[:, features].copy()
    for feature in features:
        X[feature] = pd.to_numeric(X[feature], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    valid_rows = X.notna().all(axis=1)
    skipped = int((~valid_rows).sum())
    if not valid_rows.any():
        st.error("No valid network flows found. Required feature values must be numeric and finite.")
        st.stop()
    if skipped:
        st.warning(f"Skipped {skipped:,} row(s) with invalid or missing feature values.")

    with st.spinner("Running experimental model predictions..."):
        predictions = model.predict(X.loc[valid_rows])

    results = df.loc[valid_rows].copy()
    # Original mapping retained: model output 1 -> DDoS; all other outputs -> BENIGN.
    results["AI_Prediction"] = np.where(predictions == 1, "DDoS", "BENIGN")
    results = calculate_severity(results)

    total = len(results)
    attacks = int((results["AI_Prediction"] == "DDoS").sum())
    benign = total - attacks
    attack_rate = attacks / total * 100 if total else 0

    st.divider()
    st.subheader("📊 Detection Overview")
    a, b, c, d = st.columns(4)
    a.metric("Analyzed Flows", f"{total:,}")
    b.metric("Predicted DDoS", f"{attacks:,}")
    c.metric("Predicted BENIGN", f"{benign:,}")
    d.metric("Predicted DDoS Rate", f"{attack_rate:.2f}%")
    st.caption(f"Uploaded: {len(df):,} flows • Analyzed: {total:,} • Skipped: {skipped:,}. Counts represent model predictions, not verified labels.")
    st.warning("External validation limitation: the tested model missed all DDoS records in CSE-CIC-IDS2018 (0% DDoS recall). A zero predicted-attack count does not mean traffic is safe.")

    overview_tab, analytics_tab, explain_tab, results_tab = st.tabs([
        "📈 Overview", "🔎 SOC Analytics", "🧠 Model Insights", "📋 Results & Export"
    ])

    with overview_tab:
        left, right = st.columns(2)
        with left:
            st.subheader("Predicted Traffic Classes")
            class_counts = pd.DataFrame({"Type": ["BENIGN", "DDoS"], "Flows": [benign, attacks]}).set_index("Type")
            st.bar_chart(class_counts, color="#38bdf8")
        with right:
            st.subheader("Experimental Severity")
            severity_counts = (results.loc[results["AI_Prediction"] == "DDoS", "Threat_Severity"]
                               .value_counts().reindex(["High", "Medium", "Low"], fill_value=0))
            st.bar_chart(severity_counts, color="#fb923c")
            st.caption("Severity is a heuristic based on packet rate and flow duration, not a calibrated security risk.")

    with analytics_tab:
        ports_tab, duration_tab = st.tabs(["Destination Ports", "Flow Duration"])
        with ports_tab:
            if "Destination Port" not in results.columns:
                st.info("Destination Port column is not present in this CSV.")
            else:
                ddos_rows = results.loc[results["AI_Prediction"] == "DDoS"]
                top_ports = ddos_rows["Destination Port"].value_counts().head(10)
                if top_ports.empty:
                    st.info("No DDoS predictions to chart. This does not establish that the traffic is benign.")
                else:
                    top_ports.index = top_ports.index.astype(str)
                    st.bar_chart(top_ports)
        with duration_tab:
            if "Flow Duration" not in results.columns:
                st.info("Flow Duration column is not present in this CSV.")
            else:
                duration_data = pd.to_numeric(results["Flow Duration"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                if duration_data.empty:
                    st.info("No valid duration values available.")
                else:
                    st.dataframe(duration_data.describe().to_frame(name="Duration (microseconds)"), use_container_width=True)

    with explain_tab:
        st.subheader("Global Random Forest Feature Importance")
        if not IMPORTANCE_PATH.exists():
            st.info("Feature importance report is unavailable. Run model_insights.py to generate it.")
        else:
            importance_df = pd.read_csv(IMPORTANCE_PATH)
            if not {"Feature", "Importance"}.issubset(importance_df.columns):
                st.warning("Feature importance report must contain Feature and Importance columns.")
            else:
                importance_df["Importance"] = pd.to_numeric(importance_df["Importance"], errors="coerce")
                top_features = importance_df.dropna(subset=["Importance"]).nlargest(10, "Importance")
                if top_features.empty:
                    st.info("No valid feature importance scores found.")
                else:
                    st.bar_chart(top_features.set_index("Feature")["Importance"], horizontal=True, color="#34d399")
                    with st.expander("View top feature scores"):
                        st.dataframe(top_features, hide_index=True, use_container_width=True)
            st.caption("Global feature importance is not a causal explanation or an explanation of any single prediction.")

    with results_tab:
        st.subheader("Prediction Results")
        f1, f2 = st.columns(2)
        with f1:
            traffic_filter = st.selectbox("Prediction", ["All", "DDoS", "BENIGN"])
        with f2:
            severity_filter = st.selectbox("Experimental severity", ["All", "High", "Medium", "Low", "None"])
        filtered = results.copy()
        if traffic_filter != "All":
            filtered = filtered.loc[filtered["AI_Prediction"] == traffic_filter]
        if severity_filter != "All":
            filtered = filtered.loc[filtered["Threat_Severity"] == severity_filter]
        preferred_columns = ["Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Flow Packets/s", "Label", "AI_Prediction", "Threat_Severity"]
        visible_columns = [column for column in preferred_columns if column in filtered.columns]
        st.caption(f"{len(filtered):,} matching flows • Preview limited to 1,000 rows • CSV export contains all matching rows")
        st.dataframe(filtered[visible_columns].head(1000), hide_index=True, use_container_width=True)
        report = io.StringIO()
        filtered.to_csv(report, index=False)
        st.download_button("📥 Download Filtered CSV Report", data=report.getvalue(), file_name="cyberthreat_soc_report.csv", mime="text/csv", type="primary")

    st.divider()
    st.caption("CyberThreat AI | SOC Dashboard • Research prototype • Predictions are not verified ground truth • Not suitable for operational security decisions")
    st.markdown(
        '<p style="text-align:center;color:#94a3b8;font-size:14px;margin-top:18px;">'
        'Developed by <strong>Samer Aljohani</strong></p>',
        unsafe_allow_html=True,
    )


except Exception as error:
    st.error("Analysis failed. Check the CSV format, model compatibility and application logs.")
    with st.expander("Technical error details"):
        st.exception(error)
