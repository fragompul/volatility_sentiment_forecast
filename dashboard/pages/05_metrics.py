# File: dashboard/pages/05_metrics.py
# Summary metrics: RMSE, MAE, R2, etc.

from pathlib import Path
import streamlit as st
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = ROOT / "predictions"

st.title("📊 Model Metrics")

@st.cache_data
def load_summary():
    path = PRED_DIR / "metrics_all_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

summary = load_summary()
if summary.empty:
    st.warning("metrics_all_summary.csv not found in models/")
else:
    st.dataframe(summary)

    st.subheader("Best model by asset (based on RMSE)")
    best_models = summary.assign(
        Best_Model=summary.apply(lambda row: "LSTM" if row["RMSE_LSTM"] < row["RMSE_TFT"] else "TFT", axis=1)
    )[["asset", "Best_Model"]]
    st.table(best_models)
