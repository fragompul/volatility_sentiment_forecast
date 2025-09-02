# File: dashboard/pages/04_errors.py
# Errors visualization for LSTM and TFT models

from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = ROOT / "predictions"

st.title("⚠️ Prediction Errors")

asset = st.selectbox("Select asset", ["AAPL", "MSFT", "IBEX35", "S&P500", "NASDAQ"], index=0)

@st.cache_data
def load_errors(asset, model="lstm"):
    fname = f"errors_{model}_{asset}.csv"
    path = PRED_DIR / fname
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=[0])
    df.rename(columns={df.columns[0]: "date"}, inplace=True)
    return df

lstm_err = load_errors(asset, "lstm")
tft_err = load_errors(asset, "tft")

if lstm_err.empty and tft_err.empty:
    st.warning("No error files available in models/")
else:
    if not lstm_err.empty:
        st.subheader(f"LSTM errors — {asset}")
        fig1 = px.line(lstm_err, x="date", y="error", title=f"LSTM Prediction Error ({asset})")
        st.plotly_chart(fig1, use_container_width=True)

    if not tft_err.empty:
        st.subheader(f"TFT errors — {asset}")
        fig2 = px.line(tft_err, x="date", y="error", title=f"TFT Prediction Error ({asset})")
        st.plotly_chart(fig2, use_container_width=True)
