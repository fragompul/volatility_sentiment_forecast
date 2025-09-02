# File: dashboard/pages/01_overview.py
# Market overview: prices, returns, volatility, sample predictions

from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

st.set_page_config(layout="wide")
st.title("📊 Market Overview")

asset = st.selectbox("Select asset", ["AAPL", "MSFT", "IBEX35", "S&P500", "NASDAQ"], index=0)

@st.cache_data
def load_market(asset):
    fname = f"{asset}_market.csv"
    path = DATA_DIR / fname
    if path.exists():
        df = pd.read_csv(path, parse_dates=[0])
        # Normalize column names to lowercase
        df.columns = [c.lower() for c in df.columns]
        if "date" not in df.columns:
            df.rename(columns={df.columns[0]: "date"}, inplace=True)
        return df.sort_values("date").reset_index(drop=True)
    return pd.DataFrame()

market_df = load_market(asset)

if market_df.empty:
    st.warning("No market CSV found for this asset in data/processed.")
else:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(f"Historical price — {asset}")
        fig = px.line(market_df, x="date", y="close", title=f"{asset} — Closing Price")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Quick stats")
        st.metric("Last price", f"{market_df['close'].iloc[-1]:.2f}")
        st.metric(
            "30d historical volatility",
            f"{market_df['close'].pct_change().rolling(30).std().iloc[-1]:.4f}",
        )

    st.subheader("Sample predictions (if available)")
    pred_path = MODELS_DIR / f"predictions_{asset}.csv"
    if pred_path.exists():
        preds = pd.read_csv(pred_path, parse_dates=[0])
        # Normalize columns to lowercase
        preds.columns = [c.lower() for c in preds.columns]
        if "date" not in preds.columns:
            preds.rename(columns={preds.columns[0]: "date"}, inplace=True)
        preds = preds.sort_values("date").tail(50)
        # Take first numeric columns for plotting
        y_cols = [c for c in preds.columns if c != "date"][:2]
        fig2 = px.line(preds, x="date", y=y_cols, labels={"value": "Prediction"})
        st.plotly_chart(fig2, use_container_width=True)
