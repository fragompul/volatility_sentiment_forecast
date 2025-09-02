# File: dashboard/pages/03_volatility_models.py
# Volatility model predictions (LSTM, TFT)

from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"

st.title("📉 Volatility Models")

asset = st.selectbox("Select asset", ["AAPL", "MSFT", "IBEX35", "S&P500", "NASDAQ"], index=0)

@st.cache_data
def load_predictions(asset, model="lstm"):
    fname = f"predictions_{model}_{asset}.csv"
    path = MODELS_DIR / fname
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=[0])
    # Normalize column names to lowercase
    df.columns = [c.lower() for c in df.columns]
    # Rename date column if needed
    if "date" not in df.columns:
        df.rename(columns={df.columns[0]: "date"}, inplace=True)
    # Standardize target/prediction columns
    if model == "lstm":
        for col in ["y_true", "y_pred"]:
            if col not in df.columns:
                # Try alternatives
                if col == "y_true" and "actual_volatility" in df.columns:
                    df.rename(columns={"actual_volatility": "y_true"}, inplace=True)
                if col == "y_pred" and "predicted_volatility" in df.columns:
                    df.rename(columns={"predicted_volatility": "y_pred"}, inplace=True)
    elif model == "tft":
        for col in ["y_true", "y_pred"]:
            if col not in df.columns:
                if col == "y_true" and "actual_volatility" in df.columns:
                    df.rename(columns={"actual_volatility": "y_true"}, inplace=True)
                if col == "y_pred" and "predicted_volatility" in df.columns:
                    df.rename(columns={"predicted_volatility": "y_pred"}, inplace=True)
    return df

lstm_df = load_predictions(asset, "lstm")
tft_df = load_predictions(asset, "tft")

if lstm_df.empty and tft_df.empty:
    st.warning("No predictions found in models/ for this asset.")
else:
    st.subheader(f"LSTM vs TFT predictions — {asset}")
    combined = pd.DataFrame()
    if not lstm_df.empty:
        combined = lstm_df[["date", "y_true", "y_pred"]].rename(columns={"y_pred": "LSTM_pred"})
    if not tft_df.empty:
        if combined.empty:
            combined = tft_df[["date", "y_true", "y_pred"]].rename(columns={"y_pred": "TFT_pred"})
        else:
            combined = combined.merge(
                tft_df[["date", "y_pred"]].rename(columns={"y_pred": "TFT_pred"}),
                on="date",
                how="inner"
            )
    if not combined.empty:
        plot_cols = [c for c in ["y_true", "LSTM_pred", "TFT_pred"] if c in combined.columns]
        if len(plot_cols) >= 2:
            fig = px.line(
                combined,
                x="date",
                y=plot_cols,
                labels={"value": "Volatility"},
                title=f"{asset} — True vs Predicted Volatility"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough prediction data to plot both LSTM and TFT.")

