# File: dashboard/pages/08_model_comparison.py
# Compares metrics across models & assets using metrics_all_summary.csv
# Expected header: asset,RMSE_LSTM,MAE_LSTM,R2_LSTM,MAPE (%)_LSTM,MBE_LSTM,RMSE_TFT,MAE_TFT,R2_TFT,MAPE (%)_TFT,MBE_TFT

from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = ROOT / "predictions"
MODELS_DIR = ROOT / "models"

st.title("🧾 Model comparison (summary metrics)")

@st.cache_data
def load_metrics():
    candidates = [PRED_DIR / "metrics_all_summary.csv", MODELS_DIR / "metrics_all_summary.csv", MODELS_DIR / "metrics_all_summary.csv"]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            # Normalize column names
            df.columns = [c.strip() for c in df.columns]
            return df
    return pd.DataFrame()

metrics = load_metrics()
if metrics.empty:
    st.warning("metrics_all_summary.csv not found in predictions/ or models/. If you have separate LSTM/TFT metrics, place them in predictions/metrics_all_summary.csv or in models/.")
else:
    st.subheader("Raw metrics table")
    st.dataframe(metrics)

    # Melt RMSEs to compare easily
    # look for RMSE_LSTM and RMSE_TFT (case-insensitive)
    cols = [c for c in metrics.columns if c.upper().startswith("RMSE")]
    # attempt to identify LSTM vs TFT by suffix
    rmse_cols = {}
    for c in cols:
        label = c.replace("RMSE_","").replace("RMSE ","").upper()
        rmse_cols[label] = c
    # If both present, produce a grouped bar chart
    if len(rmse_cols) >= 1:
        # build long table
        long = []
        for _, row in metrics.iterrows():
            asset = row.get('asset') if 'asset' in metrics.columns else row.iloc[0]
            for label, colname in rmse_cols.items():
                val = row.get(colname, np.nan)
                long.append({'asset': asset, 'model': label, 'RMSE': val})
        long_df = pd.DataFrame(long)
        if not long_df.empty:
            st.subheader("RMSE by asset & model")
            fig = px.bar(long_df, x='asset', y='RMSE', color='model', barmode='group', title="RMSE comparison")
            st.plotly_chart(fig, use_container_width=True)

    # Allow user to pick asset and see metrics breakdown
    if 'asset' in metrics.columns:
        asset_choice = st.selectbox("Select asset to inspect", sorted(metrics['asset'].unique()))
        row = metrics[metrics['asset'] == asset_choice].iloc[0]
        st.subheader(f"Metrics for {asset_choice}")
        st.table(row.to_frame(name='value'))

    # ranking by RMSE (if both models present)
    if 'RMSE_LSTM' in metrics.columns and 'RMSE_TFT' in metrics.columns:
        st.subheader("Which model wins per asset (RMSE)")
        def winner(r):
            if pd.isna(r['RMSE_LSTM']) and pd.isna(r['RMSE_TFT']):
                return "N/A"
            if pd.isna(r['RMSE_LSTM']):
                return "TFT"
            if pd.isna(r['RMSE_TFT']):
                return "LSTM"
            return "LSTM" if r['RMSE_LSTM'] < r['RMSE_TFT'] else "TFT"
        metrics['best_model_rmse'] = metrics.apply(winner, axis=1)
        st.table(metrics[['asset','RMSE_LSTM','RMSE_TFT','best_model_rmse']])
