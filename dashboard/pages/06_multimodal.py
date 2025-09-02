# File: dashboard/pages/06_multimodal.py
# Multimodal predictions comparison (CatBoost / LightGBM / MLP / LSTM)
# Expects multimodal prediction files in PROJECT_ROOT/models/
# Typical file headers: Date,Ticker,Actual_Volatility,Predicted_Volatility

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import glob

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
PRED_DIR = ROOT / "predictions"
st.title("🔗 Multimodal Models (ensemble & per-model view)")

@st.cache_data
def discover_multimodal_files():
    patterns = [
        MODELS_DIR / "predictions_multimodal*.csv",
        MODELS_DIR / "predictions_*multimodal*.csv",
        PRED_DIR / "predictions_multimodal*.csv",
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(str(p)))
    # deduplicate
    files = sorted(list(dict.fromkeys(files)))
    return files

@st.cache_data
def load_multimodal_file(path_str):
    path = Path(path_str)
    try:
        df = pd.read_csv(path, parse_dates=[0])
    except Exception:
        df = pd.read_csv(path, parse_dates=[0], low_memory=False)
    # normalize columns
    df.columns = [c.strip() for c in df.columns]
    if df.columns[0].lower() not in ("date","datetime"):
        df = df.rename(columns={df.columns[0]: "Date"})
    # standard names (lowercase)
    df = df.rename(columns={c: c.lower() for c in df.columns})
    # map expected columns
    # possible names: date,ticker,actual_volatility,predicted_volatility
    if "date" not in df.columns and "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    if "ticker" not in df.columns and "Ticker" in df.columns:
        df = df.rename(columns={"Ticker": "ticker"})
    # unify actual & predicted column names
    for cand in ["actual_volatility","actual_vol","actual"]:
        if cand in df.columns and "actual" not in df.columns:
            df = df.rename(columns={cand: "actual"})
    for cand in ["predicted_volatility","predicted_vol","predicted","pred"]:
        if cand in df.columns and "predicted" not in df.columns:
            df = df.rename(columns={cand: "predicted"})
    # ensure date is datetime and sorted
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df

def compute_metrics(actual, pred):
    mask = (~actual.isna()) & (~pred.isna())
    if mask.sum() == 0:
        return dict(RMSE=np.nan, MAE=np.nan, MAPE=np.nan, R2=np.nan)
    a = actual[mask].astype(float)
    p = pred[mask].astype(float)
    mse = np.mean((a - p) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(a - p))
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((a - p) / a.replace(0, np.nan))) * 100
    # r2
    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum((a - a.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    return dict(RMSE=rmse, MAE=mae, MAPE=mape, R2=r2)

# discover available files
files = discover_multimodal_files()
if not files:
    st.warning("No multimodal prediction files found in models/ or predictions/. Expected filenames like:\n"
               "`predictions_multimodal_catboost.csv`, `predictions_multimodal_lgb.csv`, etc.")
else:
    human_names = [Path(f).name for f in files]
    chosen_file = st.selectbox("Choose model/predictions file (or pick 'Ensemble')", ["Ensemble"] + human_names)
    # load all into dict
    dfs = {Path(f).name: load_multimodal_file(f) for f in files}

    # assemble list of tickers available
    tickers = set()
    for df in dfs.values():
        if "ticker" in df.columns:
            tickers.update(df["ticker"].unique().astype(str))
    tickers = sorted(list(tickers))
    asset = st.selectbox("Select asset", tickers if tickers else ["AAPL"])
    st.write(f"Selected: **{asset}**")

    # prepare per-model metrics
    metrics_list = []
    for name, df in dfs.items():
        sub = df[df["ticker"].astype(str) == str(asset)] if "ticker" in df.columns else df
        if sub.empty:
            continue
        m = compute_metrics(sub.get("actual"), sub.get("predicted"))
        m_row = {"model_file": name, **m}
        metrics_list.append(m_row)
    metrics_df = pd.DataFrame(metrics_list).sort_values("RMSE")
    if not metrics_df.empty:
        st.subheader("Per-model metrics for selected asset")
        st.dataframe(metrics_df.reset_index(drop=True))
    else:
        st.info("No metrics to show (no rows for selected asset in the detected multimodal files).")

    # If user chose Ensemble, compute average predicted across available models (align by date & ticker)
    if chosen_file == "Ensemble":
        # inner join on date and ticker for all models that have ticker
        joined = None
        valid_models = []
        for name, df in dfs.items():
            df_sub = df[df["ticker"].astype(str) == str(asset)] if "ticker" in df.columns else df
            if df_sub.empty or "predicted" not in df_sub.columns:
                continue
            df_tmp = df_sub[['date','predicted']].rename(columns={'predicted': f'pred_{Path(name).stem}'})
            if joined is None:
                joined = df_tmp
            else:
                joined = joined.merge(df_tmp, on='date', how='inner')
            valid_models.append(name)
        if joined is None or joined.empty:
            st.warning("Not enough overlapping predictions to build an ensemble for this asset.")
        else:
            # average across pred_* cols
            pred_cols = [c for c in joined.columns if c.startswith("pred_")]
            joined['ensemble_pred'] = joined[pred_cols].mean(axis=1)
            # load a reference actual series from any model that contains 'actual'
            actual_series = None
            for df in dfs.values():
                df_sub = df[df["ticker"].astype(str) == str(asset)] if "ticker" in df.columns else df
                if not df_sub.empty and 'actual' in df_sub.columns:
                    actual_series = df_sub[['date','actual']].drop_duplicates()
                    break
            if actual_series is not None:
                merged = actual_series.merge(joined[['date','ensemble_pred']], on='date', how='inner')
                m = compute_metrics(merged['actual'], merged['ensemble_pred'])
                st.subheader("Ensemble performance")
                st.metric("RMSE", f"{m['RMSE']:.6f}" if not np.isnan(m['RMSE']) else "N/A")
                st.metric("MAE", f"{m['MAE']:.6f}" if not np.isnan(m['MAE']) else "N/A")
                st.metric("MAPE (%)", f"{m['MAPE']:.3f}" if not np.isnan(m['MAPE']) else "N/A")
                # plot
                plot_df = merged.sort_values('date').tail(500)
                fig = px.line(plot_df, x='date', y=['actual','ensemble_pred'], labels={'value':'Volatility'}, title=f"{asset} — Actual vs Ensemble")
                st.plotly_chart(fig, use_container_width=True)
                st.subheader("Latest ensemble predictions")
                st.dataframe(plot_df.tail(20).reset_index(drop=True))
            else:
                st.warning("Could not find actual volatility series in the multimodal files to compare with ensemble.")
    else:
        # user chose a single model file
        df = dfs.get(chosen_file)
        if df is None or df.empty:
            st.warning("Selected file could not be read or is empty.")
        else:
            # filter by asset/ticker
            df_asset = df[df["ticker"].astype(str) == str(asset)] if "ticker" in df.columns else df
            if df_asset.empty:
                st.warning("No rows for the selected asset in this file.")
            else:
                st.subheader(f"Predictions preview — {chosen_file} — {asset}")
                # ensure actual & predicted exist
                if 'actual' not in df_asset.columns and 'actual_volatility' in df_asset.columns:
                    df_asset = df_asset.rename(columns={'actual_volatility':'actual'})
                if 'predicted' not in df_asset.columns and 'predicted_volatility' in df_asset.columns:
                    df_asset = df_asset.rename(columns={'predicted_volatility':'predicted'})
                # metrics
                m = compute_metrics(df_asset.get('actual'), df_asset.get('predicted'))
                st.metric("RMSE", f"{m['RMSE']:.6f}" if not np.isnan(m['RMSE']) else "N/A")
                st.metric("MAE", f"{m['MAE']:.6f}" if not np.isnan(m['MAE']) else "N/A")
                st.metric("MAPE (%)", f"{m['MAPE']:.3f}" if not np.isnan(m['MAPE']) else "N/A")
                # plot actual vs predicted
                plot_df = df_asset[['date','actual','predicted']].dropna().sort_values('date').tail(500)
                if not plot_df.empty:
                    fig = px.line(plot_df, x='date', y=['actual','predicted'], labels={'value':'Volatility'}, title=f"{asset} — Actual vs Predicted ({Path(chosen_file).stem})")
                    st.plotly_chart(fig, use_container_width=True)
                    st.subheader("Latest predictions")
                    st.dataframe(plot_df.tail(20).reset_index(drop=True))
                else:
                    st.info("No non-null actual/predicted pairs available to plot.")
