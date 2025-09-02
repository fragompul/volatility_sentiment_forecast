# File: dashboard/pages/07_sentiment_vs_volatility.py
# Compare sentiment indicators vs realized volatility using merged_all_assets.csv and tweet_finance_sentiment.csv
# merged_all_assets.csv header (example): Date,Close,High,Low,Open,Volume,Ticker,Return,Volatility_5d,Volatility_21d,RSI_14,...,tweet_sentiment_mean,...

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"

st.title("💬 Sentiment vs Volatility")

@st.cache_data
def load_merged_assets():
    path = DATA_DIR / "merged_all_assets.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=[0])
    df.columns = [c.strip() for c in df.columns]
    if df.columns[0].lower() not in ("date","datetime"):
        df = df.rename(columns={df.columns[0]: "Date"})
    df = df.rename(columns={c: c.lower() for c in df.columns})
    # normalize column names
    if 'date' not in df.columns and 'Date' in df.columns:
        df = df.rename(columns={'Date': 'date'})
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data
def load_tweet_sentiment_raw():
    path = RAW_DIR / "tweet_finance_sentiment.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=[1])  # header: ticker,date,text,...
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={c: c.lower() for c in df.columns})
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    return df

merged = load_merged_assets()
tweets_raw = load_tweet_sentiment_raw()

if merged.empty:
    st.warning("merged_all_assets.csv not found in data/processed/. This page requires that file (contains per-ticker sentiment features).")
else:
    tickers = sorted(merged['ticker'].unique().astype(str))
    asset = st.selectbox("Select asset", tickers, index=0)
    df_asset = merged[merged['ticker'].astype(str) == str(asset)].sort_values('date').reset_index(drop=True)
    if df_asset.empty:
        st.warning("No rows for the selected asset in merged_all_assets.csv")
    else:
        st.subheader(f"Time series — {asset}")
        cols_to_plot = []
        if 'tweet_sentiment_mean' in df_asset.columns:
            cols_to_plot.append('tweet_sentiment_mean')
        if 'volatility_21d' in df_asset.columns:
            cols_to_plot.append('volatility_21d')
        if not cols_to_plot:
            st.info("Neither 'tweet_sentiment_mean' nor 'volatility_21d' are present in merged_all_assets.csv")
        else:
            # plot sentiment and volatility normalized for comparison
            ts = df_asset[['date'] + cols_to_plot].dropna()
            if ts.empty:
                st.info("No non-null time series values to show.")
            else:
                # normalize to z-score for overlay
                ts_norm = ts.copy()
                for c in cols_to_plot:
                    ts_norm[c] = (ts_norm[c] - ts_norm[c].mean()) / (ts_norm[c].std() if ts_norm[c].std() != 0 else 1)
                fig = px.line(ts_norm, x='date', y=cols_to_plot, title=f"{asset} — Normalized sentiment vs volatility (z-score)")
                st.plotly_chart(fig, use_container_width=True)

        # Scatter and correlation
        if 'tweet_sentiment_mean' in df_asset.columns and 'volatility_21d' in df_asset.columns:
            st.subheader("Scatter: sentiment vs 21d volatility")
            scatter_df = df_asset[['tweet_sentiment_mean','volatility_21d']].dropna()
            if not scatter_df.empty:
                # compute simple linear fit
                x = scatter_df['tweet_sentiment_mean'].astype(float)
                y = scatter_df['volatility_21d'].astype(float)
                coef = np.polyfit(x, y, 1) if len(x) > 1 else [np.nan, np.nan]
                line_x = np.linspace(x.min(), x.max(), 50)
                line_y = np.polyval(coef, line_x)
                fig = px.scatter(scatter_df, x='tweet_sentiment_mean', y='volatility_21d', title=f"{asset} — sentiment vs volatility (21d)")
                fig.add_traces(px.line(x=line_x, y=line_y, labels={'x':'sentiment','y':'volatility'}).data)
                st.plotly_chart(fig, use_container_width=True)
                corr = x.corr(y)
                st.metric("Pearson corr (tweet_sentiment_mean vs volatility_21d)", f"{corr:.4f}")
            else:
                st.info("Not enough non-null pairs to compute scatter/correlation.")

            # rolling correlation
            st.subheader("Rolling correlation (30-day)")
            roll = df_asset[['date','tweet_sentiment_mean','volatility_21d']].set_index('date').dropna()
            if roll.shape[0] >= 30:
                rolling_corr = roll['tweet_sentiment_mean'].rolling(30).corr(roll['volatility_21d'])
                rolling_corr = rolling_corr.reset_index().rename(columns={0: 'rolling_corr'})
                fig = px.line(rolling_corr, x='date', y='tweet_sentiment_mean', title="30-day rolling correlation (tweet_sentiment_mean vs volatility_21d)")
                # note: px.line above will use column name 'tweet_sentiment_mean' because of how we built DF,
                # so replace with appropriate series
                fig = px.line(pd.DataFrame({'date': rolling_corr['date'], 'rolling_corr': rolling_corr['tweet_sentiment_mean']}), x='date', y='rolling_corr', title="30-day rolling correlation (tweet_sentiment_mean vs volatility_21d)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data for 30-day rolling correlation (need >= 30 points).")

        # Show top tweets (from raw tweets) for asset
        st.subheader("Sample tweets (from tweet_finance_sentiment.csv)")
        if tweets_raw.empty:
            st.info("tweet_finance_sentiment.csv not found in data/raw/")
        else:
            tw_asset = tweets_raw[tweets_raw['ticker'].astype(str) == str(asset)] if 'ticker' in tweets_raw.columns else tweets_raw
            if tw_asset.empty:
                st.info("No tweets for this ticker in tweet_finance_sentiment.csv")
            else:
                # show top positive and top negative by sentiment_score
                if 'sentiment_score' in tw_asset.columns:
                    top_pos = tw_asset.sort_values('sentiment_score', ascending=False).head(5)
                    top_neg = tw_asset.sort_values('sentiment_score', ascending=True).head(5)
                    st.markdown("**Top positive tweets**")
                    st.dataframe(top_pos[['date','text','likes','retweets','sentiment_score']].head(5))
                    st.markdown("**Top negative tweets**")
                    st.dataframe(top_neg[['date','text','likes','retweets','sentiment_score']].head(5))
                else:
                    # fallback: show most liked tweets
                    top = tw_asset.sort_values(['likes','retweets'], ascending=False).head(10)
                    st.dataframe(top[['date','text','likes','retweets']].head(10))
