# File: dashboard/pages/02_sentiment.py
# Sentiment analysis: time series, volume, sample tweets

from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"

st.title("💬 Sentiment Analysis")
asset = st.selectbox("Select asset", ["AAPL", "TSLA"], index=0)

@st.cache_data
def load_sentiment():
    path = DATA_DIR / "sentiment_features.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=[0])
        if "date" not in df.columns:
            df.rename(columns={df.columns[0]: "date"}, inplace=True)
        return df
    return pd.DataFrame()

sent = load_sentiment()
if sent.empty:
    st.warning("No sentiment data available in data/processed/sentiment_features.csv")
else:
    asset_df = sent[sent["asset"] == asset] if "asset" in sent.columns else sent
    fig = px.line(asset_df, x="date", y=[c for c in asset_df.columns if "sentiment" in c][:2])
    st.plotly_chart(fig, use_container_width=True)

    st.write("***")
    st.subheader("Mentions volume")
    if "tweet_volume" in asset_df.columns:
        st.bar_chart(asset_df.set_index("date")["tweet_volume"].tail(60))

    st.subheader("Sample of tweets (raw)")
    tweets_path = ROOT / "data" / "raw" / f"tweets_partial_{asset}.csv"
    if tweets_path.exists():
        tw = pd.read_csv(tweets_path, nrows=200)
        st.dataframe(tw.head(10))
    else:
        st.info(f"No tweet file found for {asset}. Run notebooks/02_data_collection_tweets.ipynb to generate it.")
