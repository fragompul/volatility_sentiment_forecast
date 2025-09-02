# File: dashboard/app.py
# Run with: streamlit run app.py

from pathlib import Path
import streamlit as st

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
PRED_DIR = PROJECT_ROOT / "predictions"

st.set_page_config(page_title="Volatility & Sentiment Dashboard", layout="wide")

# Load CSS
def local_css(file_name: Path):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

local_css(ASSETS_DIR / "style.css")

# Main content
st.title("📘 Volatility & Market Sentiment Dashboard")
st.markdown(
    "This project explores the relationship between **financial volatility** "
    "and **market sentiment**, integrating multimodal data sources and "
    "machine learning models for prediction and analysis.\n\n"
    
    "Using historical stock prices, technical indicators, and sentiment features "
    "extracted from tweets and financial news, the project aims to capture both "
    "quantitative and qualitative signals from the market. Predictions are generated "
    "using models such as **LSTM**, **Temporal Fusion Transformers (TFT)**, and "
    "**other advanced neural networks**, which are then combined with sentiment "
    "information to improve forecasting accuracy.\n\n"
    
    "The dashboard offers an interactive interface for users to explore market trends, "
    "analyze sentiment evolution, compare model predictions, review errors, and examine "
    "backtesting results, providing actionable insights for risk assessment and decision-making."
)

st.markdown("---")
st.subheader("📑 Project Structure")
st.write(
    "- **📈 Market Data (01_overview.py)**: Explore raw stock and index data.\n"
    "- **📰 Sentiment Analysis (02_sentiment.py)**: View sentiment features extracted from text (tweets/news).\n"
    "- **📘 Volatility Models (03_volatility_models.py)**: Predictions from LSTM and TFT models.\n"
    "- **📉 Model Errors (04_errors.py)**: Residual analysis to evaluate predictions.\n"
    "- **📑 Metrics Summary (05_metrics.py)**: Overview of error metrics across assets and models.\n"
    "- **🧩 Multimodal Predictions (06_multimodal.py)**: Combine sentiment and market data using multiple ML models.\n"
    "- **🔀 Sentiment vs Volatility (07_sentiment_vs_volatility.py)**: Study correlations between sentiment and volatility.\n"
    "- **⚖️ Model Comparison (08_model_comparison.py)**: Benchmark models against each other."
)

# Footer
st.markdown("---")
st.markdown("📌 Personal project by **Francisco Javier Gómez Pulido**")
st.markdown("📜 Linkedin profile: www.linkedin.com/in/frangomezpulido")
st.markdown("🔗 GitHub repository: https://github.com/fragompul/hospital_capacity_prediction.git")