# 📈 Stock Volatility & Market Sentiment Prediction with Multimodal AI

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-predictive_modeling-8A2BE2)

> **An advanced end-to-end Machine Learning pipeline integrating financial time-series data and Natural Language Processing (NLP) for market sentiment analysis, volatility forecasting, and strategy backtesting.**

---

## 🎯 Executive Summary

This project bridges the gap between quantitative finance and artificial intelligence. By combining historical market data with real-time public sentiment extracted from social media and news, it provides a robust **Multimodal AI system** capable of:

- 📉 **Forecasting short- and medium-term volatility** for major assets (IBEX35, S&P500, Nasdaq, AAPL, MSFT).
- 🧠 **Extracting market sentiment** using state-of-the-art Transformer models (FinBERT, RoBERTa) on tweets and financial news.
- ⚡ **Fusing data streams (Multimodal)** to generate risk signals and actionable investment strategies.
- 📊 **Delivering actionable insights** through a fully interactive, real-time Streamlit dashboard.

---

## 🏗️ Architecture & Methodology

Our methodology is divided into four core pillars, ensuring a rigorous approach from raw data to actionable predictions:

### 1️⃣ Sentiment Analysis (NLP)
- **Models:** Fine-tuned Transformers specifically adapted for financial lexicon (e.g., FinBERT).
- **Features:** Daily sentiment aggregation (-1 to +1 scale), volume spikes, and entity-specific mood tracking.
- **Sources:** Twitter API/snscrape, NewsAPI, and financial aggregators (Bloomberg, Reuters).

### 2️⃣ Volatility Prediction (Time-Series)
- **Models:** Long Short-Term Memory (LSTM), Temporal Fusion Transformers (TFT), N-BEATS.
- **Features:** Historical OHLCV, Returns, Average True Range (ATR), MACD, RSI, and rolling volatility metrics.

### 3️⃣ Multimodal Fusion
- **Approach:** Feature concatenation blending temporal market indicators with NLP-derived sentiment embeddings.
- **Ensemble Models:** CatBoost, LightGBM, Multi-Layer Perceptrons (MLP), and advanced LSTMs.
- **Goal:** Outperform single-source models by capturing the nuance of human emotion in market movements.

### 4️⃣ Strategy Backtesting
- **Simulations:** Comparing our AI-driven signals against standard baselines (Buy & Hold, Momentum, Hedging).
- **Metrics:** Capital curve simulations, Maximum Drawdown, and Risk-Adjusted Returns.

---

## 💻 Tech Stack & Project Structure

The repository is modularly designed to separate data engineering, modeling, evaluation, and deployment:

    ├── data/
    │   ├── raw/                 # Raw market data, tweets, and news
    │   └── processed/           # Cleaned, engineered, and scaled datasets
    ├── models/                  # Serialized weights, scalers, and pipelines
    ├── notebooks/
    │   ├── 01_to_03_data_collection/    # Web scraping and API integrations
    │   ├── 04_to_06_data_cleaning/      # Data wrangling and ETL pipelines
    │   ├── 07_to_08_feat_engineering/   # Technical indicators & sentiment aggregators
    │   ├── 09_sentiment_analysis/       # LLM/Transformer inference
    │   ├── 10_to_16_volatility_models/  # Time-series forecasting & error analysis
    │   └── 17_to_20_multimodal_fusion/  # LightGBM, CatBoost, and Deep Learning fusion
    ├── predictions/             # Evaluation metrics, CSV outputs, and backtest logs
    └── dashboard/               # Streamlit application source code

---

## 🚀 Getting Started

Reproduce the environment and launch the interactive dashboard locally in just a few steps:

### Prerequisites
- Python 3.9+
- Git

### Installation

**1. Clone the repository:**
```bash
git clone https://github.com/fragompul/stock-volatility-multimodal.git
cd stock-volatility-multimodal
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Launch the Streamlit Dashboard:**
```bash
cd dashboard
streamlit run app.py
```

---

## 📊 Dashboard Features

The Streamlit application acts as the front-end for our ML models, featuring several interactive modules:

- **Market Overview:** Interactive candlestick charts, volume profiles, and historical volatility.
- **Sentiment Engine:** Word clouds, daily sentiment tracking, and news impact analysis.
- **Volatility Forecasts:** Short vs. medium-term horizons with confidence intervals (optimistic/pessimistic).
- **Model Explainability & Error Analysis:** Transparent performance metrics (RMSE, MAE) and temporal error distribution.
- **Backtesting Engine:** Visual comparison of AI-driven capital curves vs. traditional benchmarks.

---

## 🏆 Key Findings & Results

- **Superiority of Multimodal Approaches:** Models fusing *Sentiment + Market Data* consistently out-predicted single-source baselines, especially during high-stress market events.
- **Alpha Generation:** AI-driven dynamic hedging strategies demonstrated significant improvements in risk-adjusted returns compared to static Buy & Hold approaches.

---

## 📬 Contact & Author

**Francisco Javier Gómez Pulido** *Data Scientist & Machine Learning Engineer*

- 📧 Email: [frangomezpulido2002@gmail.com](mailto:frangomezpulido2002@gmail.com)
- 💼 LinkedIn: [linkedin.com/in/frangomezpulido](https://www.linkedin.com/in/frangomezpulido)
- 🐙 GitHub: [github.com/fragompul](https://github.com/fragompul)

---
*If you found this project interesting or helpful, feel free to drop a ⭐ on the repository!*
