# 📈 Stock Volatility & Market Sentiment Prediction with Multimodal AI

**Advanced AI project integrating financial data and NLP for market sentiment analysis, volatility prediction, and strategy backtesting, with interactive visualizations in a Streamlit dashboard.**

---

## 🎯 Project Objective

This project aims to build an advanced system that:

- Predicts **short- and medium-term volatility** for financial assets (IBEX35, S&P500, Nasdaq, etc.).
- Analyzes **market sentiment** using NLP on tweets, news, and financial forums.
- Combines volatility and sentiment predictions to provide **risk signals and potential investment strategies**.
- Visualizes all results through an **interactive dashboard** with real-time insights.

---

## 🛠 Project Structure

The project is organized as follows:

```
notebooks/ # Jupyter notebooks for data processing, modeling, evaluation
models/ # Trained models and scalers
predictions/ # Model predictions and error/metric summaries
data/raw/ # Raw market, tweets, and news datasets
data/processed/ # Cleaned and feature-engineered datasets
dashboard/ # Streamlit app with pages, utils, and assets
```


---

## 📚 Notebooks Overview

1. **Data Collection**
   - Market: `01_data_collection_market.ipynb`
   - Tweets: `02_data_collection_tweets.ipynb`
   - News: `03_data_collection_news.ipynb`

2. **Data Cleaning**
   - Market: `04_data_cleaning_market.ipynb`
   - Tweets: `05_data_cleaning_tweets.ipynb`
   - News: `06_data_cleaning_news.ipynb`

3. **Feature Engineering**
   - Market: `07_future_engineering_market.ipynb`
   - Sentiment: `08_future_engineering_sentiment.ipynb`

4. **Sentiment Analysis**
   - Transformers models: `09_sentiment_analysis_transformers.ipynb`

5. **Volatility Prediction**
   - Time series models: `10_model_volatility_timeseries.ipynb`, `11_model_volatility_transformers.ipynb`
   - Evaluation: `12_lstm_evaluation.ipynb`, `13_tft_evaluation.ipynb`
   - Model comparison & error analysis: `14_model_comparison.ipynb`, `15_error_analysis_metrics.ipynb`, `16_temporal_error_analysis.ipynb`

6. **Multimodal Modeling**
   - LightGBM: `17_multimodal_volatility_prediction_lightgbm.ipynb`
   - CatBoost: `18_multimodal_volatility_prediction_catboost.ipynb`
   - MLP: `19_multimodal_volatility_prediction_mlp.ipynb`
   - LSTM: `20_multimodal_volatility_prediction_lstm.ipynb`

7. **Helper Notebook**
   - `help.ipynb` (guidance and utility functions)

---

## 🔧 Models & Predictions

- **Single-asset predictions:** LSTM, TFT for AAPL, IBEX35, MSFT, NASDAQ, S&P500.
- **Multimodal models:** CatBoost, LightGBM, MLP, LSTM combining market and sentiment features.
- **Preprocessing scalers** for both inputs (X) and targets (y) are provided for reproducibility.
- **Prediction outputs** include CSVs per model/asset and error metrics summaries.

---

## 🗂 Data Sources

- **Market data:** Yahoo Finance (`yfinance`) for OHLCV prices.
- **Social media:** Twitter API v2 or `snscrape` for tweets filtered by tickers and hashtags.
- **News:** NewsAPI or scraping from Bloomberg, Investing, Reuters.
- **Features:** Technical indicators (RSI, MACD, ATR), rolling volatility, returns, and sentiment aggregates.

---

## 🤖 Methodology

### 1️⃣ Sentiment Analysis
- Transformers fine-tuned for financial language: FinBERT, RoBERTa.
- Metrics: Daily sentiment scores (-1 to +1), tweet/news volumes.
- Output: Sentiment indicators for multimodal modeling.

### 2️⃣ Volatility Prediction
- Models: LSTM, GRU, Temporal Fusion Transformer (TFT), N-BEATS.
- Features: Historical volatility, returns, ATR, technical indicators, sentiment features.
- Targets: Realized volatility or implied volatility (VIX optional).

### 3️⃣ Multimodal Fusion
- Concatenation of sentiment and market features.
- Models: CatBoost, LightGBM, MLP, LSTM.
- Output: Combined predictions for risk assessment.

### 4️⃣ Backtesting
- Compare AI-based strategies with:
  - Buy & Hold
  - Momentum strategies
  - Hedging strategies
- Metrics: Simulated capital curves, risk-adjusted returns.

---

## 📊 Dashboard (Streamlit)

**Interactive pages:**

1. **Market Overview:** Prices, volatility, volume comparison.
2. **Sentiment Analysis:** Daily sentiment scores, word clouds.
3. **Volatility Predictions:** Short/medium-term forecasts, optimistic/pessimistic scenarios.
4. **Error Analysis:** Model errors and performance metrics.
5. **Multimodal Insights:** Combined risk signals and predictions.
6. **Sentiment vs Volatility:** Visual correlation analysis.
7. **Model Comparison:** Evaluate all models and multimodal fusion.
8. **Backtesting:** Simulated strategies and capital curves.

---

## ⚡ How to Run

1. Clone the repository:

```
git clone https://github.com/username/stock-volatility-multimodal.git
cd stock-volatility-multimodal/dashboard
```

2. Install dependencies:

´´´
pip install -r requirements.txt
´´´

3. Launch the dashboard:

```
cd dashboard
streamlit run app.py
```

---

## 📈 Results & Insights

- Multimodal models combining **sentiment + market** features outperform single-source predictions.
- AI-driven strategies demonstrate potential **risk-adjusted improvements** over baseline approaches.
- Dashboard enables **real-time monitoring** of volatility, sentiment, and alerts for investment decisions.

---
