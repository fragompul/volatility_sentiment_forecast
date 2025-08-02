# volatility_sentiment_forecast
Predicción de volatilidad bursátil y sentimiento de mercado con IA multimodal


Estructura:
```
volatility_sentiment_forecast/
│
├── data/
│   ├── raw/
│   │   ├── market_data.csv
│   │   ├── tweets_data.csv
│   │   └── news_data.csv
│   ├── processed/
│   │   ├── market_features.csv
│   │   ├── sentiment_features.csv
│   │   └── merged_dataset.csv
│
├── notebooks/
│   ├── 01_data_collection_market.ipynb
│   ├── 02_data_collection_tweets.ipynb
│   ├── 03_data_collection_news.ipynb
│   ├── 04_data_cleaning_market.ipynb
│   ├── 05_data_cleaning_tweets.ipynb
│   ├── 06_data_cleaning_news.ipynb
│   ├── 07_feature_engineering_market.ipynb
│   ├── 08_feature_engineering_sentiment.ipynb
│   ├── 09_sentiment_analysis_transformers.ipynb
│   ├── 10_model_volatility_timeseries.ipynb
│   ├── 11_model_volatility_transformers.ipynb
│   ├── 12_model_sentiment_impact.ipynb
│   ├── 13_model_multimodal_fusion.ipynb
│   ├── 14_backtesting_strategies.ipynb
│   └── 15_final_predictions.ipynb
│
├── models/
│   ├── volatility_model_lstm.pkl
│   ├── volatility_model_transformer.pkl
│   ├── sentiment_model_roberta.pkl
│   └── multimodal_fusion.pkl
│
├── dashboard/
│   ├── app.py
│   ├── components/
│   │   ├── market_view.py
│   │   ├── sentiment_view.py
│   │   ├── combined_analysis.py
│   │   └── alerts.py
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── sentiment_analysis.py
│   ├── volatility_prediction.py
│   ├── fusion_model.py
│   └── backtesting.py
│
├── docs/
│   ├── dashboard_screenshot.png
│   ├── architecture_diagram.png
│   └── results_summary.png
│
├── requirements.txt
│
├── Memoria.pdf
│
└── README.md
´´´
