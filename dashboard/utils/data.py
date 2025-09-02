# utils/data.py
import pandas as pd
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_csv(filename: str) -> pd.DataFrame:
    """
    Generic CSV loader from data directory.
    """
    path = DATA_DIR / filename
    return pd.read_csv(path)


def load_market(asset: str) -> pd.DataFrame:
    """
    Load market data for a given asset.
    Example: AAPL_market.csv
    """
    return load_csv(f"processed/{asset}_market.csv")


def load_sentiment() -> pd.DataFrame:
    """
    Load sentiment features (tweets + news).
    """
    return load_csv("processed/sentiment_features.csv")


def load_predictions(model: str, asset: str) -> pd.DataFrame:
    """
    Load predictions for a given model and asset.
    Supported models: lstm, tft
    Example: predictions_lstm_AAPL.csv
    """
    return load_csv(f"processed/predictions_{model}_{asset}.csv")


def load_errors(model: str, asset: str) -> pd.DataFrame:
    """
    Load errors for a given model and asset.
    Example: errors_lstm_AAPL.csv
    """
    return load_csv(f"processed/errors_{model}_{asset}.csv")


def load_metrics(summary_type: str = "all") -> pd.DataFrame:
    """
    Load metrics summary.
    summary_type: all | lstm | tft
    """
    if summary_type == "all":
        return load_csv("processed/metrics_all_summary.csv")
    elif summary_type == "lstm":
        return load_csv("processed/metrics_lstm_summary.csv")
    elif summary_type == "tft":
        return load_csv("processed/metrics_tft_summary.csv")
    else:
        raise ValueError("Invalid summary_type. Choose from all, lstm, tft.")


def load_multimodal_predictions(model: str) -> pd.DataFrame:
    """
    Load multimodal predictions.
    Models: catboost, lgb, lstm, mlp
    """
    return load_csv(f"processed/predictions_multimodal_{model}.csv")


def load_merged_all_assets() -> pd.DataFrame:
    """
    Load merged dataset with all assets, technical indicators, and sentiment.
    """
    return load_csv("processed/merged_all_assets.csv")
