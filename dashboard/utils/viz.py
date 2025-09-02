# utils/viz.py
import matplotlib.pyplot as plt
import pandas as pd


def plot_price(df: pd.DataFrame, asset: str):
    """
    Plot closing price of an asset.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(df["Date"], df["Close"], label=f"{asset} Close Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"{asset} Closing Price")
    plt.legend()
    return plt


def plot_predictions(df: pd.DataFrame, model: str, asset: str):
    """
    Plot true vs predicted values.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["y_true"], label="True")
    plt.plot(df["date"], df["y_pred"], label=f"Predicted ({model.upper()})")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.title(f"{asset} - {model.upper()} Predictions")
    plt.legend()
    return plt


def plot_errors(df: pd.DataFrame, model: str, asset: str):
    """
    Plot error over time.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["error"], label="Error")
    plt.xlabel("Date")
    plt.ylabel("Error")
    plt.title(f"{asset} - {model.upper()} Errors")
    plt.legend()
    return plt
