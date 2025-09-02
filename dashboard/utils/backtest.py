# utils/backtest.py
import pandas as pd


def simple_backtest(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """
    Simple trading strategy backtest:
    - Long if predicted volatility < threshold
    - Short if predicted volatility >= threshold
    """
    df = df.copy()
    df["signal"] = (df["y_pred"] < threshold).astype(int) * 2 - 1  # 1 = long, -1 = short
    df["strategy_return"] = df["signal"].shift() * df["y_true"].pct_change().fillna(0)
    df["cumulative_return"] = (1 + df["strategy_return"]).cumprod()
    return df
