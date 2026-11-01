"""Signal definitions shared by the batch job and the backtester."""

import pandas as pd


def rolling_mean_signal(close: pd.Series, window: int) -> pd.Series:
    """1 when close is above its rolling mean, else 0.

    The first ``window - 1`` bars have no rolling mean and get signal 0.
    """
    if window < 1:
        raise ValueError("window must be a positive integer")
    rolling_mean = close.rolling(window=window).mean()
    signal = (close > rolling_mean).astype(int)
    signal[rolling_mean.isna()] = 0
    return signal
