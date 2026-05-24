"""Simple baseline forecasting models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def naive_forecast(series: pd.Series, horizon: int) -> np.ndarray:
    """Forecast future values as the last observed value."""
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    clean_series = series.dropna().astype(float)
    if clean_series.empty:
        raise ValueError("series cannot be empty.")
    last_value = float(clean_series.iloc[-1])
    return np.repeat(last_value, horizon)


def moving_average_forecast(series: pd.Series, horizon: int, window: int = 30) -> np.ndarray:
    """Forecast future values as the average of the most recent window."""
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if window <= 0:
        raise ValueError("window must be positive.")
    clean_series = series.dropna().astype(float)
    if clean_series.empty:
        raise ValueError("series cannot be empty.")

    effective_window = min(window, len(clean_series))
    average_value = float(clean_series.tail(effective_window).mean())
    return np.repeat(average_value, horizon)
