"""Data loading utilities for financial time-series data."""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def load_market_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Download historical market data for a ticker.

    Args:
        ticker: Market ticker, for example "TSLA", "AAPL", or "BTC-USD".
        period: yfinance period string, for example "6mo", "1y", "2y", or "5y".

    Returns:
        A DataFrame indexed by date with at least a Close column.

    Raises:
        ValueError: If no data is returned or the Close column is missing.
    """
    cleaned_ticker = ticker.strip().upper()
    if not cleaned_ticker:
        raise ValueError("Ticker cannot be empty.")

    data = yf.download(cleaned_ticker, period=period, auto_adjust=False, progress=False)

    if data.empty:
        raise ValueError(f"No market data returned for ticker: {cleaned_ticker}")

    # yfinance can return multi-index columns in some cases. Flatten them defensively.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

    if "Close" not in data.columns:
        raise ValueError(f"Downloaded data for {cleaned_ticker} does not contain a Close column.")

    data = data.dropna(subset=["Close"]).copy()
    data.index = pd.to_datetime(data.index)
    return data


def get_close_series(data: pd.DataFrame) -> pd.Series:
    """Extract a clean Close price series from a market data DataFrame."""
    if "Close" not in data.columns:
        raise ValueError("Input DataFrame must contain a Close column.")
    close = data["Close"].dropna().astype(float)
    close.name = "Close"
    return close
