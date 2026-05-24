"""Plotly visualization helpers."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def make_future_index(last_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    """Create a business-day future index after the last observed date."""
    return pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon)


def plot_historical_and_forecast(
    history: pd.Series,
    forecasts: dict[str, object],
    future_index: pd.DatetimeIndex | None = None,
) -> go.Figure:
    """Create a Plotly chart with historical values and forecast lines."""
    if history.empty:
        raise ValueError("history cannot be empty.")

    horizon = len(next(iter(forecasts.values()))) if forecasts else 0
    if future_index is None and horizon > 0:
        future_index = make_future_index(pd.Timestamp(history.index[-1]), horizon)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history.values,
            mode="lines",
            name="Historical Close",
        )
    )

    for model_name, forecast_values in forecasts.items():
        fig.add_trace(
            go.Scatter(
                x=future_index,
                y=forecast_values,
                mode="lines",
                name=model_name,
            )
        )

    fig.update_layout(
        title="Historical Prices and Forecasts",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        legend_title="Series",
    )
    return fig
