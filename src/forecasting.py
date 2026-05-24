"""TimesFM forecasting wrapper.

This module intentionally keeps TimesFM integration isolated. The baseline
pipeline should work even when TimesFM is not installed in the local environment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def timesfm_forecast(series: pd.Series, horizon: int) -> np.ndarray:
    """Generate a TimesFM forecast for a numeric time series.

    This starter implementation provides a safe placeholder so the app can run
    before the TimesFM environment is configured. Replace the fallback section
    with the actual TimesFM model call once the dependency is installed.

    Args:
        series: Historical numeric observations.
        horizon: Number of future steps to forecast.

    Returns:
        Forecast values as a NumPy array.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    clean_series = series.dropna().astype(float)
    if clean_series.empty:
        raise ValueError("series cannot be empty.")

    try:
        import timesfm  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "TimesFM is not installed in this environment. Install a compatible "
            "TimesFM package, then replace the placeholder implementation in "
            "src/forecasting.py with the actual model call."
        ) from exc

    # TODO: Replace this placeholder with the actual TimesFM API call.
    # Current fallback mirrors a naive forecast after confirming TimesFM imports.
    last_value = float(clean_series.iloc[-1])
    return np.repeat(last_value, horizon)
