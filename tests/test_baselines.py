import numpy as np
import pandas as pd
import pytest

from src.baselines import moving_average_forecast, naive_forecast


def test_naive_forecast_repeats_last_non_null_value():
    series = pd.Series([10.0, 11.5, np.nan, 13.0])

    forecast = naive_forecast(series, horizon=3)

    np.testing.assert_array_equal(forecast, np.array([13.0, 13.0, 13.0]))


def test_naive_forecast_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="horizon must be positive"):
        naive_forecast(pd.Series([1.0, 2.0]), horizon=0)

    with pytest.raises(ValueError, match="series cannot be empty"):
        naive_forecast(pd.Series([np.nan, np.nan]), horizon=2)


def test_moving_average_forecast_uses_last_window_average():
    series = pd.Series([10.0, 20.0, 30.0, 40.0])

    forecast = moving_average_forecast(series, horizon=2, window=3)

    np.testing.assert_array_equal(forecast, np.array([30.0, 30.0]))


def test_moving_average_forecast_uses_available_values_when_window_is_large():
    series = pd.Series([10.0, 20.0])

    forecast = moving_average_forecast(series, horizon=2, window=30)

    np.testing.assert_array_equal(forecast, np.array([15.0, 15.0]))


def test_moving_average_forecast_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="horizon must be positive"):
        moving_average_forecast(pd.Series([1.0, 2.0]), horizon=0, window=2)

    with pytest.raises(ValueError, match="window must be positive"):
        moving_average_forecast(pd.Series([1.0, 2.0]), horizon=2, window=0)

    with pytest.raises(ValueError, match="series cannot be empty"):
        moving_average_forecast(pd.Series([np.nan]), horizon=2, window=1)
