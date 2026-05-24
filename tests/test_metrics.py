import math

import pandas as pd
import pytest

from src.metrics import (
    calculate_mae,
    calculate_mape,
    calculate_rmse,
    evaluate_forecasts,
)


def test_calculate_mae():
    assert calculate_mae([10, 20, 30], [12, 18, 33]) == pytest.approx(7 / 3)


def test_calculate_rmse():
    assert calculate_rmse([10, 20, 30], [12, 18, 33]) == pytest.approx(
        math.sqrt(17 / 3)
    )


def test_calculate_mape_ignores_zero_actual_values():
    assert calculate_mape([0, 100, 200], [50, 110, 180]) == pytest.approx(10.0)


def test_calculate_mape_returns_nan_when_all_actual_values_are_zero():
    assert math.isnan(calculate_mape([0, 0], [1, 2]))


def test_evaluate_forecasts_returns_sorted_metrics_dataframe():
    y_true = [10, 20, 30]
    predictions = {
        "Worse": [20, 30, 40],
        "Better": [11, 19, 31],
    }

    result = evaluate_forecasts(y_true, predictions)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["Model", "MAE", "RMSE", "MAPE (%)"]
    assert result["Model"].tolist() == ["Better", "Worse"]
    assert result.loc[0, "MAE"] == pytest.approx(1.0)
    assert result.loc[1, "MAE"] == pytest.approx(10.0)


def test_evaluate_forecasts_rejects_prediction_length_mismatch():
    with pytest.raises(ValueError, match="Prediction length mismatch for Short"):
        evaluate_forecasts([1, 2, 3], {"Short": [1, 2]})
