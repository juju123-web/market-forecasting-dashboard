"""Forecast evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_mae(y_true, y_pred) -> float:
    """Calculate mean absolute error."""
    return float(mean_absolute_error(y_true, y_pred))


def calculate_rmse(y_true, y_pred) -> float:
    """Calculate root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def calculate_mape(y_true, y_pred) -> float:
    """Calculate mean absolute percentage error.

    Zero actual values are ignored to avoid division by zero.
    """
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    mask = y_true_array != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true_array[mask] - y_pred_array[mask]) / y_true_array[mask])) * 100)


def evaluate_forecasts(y_true, predictions_dict: dict[str, object]) -> pd.DataFrame:
    """Evaluate multiple forecast arrays against actual values."""
    rows = []
    y_true_array = np.asarray(y_true, dtype=float)

    for model_name, y_pred in predictions_dict.items():
        y_pred_array = np.asarray(y_pred, dtype=float)
        if len(y_pred_array) != len(y_true_array):
            raise ValueError(f"Prediction length mismatch for {model_name}.")
        rows.append(
            {
                "Model": model_name,
                "MAE": calculate_mae(y_true_array, y_pred_array),
                "RMSE": calculate_rmse(y_true_array, y_pred_array),
                "MAPE (%)": calculate_mape(y_true_array, y_pred_array),
            }
        )

    return pd.DataFrame(rows).sort_values("MAE", ascending=True).reset_index(drop=True)
