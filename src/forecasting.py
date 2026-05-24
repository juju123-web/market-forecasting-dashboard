"""TimesFM forecasting wrapper.

The public TimesFM package has changed APIs across versions. This module keeps
the dashboard integration isolated so the rest of the project can continue to
run even when TimesFM is unavailable or needs environment-specific setup.
"""

from __future__ import annotations

import inspect
from functools import lru_cache

import numpy as np
import pandas as pd

MAX_TIMESFM_2P5_CONTEXT = 16_384
DEFAULT_MODEL_ID = "google/timesfm-2.5-200m-pytorch"


def _clean_numeric_series(series: pd.Series) -> np.ndarray:
    values = pd.Series(series, dtype="float64").dropna().to_numpy()
    if len(values) == 0:
        raise ValueError("TimesFM input series must contain numeric values.")
    return values


def _extract_forecast_values(result, horizon: int) -> np.ndarray:
    """Normalize TimesFM output shapes into a one-dimensional forecast array."""
    if isinstance(result, tuple):
        result = result[0]

    if isinstance(result, pd.DataFrame):
        for column in ("timesfm", "forecast", "prediction", "mean"):
            if column in result.columns:
                return result[column].to_numpy(dtype=float)[:horizon]
        numeric = result.select_dtypes(include=[np.number])
        if not numeric.empty:
            return numeric.iloc[:, -1].to_numpy(dtype=float)[:horizon]

    array = np.asarray(result, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim > 1:
        array = array[0]
    return array[:horizon].astype(float)


def _ensure_complete_forecast(values: np.ndarray, horizon: int) -> np.ndarray:
    if len(values) < horizon:
        raise RuntimeError(
            f"TimesFM returned {len(values)} forecast values, expected {horizon}."
        )
    if np.isnan(values).all():
        raise RuntimeError("TimesFM returned only NaN forecast values.")
    return values[:horizon].astype(float)


def _call_forecast(model, values: np.ndarray, horizon: int) -> np.ndarray:
    forecast = getattr(model, "forecast", None)
    if forecast is None:
        raise RuntimeError("Loaded TimesFM model does not expose a forecast method.")

    attempts = (
        {"horizon": horizon, "inputs": [values]},
        {"inputs": [values], "freq": [0]},
        {"contexts": [values], "freq": [0]},
        {"context": [values], "freq": [0]},
        {"inputs": [values]},
        {},
    )

    for kwargs in attempts:
        try:
            if kwargs:
                result = forecast(**kwargs)
            else:
                result = forecast([values], [0])
            output = _extract_forecast_values(result, horizon)
            if len(output) >= horizon:
                return _ensure_complete_forecast(output, horizon)
        except TypeError:
            continue

    raise RuntimeError("TimesFM forecast call failed for the installed package version.")


def _filtered_config_kwargs(config_class, kwargs: dict[str, object]) -> dict[str, object]:
    try:
        signature = inspect.signature(config_class)
    except (TypeError, ValueError):
        return kwargs

    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _configure_torch_runtime() -> None:
    try:
        import torch
    except ImportError:
        return

    set_precision = getattr(torch, "set_float32_matmul_precision", None)
    if callable(set_precision):
        set_precision("high")


def _load_new_timesfm(timesfm_module, horizon: int, context_length: int):
    model_candidates = [
        "TimesFM_2p5_200M_torch",
        "TimesFM_2p0_500M_torch",
        "TimesFM_1p0_200M_torch",
    ]
    for class_name in model_candidates:
        model_class = getattr(timesfm_module, class_name, None)
        if model_class is None or not hasattr(model_class, "from_pretrained"):
            continue

        model_id = {
            "TimesFM_2p5_200M_torch": DEFAULT_MODEL_ID,
            "TimesFM_2p0_500M_torch": "google/timesfm-2.0-500m-pytorch",
            "TimesFM_1p0_200M_torch": "google/timesfm-1.0-200m-pytorch",
        }[class_name]
        _configure_torch_runtime()
        model = model_class.from_pretrained(model_id)

        config_class = getattr(timesfm_module, "ForecastConfig", None)
        if config_class is not None and hasattr(model, "compile"):
            config_kwargs = {
                "max_context": min(context_length, MAX_TIMESFM_2P5_CONTEXT),
                "max_horizon": horizon,
                "normalize_inputs": True,
                "use_continuous_quantile_head": True,
                "force_flip_invariance": True,
                "infer_is_positive": True,
                "fix_quantile_crossing": True,
            }
            config_kwargs = _filtered_config_kwargs(config_class, config_kwargs)
            model.compile(config_class(**config_kwargs))
        return model

    return None


def _load_legacy_timesfm(timesfm_module, horizon: int):
    timesfm_class = getattr(timesfm_module, "TimesFm", None)
    hparams_class = getattr(timesfm_module, "TimesFmHparams", None)
    checkpoint_class = getattr(timesfm_module, "TimesFmCheckpoint", None)
    if timesfm_class is None or hparams_class is None or checkpoint_class is None:
        return None

    legacy_configs = (
        {
            "hparams": {
                "backend": "cpu",
                "per_core_batch_size": 32,
                "horizon_len": horizon,
                "context_len": 2048,
                "num_layers": 50,
                "use_positional_embedding": False,
            },
            "repo_id": "google/timesfm-2.0-500m-pytorch",
        },
        {
            "hparams": {
                "backend": "cpu",
                "per_core_batch_size": 32,
                "horizon_len": horizon,
                "input_patch_len": 32,
                "output_patch_len": 128,
                "num_layers": 20,
                "model_dims": 1280,
            },
            "repo_id": "google/timesfm-1.0-200m-pytorch",
        },
    )

    last_error = None
    for config in legacy_configs:
        try:
            hparams = hparams_class(**config["hparams"])
            checkpoint = checkpoint_class(huggingface_repo_id=config["repo_id"])
            return timesfm_class(hparams=hparams, checkpoint=checkpoint)
        except TypeError as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise RuntimeError(f"Legacy TimesFM initialization failed: {last_error}") from last_error
    return None


@lru_cache(maxsize=4)
def _load_timesfm_model(horizon: int, context_length: int):
    try:
        import timesfm  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "TimesFM is not installed. Install TimesFM with a compatible Python "
            "environment, or run the app with baseline models only."
        ) from exc

    model = _load_new_timesfm(timesfm, horizon, context_length)
    if model is None:
        model = _load_legacy_timesfm(timesfm, horizon)
    if model is None:
        raise RuntimeError("No supported TimesFM model class was found.")
    return model


def timesfm_forecast(series: pd.Series, horizon: int) -> np.ndarray:
    """Generate a TimesFM forecast for a numeric pandas Series.

    Args:
        series: Historical numeric values.
        horizon: Number of future steps to forecast.

    Returns:
        A one-dimensional numpy array of forecast values.

    Raises:
        RuntimeError: If TimesFM is not installed, cannot be loaded, or cannot
            forecast with the installed API version.
    """
    if horizon <= 0:
        raise ValueError("Forecast horizon must be greater than zero.")

    values = _clean_numeric_series(series)

    try:
        model = _load_timesfm_model(horizon, min(len(values), MAX_TIMESFM_2P5_CONTEXT))
        return _call_forecast(model, values, horizon)
    except RuntimeError as exc:
        if str(exc).startswith("TimesFM is not installed"):
            raise
        raise RuntimeError(f"TimesFM forecasting failed: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"TimesFM forecasting failed: {exc}") from exc
