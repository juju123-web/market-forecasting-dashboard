# Project Summary

## Project

Market Forecasting Dashboard with TimesFM

## Goal

Build an interactive dashboard that demonstrates a complete time-series forecasting workflow using real financial market data, simple baselines, model evaluation, and a modular TimesFM wrapper.

## MVP Scope

- Load historical market data from yfinance.
- Use Close price as the target time series.
- Backtest on the most recent forecast horizon.
- Compare naive and moving-average baseline forecasts.
- Add a TimesFM wrapper that can be replaced with the real model call after environment setup.
- Show results in a Streamlit dashboard.

## Resume Angle

Built an interactive financial time-series forecasting dashboard using Python, Streamlit, pandas, Plotly, and yfinance; implemented backtesting and benchmarked forecasts against naive and moving-average baseline models using MAE, RMSE, and MAPE.

## Next Development Tasks

1. Replace the placeholder TimesFM wrapper with the actual model inference API.
2. Add unit tests for baselines and metrics.
3. Add rolling-window backtesting.
4. Add deployment instructions and screenshot.
5. Deploy to Streamlit Community Cloud or Hugging Face Spaces.
