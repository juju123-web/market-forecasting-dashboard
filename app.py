"""Streamlit app for the market forecasting dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.baselines import moving_average_forecast, naive_forecast
from src.data_loader import get_close_series, load_market_data
from src.forecasting import timesfm_forecast
from src.metrics import evaluate_forecasts
from src.plotting import make_future_index, plot_historical_and_forecast


DEFAULT_TICKERS = ["AAPL", "TSLA", "NVDA", "SPY", "QQQ", "BTC-USD"]


st.set_page_config(page_title="Market Forecasting Dashboard", layout="wide")

st.title("Market Forecasting Dashboard")
st.caption("Educational time-series forecasting dashboard using market data, baselines, and a TimesFM wrapper.")

st.warning(
    "This project is for educational and research purposes only. It is not financial advice."
)

with st.sidebar:
    st.header("Settings")
    ticker = st.selectbox("Ticker", DEFAULT_TICKERS, index=1)
    custom_ticker = st.text_input("Or enter a custom ticker", value="")
    selected_ticker = custom_ticker.strip().upper() if custom_ticker.strip() else ticker

    period = st.selectbox("Historical period", ["6mo", "1y", "2y", "5y"], index=2)
    horizon = st.slider("Forecast horizon", min_value=5, max_value=90, value=30, step=5)
    ma_window = st.slider("Moving average window", min_value=5, max_value=90, value=30, step=5)
    include_timesfm = st.checkbox("Try TimesFM forecast", value=False)
    run_button = st.button("Run forecast", type="primary")


if run_button:
    try:
        data = load_market_data(selected_ticker, period=period)
        close = get_close_series(data)

        if len(close) <= horizon:
            st.error("Not enough observations for the selected forecast horizon.")
            st.stop()

        train = close.iloc[:-horizon]
        test = close.iloc[-horizon:]
        future_index = test.index

        predictions: dict[str, object] = {
            "Naive": naive_forecast(train, horizon),
            f"Moving Average ({ma_window})": moving_average_forecast(train, horizon, window=ma_window),
        }

        if include_timesfm:
            try:
                predictions["TimesFM"] = timesfm_forecast(train, horizon)
            except Exception as exc:  # noqa: BLE001
                st.info(f"TimesFM forecast skipped: {exc}")

        metrics_df = evaluate_forecasts(test.values, predictions)

        col1, col2, col3 = st.columns(3)
        col1.metric("Ticker", selected_ticker)
        col2.metric("Observations", len(close))
        col3.metric("Test horizon", horizon)

        st.subheader("Forecast Chart")
        fig = plot_historical_and_forecast(train.tail(252), predictions, future_index=future_index)
        fig.add_scatter(x=test.index, y=test.values, mode="lines", name="Actual Test")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Evaluation Metrics")
        st.dataframe(metrics_df, use_container_width=True)

        st.subheader("Recent Data")
        st.dataframe(data.tail(10), use_container_width=True)

        forecast_df = pd.DataFrame(index=future_index)
        forecast_df["Actual"] = test.values
        for model_name, values in predictions.items():
            forecast_df[model_name] = values

        csv = forecast_df.to_csv(index=True).encode("utf-8")
        st.download_button(
            label="Download forecast CSV",
            data=csv,
            file_name=f"{selected_ticker}_forecast.csv",
            mime="text/csv",
        )

    except Exception as exc:  # noqa: BLE001
        st.error(f"Forecast failed: {exc}")
else:
    st.info("Choose settings in the sidebar and click Run forecast.")
    st.markdown(
        """
        ### What this dashboard does

        - Downloads historical market data with yfinance.
        - Splits the most recent horizon as a test set.
        - Forecasts the test period with simple baselines.
        - Optionally attempts to call a TimesFM wrapper.
        - Compares models using MAE, RMSE, and MAPE.
        """
    )
