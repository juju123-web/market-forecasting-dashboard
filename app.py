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
METRIC_HELP = {
    "MAE": "Average absolute dollar error. Lower is better.",
    "RMSE": "Like MAE, but large misses are penalized more heavily. Lower is better.",
    "MAPE (%)": "Average absolute percentage error. Lower is better; zero actual values are ignored.",
}


def build_forecasts(
    history: pd.Series,
    horizon: int,
    ma_window: int,
    include_timesfm: bool,
) -> tuple[dict[str, object], str | None]:
    """Generate baseline forecasts and optionally TimesFM forecasts."""
    predictions: dict[str, object] = {
        "Naive": naive_forecast(history, horizon),
        f"Moving Average ({ma_window})": moving_average_forecast(
            history,
            horizon,
            window=ma_window,
        ),
    }

    timesfm_error = None
    if include_timesfm:
        try:
            predictions["TimesFM"] = timesfm_forecast(history, horizon)
        except Exception as exc:  # noqa: BLE001
            timesfm_error = str(exc)

    return predictions, timesfm_error


st.set_page_config(page_title="Market Forecasting Dashboard", layout="wide")

st.title("Market Forecasting Dashboard")
st.caption(
    "Backtest financial time-series forecasts with yfinance data, simple baselines, "
    "and an optional TimesFM model."
)

st.warning(
    "Educational and research use only. This dashboard is not financial advice and "
    "does not account for news, fundamentals, macro events, or execution costs."
)

with st.sidebar:
    st.header("Forecast Settings")
    forecast_mode = st.radio(
        "Mode",
        ["Backtest Mode", "Future Forecast Mode"],
        help=(
            "Backtest Mode holds out recent data for evaluation. Future Forecast Mode "
            "uses all available close prices and predicts the next business days."
        ),
    )
    ticker = st.selectbox(
        "Ticker",
        DEFAULT_TICKERS,
        index=1,
        help="Choose a preset Yahoo Finance ticker.",
    )
    custom_ticker = st.text_input(
        "Custom ticker",
        value="",
        placeholder="Example: MSFT",
        help="Optional. Overrides the preset ticker when filled.",
    )
    selected_ticker = custom_ticker.strip().upper() if custom_ticker.strip() else ticker

    period = st.selectbox(
        "Historical period",
        ["6mo", "1y", "2y", "5y"],
        index=2,
        help="Amount of historical data to download before backtesting.",
    )
    horizon = st.slider(
        "Forecast horizon",
        min_value=5,
        max_value=90,
        value=30,
        step=5,
        help=(
            "Backtest Mode holds out the last N observations. Future Forecast Mode "
            "predicts the next N business days."
        ),
    )
    ma_window = st.slider(
        "Moving average window",
        min_value=5,
        max_value=90,
        value=30,
        step=5,
        help="Number of recent observations averaged by the moving-average baseline.",
    )
    include_timesfm = st.checkbox(
        "Try TimesFM forecast",
        value=False,
        help="Runs TimesFM when it is installed and compatible with the local Python environment.",
    )
    run_button = st.button("Run forecast", type="primary")

with st.expander("How this backtest works", expanded=False):
    st.markdown(
        """
        **Backtest Mode:** downloads historical close prices, keeps the last selected
        horizon as the test set, forecasts that same period from earlier data, and
        compares predictions with the actual held-out prices.

        **Future Forecast Mode:** downloads historical close prices, trains each model on
        all available close prices, and forecasts the next selected number of business days.
        Future forecasts do not have evaluation metrics because actual future prices are
        not available yet.
        """
    )


if run_button:
    try:
        with st.spinner(f"Downloading {selected_ticker} data and running forecasts..."):
            data = load_market_data(selected_ticker, period=period)
            close = get_close_series(data)

        if forecast_mode == "Backtest Mode":
            if len(close) <= horizon:
                st.error(
                    f"Not enough usable observations for a {horizon}-day backtest. "
                    f"{selected_ticker} returned {len(close)} close-price rows for period '{period}'. "
                    "Choose a shorter horizon or a longer historical period."
                )
                st.stop()

            train = close.iloc[:-horizon]
            test = close.iloc[-horizon:]
            forecast_index = test.index
            predictions, timesfm_error = build_forecasts(
                train,
                horizon,
                ma_window,
                include_timesfm,
            )

            if timesfm_error:
                st.warning(
                    "TimesFM forecast skipped. Baseline forecasts are still shown. "
                    f"Details: {timesfm_error}"
                )

            metrics_df = evaluate_forecasts(test.values, predictions)
            metrics_display = metrics_df.copy()
            for column in ["MAE", "RMSE", "MAPE (%)"]:
                metrics_display[column] = metrics_display[column].round(3)

            best_model = str(metrics_df.iloc[0]["Model"])
            best_mae = float(metrics_df.iloc[0]["MAE"])

            st.success(f"Backtest completed for {selected_ticker}. Best MAE: {best_model}.")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Ticker", selected_ticker)
            col2.metric("Observations", f"{len(close):,}")
            col3.metric("Test horizon", f"{horizon} days")
            col4.metric(
                "Best MAE",
                f"{best_mae:,.2f}",
                help="Lowest mean absolute error in the backtest.",
            )

            st.subheader("Forecast vs. Actual")
            st.caption(
                f"Training data ends on {train.index[-1].date()}; the test window runs from "
                f"{test.index[0].date()} to {test.index[-1].date()}."
            )
            fig = plot_historical_and_forecast(
                train.tail(252),
                predictions,
                future_index=forecast_index,
            )
            fig.add_scatter(x=test.index, y=test.values, mode="lines", name="Actual Test")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Evaluation Metrics")
            st.caption(
                "Models are sorted by MAE, with lower values indicating smaller forecast errors."
            )
            st.dataframe(metrics_display, use_container_width=True, hide_index=True)

            with st.expander("What do these metrics mean?", expanded=True):
                for metric_name, explanation in METRIC_HELP.items():
                    st.markdown(f"**{metric_name}:** {explanation}")

            forecast_df = pd.DataFrame(index=forecast_index)
            forecast_df["Actual"] = test.values

        else:
            forecast_index = make_future_index(pd.Timestamp(close.index[-1]), horizon)
            predictions, timesfm_error = build_forecasts(
                close,
                horizon,
                ma_window,
                include_timesfm,
            )

            if timesfm_error:
                st.warning(
                    "TimesFM forecast skipped. Baseline forecasts are still shown. "
                    f"Details: {timesfm_error}"
                )

            st.success(
                f"Future forecast completed for {selected_ticker}. "
                f"Predicted the next {horizon} business days."
            )

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Ticker", selected_ticker)
            col2.metric("Observations", f"{len(close):,}")
            col3.metric("Forecast horizon", f"{horizon} business days")
            col4.metric("Last close", f"{float(close.iloc[-1]):,.2f}")

            st.subheader("Future Forecast")
            st.caption(
                f"Models trained on all available close prices through {close.index[-1].date()}; "
                f"forecast window runs from {forecast_index[0].date()} "
                f"to {forecast_index[-1].date()}."
            )
            fig = plot_historical_and_forecast(
                close.tail(252),
                predictions,
                future_index=forecast_index,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.info(
                "Future Forecast Mode does not show MAE, RMSE, or MAPE because actual "
                "future prices are not available yet. Use Backtest Mode to evaluate errors."
            )

            forecast_df = pd.DataFrame(index=forecast_index)

        st.subheader("Recent Data")
        st.caption("Most recent downloaded rows from Yahoo Finance.")
        st.dataframe(data.tail(10), use_container_width=True)

        for model_name, values in predictions.items():
            forecast_df[model_name] = values

        forecast_df.index.name = "Date"
        csv = forecast_df.to_csv(index=True).encode("utf-8")
        mode_slug = "future" if forecast_mode == "Future Forecast Mode" else "backtest"
        st.download_button(
            label="Download forecast CSV",
            data=csv,
            file_name=f"{selected_ticker}_{mode_slug}_forecast.csv",
            mime="text/csv",
        )

    except ValueError as exc:
        st.error(f"Input or data issue: {exc}")
        st.info(
            "Try a preset ticker, a longer history period, or a shorter forecast horizon. "
            "If the ticker is custom, confirm it is a valid Yahoo Finance symbol."
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Forecast failed unexpectedly: {exc}")
        st.info(
            "Please rerun with a shorter horizon or without TimesFM if the model "
            "environment is unavailable."
        )
else:
    st.info("Choose settings in the sidebar and click Run forecast.")
    overview_col, metrics_col = st.columns([2, 1])
    with overview_col:
        st.subheader("Workflow")
        st.markdown(
            """
            1. Download historical market data with yfinance.
            2. Use Backtest Mode to hold out recent prices and evaluate errors.
            3. Use Future Forecast Mode to train on all close prices and predict the next business days.
            4. Export the forecast table as CSV.
            """
        )
    with metrics_col:
        st.subheader("Metrics")
        for metric_name, explanation in METRIC_HELP.items():
            st.markdown(f"**{metric_name}:** {explanation}")
