import streamlit as sl
import plotly.graph_objects as go
from data import get_data
from anal import add_indicators
from model import train_and_evaluate

sl.title("StockSense")

from tickers import TICKERS

choice = sl.selectbox("Stock / Crypto", list(TICKERS.keys()))
ticker = TICKERS[choice]
period = sl.selectbox("Period",["1mo","3mo","6mo","1y","2y","5y"],index=3)

from datetime import datetime, timedelta

PERIOD_DAYS = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}



@sl.cache_data(ttl=3600, show_spinner="Training models...")
def run_models(ticker):
    return train_and_evaluate(get_data(ticker)) 

full_df = get_data(ticker)

if full_df.empty:
    sl.error("No data found. Check the ticker symbol.")
else:
    df = add_indicators(full_df)
    df = df[df.index >= datetime.now() - timedelta(days=PERIOD_DAYS[period])]
    # ...chart code stays the same
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close"))
    fig.add_trace(go.Scatter(x=df.index, y=df["avg_20"], name="Average:20 Days"))
    fig.add_trace(go.Scatter(x=df.index, y=df["avg_50"], name="Average:50 Days"))
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=df.index, y=df["return"] * 100, name="Daily return (%)"))
    fig2.update_layout(title="Daily return (%)", xaxis_title="Date", yaxis_title="%")
    sl.plotly_chart(fig2, use_container_width=True)
    fig.update_layout(title=f"{ticker} price", xaxis_title="Date", yaxis_title="Price")

    sl.plotly_chart(fig, use_container_width=True)
    sl.dataframe(df.tail(10))

    sl.subheader("Model comparison (experimental)")
    try:
        result = run_models(ticker)
        sl.caption(f"Trained on {result['n_train']} days, tested on the latest "
                   f"{result['n_test']} days (from {result['test_start'].date()}).")
        sl.dataframe(result["results"].style.format({
            "Direction acc (%)": "{:.1f}", "MAE (%)": "{:.3f}",
            "RMSE (%)": "{:.3f}", "R2": "{:.3f}", "Next-day return (%)": "{:.2f}"}))
        c1, c2 = sl.columns(2)
        c1.metric("Baseline: always predict 'up'", f"{result['baseline_direction']:.1f}%")
        c2.metric("Baseline MAE (predict 0%)", f"{result['baseline_mae']:.3f}%")
    except ValueError as e:
        sl.warning(str(e))
    sl.caption("Learning project, not financial advice.")
