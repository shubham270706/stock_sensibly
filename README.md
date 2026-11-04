# stock_sensibly
Interactive stock and crypto dashboard built with Python and my genius

## Features
- Live price data for stocks and crypto via yfinance
- 20/50-day moving averages and daily returns
- Interactive Plotly charts with adjustable time range
- SQLite caching so data isn't re-downloaded on every refresh
- Experimental next-day return models: Linear Regression, Ridge,
  Random Forest and Gradient Boosting, compared against baselines

## Tech stack
Python, pandas, Streamlit, Plotly, SQLite, scikit-learn

## How it works
yfinance → SQLite cache → pandas indicators → Streamlit UI

## Model results
Four models (Linear Regression, Ridge, Random Forest, Gradient Boosting)
predict the next day's return from 5 lagged returns, distance from the
20- and 50-day moving averages, 10-day volatility and volume change.

- Trained on the oldest 80% of the full price history, tested on the
  newest 20% (chronological split, no shuffling).
- Features are shifted by one day so the model only uses information
  available before the day it predicts (no look-ahead leakage).
- Compared against two baselines: "always predict up" (direction) and
  "predict 0%" (error).

**Finding:** all models perform close to chance (~50% direction
accuracy, R² near or below 0) and do not reliably beat the baselines.
This is consistent with daily returns being mostly noise. The models are
included as a learning exercise in time-series evaluation, not as a
trading tool.

## Run locally
    python -m venv .myenv
    source .myenv/bin/activate
## Check  if git is installed locally via git--version

    git clone https://github.com/zayemo7/stock_sensibly.git
    cd stocksense
    pip install -r requirements.txt
    streamlit run ss.py

## Limitations / future work
- Single 80/20 split, so metrics are noisy; next step is walk-forward
  validation
- Cache resets on redeploy
- Price alerts
