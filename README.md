# stock_sensibly
Interactive stock and crypto dashboard built with Python and my genius

## Features
- Live price data for stocks and crypto via yfinance
- 20/50-day moving averages and daily returns
- Interactive Plotly charts with adjustable time range
- SQLite caching so data isn't re-downloaded on every refresh
- Experimental next-day return model (linear regression)

## Tech stack
Python, pandas, Streamlit, Plotly, SQLite, scikit-learn

## How it works
yfinance → SQLite cache → pandas indicators → Streamlit UI

## Model results
A linear regression on the last 5 daily returns scored ~48%
direction accuracy and did not beat a "predict zero" baseline.
This is consistent with market efficiency. The model is included
as a learning exercise in time-series evaluation (chronological
train/test split, baseline comparison), not as a trading tool.

## Run locally
    python -m venv .myenv
    source .myenv/bin/activate
## Check  if git is installed locally via git--version

    git clone https://github.com/zayemo7/stock_sensibly.git
    cd stocksense
    pip install -r requirements.txt
    streamlit run ss.py

## Limitations / future work
- Cache resets on redeploy
- Add more features (volume, volatility) to the model
- Price alerts
