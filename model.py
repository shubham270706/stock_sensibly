import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LAGS = 5
MIN_ROWS = 20  # below this the train/test split is too small to mean anything


def make_models():
    """Fresh, untrained models. Add or remove entries to change the comparison."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=5, min_samples_leaf=20,
            random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, max_depth=2, learning_rate=0.05,
            subsample=0.8, random_state=42),
    }


def raw_features(df, lags=LAGS):
    """Features describing day t, using only data up to and including day t."""
    close = df["Close"]
    ret = close.pct_change()

    f = pd.DataFrame(index=df.index)
    for i in range(lags):
        f[f"ret_{i}"] = ret.shift(i)                 # today's return, yesterday's, ...
    f["dist_ma20"] = close / close.rolling(20).mean() - 1
    f["dist_ma50"] = close / close.rolling(50).mean() - 1
    f["vol_10"] = ret.rolling(10).std()              # recent volatility
    f["vol_chg"] = df["Volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    return f, ret


def build_dataset(df, lags=LAGS):
    """
    Row t of X = features known at the END of day t-1; y = return on day t.
    The shift(1) is what prevents look-ahead leakage.
    Also returns `latest`: features as of the last day, used to predict tomorrow.
    """
    feats, ret = raw_features(df, lags)
    data = feats.shift(1)
    data["target"] = ret
    data = data.dropna()
    latest = feats.iloc[[-1]].fillna(0)
    return data.drop(columns="target"), data["target"], latest


def train_and_evaluate(df, test_frac=0.2):
    """
    Trains every model on the oldest 80% of the history, tests on the newest 20%.
    Pass the FULL history here (not a zoomed-in period) so there is enough data.
    """
    X, y, latest = build_dataset(df)
    if len(X) < MIN_ROWS:
        raise ValueError(f"Need at least {MIN_ROWS} usable rows, got {len(X)}.")

    split = int(len(X) * (1 - test_frac))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    baseline_dir = (y_test > 0).mean()        # "always predict up"
    baseline_mae = y_test.abs().mean()        # "always predict 0% return"

    rows, test_preds = [], pd.DataFrame({"actual": y_test})
    for name, model in make_models().items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        test_preds[name] = preds

        mae = float(np.mean(np.abs(preds - y_test)))
        rows.append({
            "Model": name,
            "Direction acc (%)": ((preds > 0) == (y_test > 0)).mean() * 100,
            "MAE (%)": mae * 100,
            "RMSE (%)": np.sqrt(mean_squared_error(y_test, preds)) * 100,
            "R2": r2_score(y_test, preds),
            "Beats zero baseline": mae < baseline_mae,
            "Next-day return (%)": float(model.predict(latest)[0]) * 100,
        })

    results = pd.DataFrame(rows).set_index("Model")
    return {
        "results": results,
        "baseline_direction": baseline_dir * 100,
        "baseline_mae": baseline_mae * 100,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_start": X_test.index[0],
        "test_preds": test_preds,   # actual vs predicted per day, handy for backtesting
        "best_model": results["RMSE (%)"].idxmin(),
    }
