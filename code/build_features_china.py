"""
Cross-Market Generalisation -- Feature Engineering for Chinese A-Shares
Mirrors build_features.py exactly -- same 32 features, same logic.

Reads  : data/raw_china/*.csv
Outputs: data/processed_china/features/{STOCK}.csv
         data/processed_china/panel_china.pkl
"""

import os
import pandas as pd
import numpy as np
from warnings import filterwarnings
filterwarnings('ignore')

# -- Paths --------------------------------------------------------------------
BASE_DIR = r"C:\Users\EV-Car\Main_Project_2_Review1"
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw_china")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed_china")
FEAT_DIR = os.path.join(PROC_DIR, "features")
os.makedirs(FEAT_DIR, exist_ok=True)

# Exclude the index file -- stocks only
stock_files = sorted([
    f for f in os.listdir(RAW_DIR)
    if f.endswith('.csv') and f != 'SSE_INDEX.csv'
])
print(f"Loading {len(stock_files)} Chinese A-share stocks from {RAW_DIR}...")


# -- Feature Engineering (identical to build_features.py) ---------------------
def compute_rsi(series, period=14):
    """Standard RSI calculation."""
    delta = series.diff()
    gain  = delta.where(delta > 0, 0).rolling(period).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs    = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the same 32 OHLCV-derived features used for the NIFTY-50 paper.
    Feature set is market-agnostic -- no country-specific inputs.
    """
    df = df.copy()

    # Normalise column names
    df.columns = [c.strip().title().replace(" ", "_") for c in df.columns]
    if "Adj_Close" in df.columns and "Close" not in df.columns:
        df.rename(columns={"Adj_Close": "Close"}, inplace=True)
    if "Date" not in df.columns:
        df = df.reset_index()
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates(subset="Date", keep="last").reset_index(drop=True)

    c = df["Close"]
    o = df["Open"]
    h = df["High"]
    l = df["Low"]
    v = df["Volume"].replace(0, np.nan) if "Volume" in df.columns else pd.Series(np.nan, index=df.index)

    # === RETURN FEATURES ===
    df["ret_1d"]   = c.pct_change(1)
    df["ret_5d"]   = c.pct_change(5)
    df["ret_20d"]  = c.pct_change(20)
    df["ret_60d"]  = c.pct_change(60)
    df["ret_120d"] = c.pct_change(120)
    df["ret_250d"] = c.pct_change(250)

    # === VOLATILITY FEATURES ===
    daily_ret      = c.pct_change()
    df["vol_5d"]   = daily_ret.rolling(5).std()
    df["vol_20d"]  = daily_ret.rolling(20).std()
    df["vol_60d"]  = daily_ret.rolling(60).std()
    df["vol_120d"] = daily_ret.rolling(120).std()

    # === RANGE / ATR FEATURES ===
    df["range_pct"] = (h - l) / c
    df["atr_14"]    = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1).rolling(14).mean() / c

    # === TREND FEATURES ===
    df["sma_20"]  = c.rolling(20).mean()
    df["sma_50"]  = c.rolling(50).mean()
    df["sma_200"] = c.rolling(200).mean()
    df["price_to_sma20"]  = c / df["sma_20"]  - 1
    df["price_to_sma50"]  = c / df["sma_50"]  - 1
    df["price_to_sma200"] = c / df["sma_200"] - 1

    # === DRAWDOWN FEATURES ===
    df["drawdown_60d"]  = (c - c.rolling(60).max())  / c.rolling(60).max()
    df["drawdown_250d"] = (c - c.rolling(250).max()) / c.rolling(250).max()

    # === MOMENTUM / MEAN-REVERSION FEATURES ===
    df["rsi_14"]    = compute_rsi(c, 14)
    df["zscore_20"] = (c - c.rolling(20).mean()) / c.rolling(20).std()
    df["zscore_60"] = (c - c.rolling(60).mean()) / c.rolling(60).std()

    # === VOLUME FEATURES ===
    df["vol_ratio_5_20"] = v.rolling(5).mean() / v.rolling(20).mean()
    df["vol_change_5d"]  = v.rolling(5).mean().pct_change(5)

    # === OPEN-CLOSE / HIGH-LOW DYNAMICS ===
    df["oc_ratio"] = (c - o) / (h - l + 1e-8)
    df["hl_ratio"] = (h - l) / c.shift(1)

    return df


# -- Process all stocks -------------------------------------------------------
all_dfs      = {}
feature_cols = None

for i, fname in enumerate(stock_files, 1):
    name = fname.replace(".csv", "")
    raw  = pd.read_csv(os.path.join(RAW_DIR, fname))
    df   = compute_features(raw)

    # Save per-stock feature file
    df.to_csv(os.path.join(FEAT_DIR, fname), index=False)

    if feature_cols is None:
        exclude     = {"Date", "Open", "High", "Low", "Close", "Volume",
                       "Adj_Close", "Dividends", "Stock_Splits"}
        feature_cols = [col for col in df.columns
                        if col not in exclude and not col.startswith("Unnamed")]

    all_dfs[name] = df
    post_warmup_nans = df[feature_cols].iloc[250:].isnull().sum().sum()
    print(f"  [{i:2d}/{len(stock_files)}] {name:16s} | "
          f"{len(df):5d} rows | {len(feature_cols)} features | "
          f"post-warmup NaNs: {post_warmup_nans}")


# -- Build cross-sectional panel -----------------------------------------------
print(f"\nBuilding Chinese A-share cross-sectional panel...")

date_counts = pd.DataFrame({
    name: df.set_index("Date")["Close"]
    for name, df in all_dfs.items()
})
date_counts.index = pd.to_datetime(date_counts.index)
# Require at least 25 stocks to have data on a given day
valid_dates = date_counts.index[date_counts.notna().sum(axis=1) >= 25]
print(f"  Valid dates (>=25 stocks): {len(valid_dates)}  "
      f"({valid_dates[0].date()} -> {valid_dates[-1].date()})")

panel = {}
for feat in feature_cols:
    feat_df = pd.DataFrame({
        name: df.set_index("Date")[feat]
        for name, df in all_dfs.items()
        if feat in df.columns
    })
    feat_df.index = pd.to_datetime(feat_df.index)
    panel[feat]   = feat_df.loc[valid_dates]

# Cross-sectional rank features
for feat in ["ret_20d", "ret_60d", "ret_250d", "vol_20d", "zscore_20"]:
    rank_name        = f"rank_{feat}"
    panel[rank_name] = panel[feat].rank(axis=1, pct=True)
    feature_cols.append(rank_name)

# Forward return targets
close_panel = pd.DataFrame({
    name: df.set_index("Date")["Close"]
    for name, df in all_dfs.items()
})
close_panel.index = pd.to_datetime(close_panel.index)
close_panel       = close_panel.loc[valid_dates]

for horizon in [5, 20, 60]:
    panel[f"fwd_ret_{horizon}d"] = (
        close_panel.pct_change(horizon).shift(-horizon)
    )

# Save panel
panel_path = os.path.join(PROC_DIR, "panel_china.pkl")
pd.to_pickle(panel, panel_path)

print(f"\n{'=' * 65}")
print("  Chinese A-Share Feature Engineering Complete!")
print(f"{'=' * 65}")
print(f"  Stocks processed       : {len(all_dfs)}")
print(f"  Features per stock     : {len(feature_cols)}")
print(f"  Valid dates            : {len(valid_dates)}")
print(f"  Per-stock CSVs         : {FEAT_DIR}")
print(f"  Cross-sectional panel  : {panel_path}")
print(f"{'=' * 65}")
