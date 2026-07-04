"""
Phase 1: Feature Engineering from OHLC
Builds the operand set that the GP alpha mining engine will use.

Two outputs:
1. Per-stock features (data/processed/features/{STOCK}.csv)
2. Cross-sectional panel (data/processed/panel.pkl) - all stocks aligned by date
"""

import os
import pandas as pd
import numpy as np
from warnings import filterwarnings
filterwarnings('ignore')

PROC_DIR = r"C:\Users\EV-Car\Main_Project_2\data\processed"
FEAT_DIR = os.path.join(PROC_DIR, "features")
os.makedirs(FEAT_DIR, exist_ok=True)

# Load all stocks (exclude index)
stock_files = sorted([f for f in os.listdir(PROC_DIR) if f.endswith('.csv') and f != 'NIFTY50_INDEX.csv'])
print(f"Loading {len(stock_files)} stocks...")


def compute_features(df):
    """
    Compute OHLC-derived features for a single stock.
    These become the 'operands' for GP formula construction.
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Remove duplicate dates (keep last)
    df = df.drop_duplicates(subset='Date', keep='last').reset_index(drop=True)
    
    c = df['Close']
    o = df['Open']
    h = df['High']
    l = df['Low']
    v = df['Volume'].replace(0, np.nan)
    
    # === RETURN FEATURES ===
    df['ret_1d']   = c.pct_change(1)
    df['ret_5d']   = c.pct_change(5)
    df['ret_20d']  = c.pct_change(20)
    df['ret_60d']  = c.pct_change(60)
    df['ret_120d'] = c.pct_change(120)
    df['ret_250d'] = c.pct_change(250)
    
    # === VOLATILITY FEATURES ===
    daily_ret = c.pct_change()
    df['vol_5d']   = daily_ret.rolling(5).std()
    df['vol_20d']  = daily_ret.rolling(20).std()
    df['vol_60d']  = daily_ret.rolling(60).std()
    df['vol_120d'] = daily_ret.rolling(120).std()
    
    # === RANGE / ATR FEATURES ===
    df['range_pct'] = (h - l) / c  # Daily range as % of close
    df['atr_14'] = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1).rolling(14).mean() / c
    
    # === TREND FEATURES ===
    df['sma_20']  = c.rolling(20).mean()
    df['sma_50']  = c.rolling(50).mean()
    df['sma_200'] = c.rolling(200).mean()
    df['price_to_sma20']  = c / df['sma_20'] - 1
    df['price_to_sma50']  = c / df['sma_50'] - 1
    df['price_to_sma200'] = c / df['sma_200'] - 1
    
    # === DRAWDOWN FEATURES ===
    rolling_max_60  = c.rolling(60).max()
    rolling_max_250 = c.rolling(250).max()
    df['drawdown_60d']  = (c - rolling_max_60) / rolling_max_60
    df['drawdown_250d'] = (c - rolling_max_250) / rolling_max_250
    
    # === MOMENTUM / MEAN-REVERSION FEATURES ===
    df['rsi_14'] = compute_rsi(c, 14)
    df['zscore_20'] = (c - c.rolling(20).mean()) / c.rolling(20).std()
    df['zscore_60'] = (c - c.rolling(60).mean()) / c.rolling(60).std()
    
    # === VOLUME FEATURES ===
    df['vol_ratio_5_20'] = v.rolling(5).mean() / v.rolling(20).mean()
    df['vol_change_5d']  = v.rolling(5).mean().pct_change(5)
    
    # === OPEN-CLOSE / HIGH-LOW DYNAMICS ===
    df['oc_ratio'] = (c - o) / (h - l + 1e-8)  # Body relative to range
    df['hl_ratio'] = (h - l) / c.shift(1)       # Range relative to prev close
    
    return df


def compute_rsi(series, period=14):
    """Standard RSI calculation"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))


# === PROCESS ALL STOCKS ===
all_dfs = {}
feature_cols = None

for i, f in enumerate(stock_files, 1):
    name = f.replace('.csv', '')
    df = pd.read_csv(os.path.join(PROC_DIR, f))
    df = compute_features(df)
    
    # Save per-stock features
    df.to_csv(os.path.join(FEAT_DIR, f), index=False)
    
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    all_dfs[name] = df
    
    rows = len(df)
    nans = df[feature_cols].iloc[250:].isnull().sum().sum()  # Check after warmup
    print(f"  [{i:2d}/50] {name:20s} | {rows:5d} rows | {len(feature_cols)} features | post-warmup NaNs: {nans}")


# === BUILD CROSS-SECTIONAL PANEL ===
print(f"\nBuilding cross-sectional panel...")

# Get common dates across all stocks (only dates where at least 30 stocks have data)
date_counts = pd.DataFrame({name: df.set_index('Date')['Close'] for name, df in all_dfs.items()})
date_counts.index = pd.to_datetime(date_counts.index)
valid_dates = date_counts.index[date_counts.notna().sum(axis=1) >= 30]
print(f"  Dates with >=30 stocks: {len(valid_dates)} ({valid_dates[0].date()} to {valid_dates[-1].date()})")

# Build panel: dict of {feature_name: DataFrame(dates x stocks)}
panel = {}
for feat in feature_cols:
    feat_df = pd.DataFrame({
        name: df.set_index('Date')[feat] 
        for name, df in all_dfs.items()
    })
    feat_df.index = pd.to_datetime(feat_df.index)
    feat_df = feat_df.loc[valid_dates]
    panel[feat] = feat_df

# Add cross-sectional rank features (rank each stock relative to others on each day)
for feat in ['ret_20d', 'ret_60d', 'ret_250d', 'vol_20d', 'zscore_20']:
    rank_name = f'rank_{feat}'
    panel[rank_name] = panel[feat].rank(axis=1, pct=True)
    feature_cols.append(rank_name)

# Also add forward returns as TARGET (what we're trying to predict)
close_panel = pd.DataFrame({
    name: df.set_index('Date')['Close'] 
    for name, df in all_dfs.items()
})
close_panel.index = pd.to_datetime(close_panel.index)
close_panel = close_panel.loc[valid_dates]

for horizon in [5, 20, 60]:
    panel[f'fwd_ret_{horizon}d'] = close_panel.pct_change(horizon).shift(-horizon)

# Save panel
panel_path = os.path.join(PROC_DIR, 'panel.pkl')
pd.to_pickle(panel, panel_path)

print(f"\n{'='*60}")
print(f"Feature Engineering Complete!")
print(f"{'='*60}")
print(f"  Features per stock: {len(feature_cols)}")
print(f"  Feature list: {feature_cols}")
print(f"  Per-stock CSVs: {FEAT_DIR}")
print(f"  Cross-sectional panel: {panel_path}")
print(f"  Panel shape: {len(valid_dates)} dates x {len(stock_files)} stocks x {len(feature_cols)} features")
print(f"  Forward return targets: fwd_ret_5d, fwd_ret_20d, fwd_ret_60d")
print(f"{'='*60}")
