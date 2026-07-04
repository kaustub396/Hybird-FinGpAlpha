"""
Broader Market Regime Quality Test
Tests HMM regime separation across multiple Indian indices
and other global markets to find which have strong regime signals.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import yfinance as yf
import pandas as pd
import numpy as np
import os
from scipy import stats

sys.path.insert(0, r"C:\Users\EV-Car\Main_Project_2_Review1")
from regime_detector import RegimeDetector

SEP = "=" * 75
THIN = "-" * 75


def download_index(ticker, name, start="2000-01-01", end="2025-12-31"):
    """Download index data via yfinance."""
    print(f"  Downloading {name} ({ticker})...", end=" ")
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            print("EMPTY")
            return None
        # Handle MultiIndex columns from newer yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df.rename(columns={"index": "Date", "Adj Close": "Close"}, inplace=True)
        if "Date" not in df.columns:
            df.rename(columns={df.columns[0]: "Date"}, inplace=True)
        for col in ["Close", "High", "Low", "Open", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["Close"])
        print(f"OK ({len(df)} rows, {df['Date'].min().date()} to {df['Date'].max().date()})")
        return df
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def analyze_regime(df, name, train_end="2018-12-31"):
    """Quick regime quality analysis."""
    
    if df is None or len(df) < 500:
        return None
    
    # Ensure format
    df = df.copy()
    if "Date" not in df.columns:
        df = df.reset_index()
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    
    try:
        detector = RegimeDetector(n_regimes=2, random_state=42)
        train_df = df[df["Date"] <= pd.to_datetime(train_end)].copy()
        if len(train_df) < 300:
            print(f"    {name}: insufficient training data ({len(train_df)} rows)")
            return None
        detector.fit(train_df)
        
        all_labels = detector.predict(df)
        all_proba = detector.predict_proba(df)
        regime_names = detector._get_regime_names()
        
        # Test period
        test_mask = all_labels.index > pd.to_datetime(train_end)
        test_labels = all_labels[test_mask]
        test_proba = all_proba[test_mask]
        
        if len(test_labels) < 100:
            print(f"    {name}: insufficient test data ({len(test_labels)} rows)")
            return None
        
        # Daily returns
        df_indexed = df.set_index("Date")
        daily_ret = df_indexed["Close"].pct_change().dropna()
        
        common = test_labels.index.intersection(daily_ret.index)
        
        # Regime separation
        r0_rets = daily_ret.loc[common][test_labels.loc[common] == 0]
        r1_rets = daily_ret.loc[common][test_labels.loc[common] == 1]
        
        if len(r0_rets) < 10 or len(r1_rets) < 10:
            return None
        
        r0_ann_ret = r0_rets.mean() * 252 * 100
        r1_ann_ret = r1_rets.mean() * 252 * 100
        r0_vol = r0_rets.std() * np.sqrt(252) * 100
        r1_vol = r1_rets.std() * np.sqrt(252) * 100
        
        separation = abs(r0_ann_ret - r1_ann_ret)
        t_stat, p_val = stats.ttest_ind(r0_rets, r1_rets)
        
        # Volatility ratio
        vol_ratio = max(r0_vol, r1_vol) / min(r0_vol, r1_vol)
        
        # Transition rate
        transitions = (test_labels.diff().abs() > 0).sum()
        total_days = len(test_labels)
        transition_rate = transitions / total_days * 100
        avg_regime_len = total_days / max(transitions, 1)
        
        # Confidence
        max_prob = test_proba.max(axis=1)
        avg_confidence = max_prob.mean()
        decisive_pct = (max_prob >= 0.8).mean() * 100
        
        # Regime distribution
        r0_pct = (test_labels == 0).mean() * 100
        r1_pct = (test_labels == 1).mean() * 100
        
        # Identify which is Bull/Bear by volatility
        low_vol_id = 0 if r0_vol < r1_vol else 1
        high_vol_id = 1 - low_vol_id
        
        low_vol = min(r0_vol, r1_vol)
        high_vol = max(r0_vol, r1_vol)
        low_vol_ret = r0_ann_ret if low_vol_id == 0 else r1_ann_ret
        high_vol_ret = r1_ann_ret if low_vol_id == 0 else r0_ann_ret
        
        return {
            "name": name,
            "separation": separation,
            "p_value": p_val,
            "vol_ratio": vol_ratio,
            "low_vol": low_vol,
            "high_vol": high_vol,
            "low_vol_ret": low_vol_ret,
            "high_vol_ret": high_vol_ret,
            "transitions": transitions,
            "avg_regime_len": avg_regime_len,
            "transition_rate": transition_rate,
            "avg_confidence": avg_confidence,
            "decisive_pct": decisive_pct,
            "test_days": total_days,
        }
    except Exception as e:
        print(f"    {name}: ERROR - {e}")
        return None


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    
    print(SEP)
    print("  GLOBAL REGIME QUALITY SCAN")
    print("  Testing HMM regime separation across markets")
    print(SEP)
    
    # Define indices to test
    indices = {
        # ---- Indian Markets ----
        "^NSEI": "NIFTY-50 (India)",
        "^BSESN": "SENSEX/BSE-30 (India)",
        "^NSEBANK": "NIFTY Bank (India)",
        "^CNXIT": "NIFTY IT (India)",
        "^CNXPHARMA": "NIFTY Pharma (India)",
        # ---- US Markets ----
        "^GSPC": "S&P 500 (US)",
        "^DJI": "Dow Jones (US)",
        "^IXIC": "NASDAQ Composite (US)",
        "^RUT": "Russell 2000 (US)",
        # ---- Other Developed ----
        "^FTSE": "FTSE 100 (UK)",
        "^GDAXI": "DAX (Germany)",
        "^N225": "Nikkei 225 (Japan)",
        # ---- Emerging ----
        "^HSI": "Hang Seng (Hong Kong)",
        "^BVSP": "Bovespa (Brazil)",
        "000001.SS": "SSE Composite (China)",
    }
    
    # Download all
    print("\n  Downloading index data...")
    print(THIN)
    data = {}
    for ticker, name in indices.items():
        df = download_index(ticker, name)
        if df is not None:
            data[ticker] = (name, df)
    
    # Analyze regimes
    print(f"\n  Analyzing regime quality...")
    print(THIN)
    results = []
    for ticker, (name, df) in data.items():
        print(f"\n  Analyzing: {name}")
        result = analyze_regime(df, name)
        if result is not None:
            results.append(result)
            print(f"    Separation: {result['separation']:.1f}% | "
                  f"p={result['p_value']:.4f} | "
                  f"Vol ratio: {result['vol_ratio']:.1f}x | "
                  f"Transitions: {result['transitions']} | "
                  f"Avg regime: {result['avg_regime_len']:.0f} days")
    
    # ================================================================
    # RESULTS TABLE
    # ================================================================
    print(f"\n\n{SEP}")
    print("  REGIME QUALITY RANKING (sorted by separation)")
    print(SEP)
    
    results.sort(key=lambda x: x["separation"], reverse=True)
    
    print(f"\n  {'Market':<28} {'Sep.':<8} {'p-val':<8} {'VolRat':<8} "
          f"{'Trans':<7} {'AvgLen':<8} {'LowV%':<8} {'HiV%':<8} {'LowVR%':<8} {'HiVR%':<8}")
    print(f"  {'-'*110}")
    
    for r in results:
        sig = "***" if r["p_value"] < 0.01 else "** " if r["p_value"] < 0.05 else "*  " if r["p_value"] < 0.10 else "   "
        print(f"  {r['name']:<28} {r['separation']:>6.1f}% {r['p_value']:>7.4f}{sig} "
              f"{r['vol_ratio']:>6.1f}x {r['transitions']:>5} "
              f"{r['avg_regime_len']:>6.0f}d  {r['low_vol']:>6.1f}  {r['high_vol']:>6.1f}  "
              f"{r['low_vol_ret']:>6.1f}  {r['high_vol_ret']:>6.1f}")
    
    # ================================================================
    # CLASSIFICATION
    # ================================================================
    print(f"\n\n{SEP}")
    print("  MARKET CLASSIFICATION BY REGIME STRENGTH")
    print(SEP)
    
    strong = [r for r in results if r["separation"] > 20 and r["p_value"] < 0.20]
    moderate = [r for r in results if 10 <= r["separation"] <= 20 or (r["separation"] > 20 and r["p_value"] >= 0.20)]
    weak = [r for r in results if r["separation"] < 10]
    
    print(f"\n  STRONG regime signals (sep > 20%, p < 0.20) -- HMM-GPa should help:")
    for r in strong:
        print(f"    - {r['name']}: {r['separation']:.1f}% gap, p={r['p_value']:.4f}, "
              f"vol ratio {r['vol_ratio']:.1f}x")
    
    print(f"\n  MODERATE regime signals (10-20% gap or high separation but not significant):")
    for r in moderate:
        print(f"    - {r['name']}: {r['separation']:.1f}% gap, p={r['p_value']:.4f}")
    
    print(f"\n  WEAK regime signals (sep < 10%) -- Vanilla GP preferred:")
    for r in weak:
        print(f"    - {r['name']}: {r['separation']:.1f}% gap, p={r['p_value']:.4f}")
    
    print(f"\n  Done. Tested {len(results)} markets.")
