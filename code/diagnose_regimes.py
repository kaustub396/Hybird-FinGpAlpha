"""
Regime Diagnostic: Why does HMM-GPa underperform Vanilla GP on S&P 500?

Tests:
  1. Regime Separation Quality  - Are Bull/Bear regimes clearly distinct?
  2. Regime Probability Distribution - Are probabilities decisive (near 0/1) or uncertain (near 0.5)?
  3. Regime Transition Frequency - How often does regime flip? (noisy = bad)
  4. Per-Regime Formula Performance - Do regime-specific formulas work in their own regime?
  5. Combination Method Analysis - Is soft-weighting the problem?
  6. Cross-Market Comparison - Compare S&P 500 vs NIFTY-50 regime quality
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\EV-Car\Main_Project_2_Review1")
from regime_detector import RegimeDetector

SEP = "=" * 70
THIN = "-" * 70


# ================================================================
# HELPER: Train HMM and get regime stats for any index
# ================================================================
def analyze_regime_quality(index_df, name, train_end="2018-12-31"):
    """Full regime quality analysis for a given index."""
    
    print(f"\n{SEP}")
    print(f"  REGIME ANALYSIS: {name}")
    print(SEP)
    
    # Ensure proper format
    df = index_df.copy()
    if "Date" not in df.columns:
        df = df.reset_index()
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    for col in ["Close", "High", "Low", "Open", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Close"])
    
    # Train HMM
    detector = RegimeDetector(n_regimes=2, random_state=42)
    train_df = df[df["Date"] <= pd.to_datetime(train_end)].copy()
    detector.fit(train_df)
    
    # Predict on full data
    all_labels = detector.predict(df)
    all_proba = detector.predict_proba(df)
    
    regime_names = detector._get_regime_names()
    
    # Split into train/test
    train_mask = all_labels.index <= pd.to_datetime(train_end)
    test_mask = ~train_mask
    
    train_labels = all_labels[train_mask]
    test_labels = all_labels[test_mask]
    train_proba = all_proba[train_mask]
    test_proba = all_proba[test_mask]
    
    # ---- TEST 1: Regime Separation Quality ----
    print(f"\n  TEST 1: Regime Separation Quality")
    print(THIN)
    
    # Get daily returns for each regime
    df_indexed = df.set_index("Date")
    daily_ret = df_indexed["Close"].pct_change().dropna()
    
    for period_name, labels in [("Train", train_labels), ("Test", test_labels)]:
        common = labels.index.intersection(daily_ret.index)
        for rid in range(2):
            rname = regime_names[rid]
            mask = labels.loc[common] == rid
            rets = daily_ret.loc[common][mask]
            n = len(rets)
            mean_ret = rets.mean() * 252  # annualized
            vol = rets.std() * np.sqrt(252)
            print(f"    {period_name} {rname}: n={n:,} days, "
                  f"Ann.Ret={mean_ret*100:.1f}%, Vol={vol*100:.1f}%")
    
    # Statistical separation: difference in means
    common_test = test_labels.index.intersection(daily_ret.index)
    bull_rets = daily_ret.loc[common_test][test_labels.loc[common_test] == 0]
    bear_rets = daily_ret.loc[common_test][test_labels.loc[common_test] == 1]
    
    if len(bull_rets) > 10 and len(bear_rets) > 10:
        from scipy import stats
        t_stat, p_val = stats.ttest_ind(bull_rets, bear_rets)
        separation = abs(bull_rets.mean() - bear_rets.mean()) * 252 * 100
        print(f"    Regime separation (test): {separation:.1f}% ann. return gap, "
              f"t={t_stat:.2f}, p={p_val:.4f}")
    
    # ---- TEST 2: Probability Distribution ----
    print(f"\n  TEST 2: Regime Probability Distribution (Test Period)")
    print(THIN)
    
    # Max probability = confidence
    max_prob = test_proba.max(axis=1)
    
    bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    print(f"    Confidence Distribution (max(P(Bull), P(Bear))):")
    for lo, hi in bins:
        n = ((max_prob >= lo) & (max_prob < hi)).sum()
        pct = n / len(max_prob) * 100
        bar = "#" * int(pct / 2)
        print(f"      {lo:.1f}-{hi:.1f}: {n:5d} ({pct:5.1f}%) {bar}")
    # 1.0 exactly
    n = (max_prob >= 1.0 - 1e-10).sum()
    pct = n / len(max_prob) * 100
    print(f"      ~1.00 : {n:5d} ({pct:5.1f}%)")
    
    avg_confidence = max_prob.mean()
    decisive_pct = (max_prob >= 0.8).mean() * 100
    uncertain_pct = (max_prob < 0.6).mean() * 100
    
    print(f"\n    Avg confidence: {avg_confidence:.4f}")
    print(f"    Decisive (>0.8): {decisive_pct:.1f}%")
    print(f"    Uncertain (<0.6): {uncertain_pct:.1f}%")
    
    # ---- TEST 3: Regime Transition Frequency ----
    print(f"\n  TEST 3: Regime Transition Frequency (Test Period)")
    print(THIN)
    
    transitions = (test_labels.diff().abs() > 0).sum()
    total_days = len(test_labels)
    transition_rate = transitions / total_days * 100
    avg_regime_length = total_days / max(transitions, 1)
    
    print(f"    Total test days: {total_days}")
    print(f"    Regime transitions: {transitions}")
    print(f"    Transition rate: {transition_rate:.2f}% of days")
    print(f"    Avg regime duration: {avg_regime_length:.0f} days")
    
    # Distribution of regime lengths
    regime_runs = []
    current_regime = test_labels.iloc[0]
    current_len = 1
    for i in range(1, len(test_labels)):
        if test_labels.iloc[i] == current_regime:
            current_len += 1
        else:
            regime_runs.append(current_len)
            current_regime = test_labels.iloc[i]
            current_len = 1
    regime_runs.append(current_len)
    
    runs_arr = np.array(regime_runs)
    print(f"    Regime run lengths: min={runs_arr.min()}, "
          f"median={np.median(runs_arr):.0f}, "
          f"max={runs_arr.max()}, "
          f"mean={runs_arr.mean():.1f}")
    short_runs = (runs_arr <= 5).sum()
    print(f"    Very short runs (<=5 days): {short_runs} ({short_runs/len(runs_arr)*100:.0f}%)")
    
    # ---- TEST 4: Regime distribution in test ----
    print(f"\n  TEST 4: Regime Distribution")
    print(THIN)
    for period_name, labels in [("Train", train_labels), ("Test", test_labels)]:
        for rid in range(2):
            rname = regime_names[rid]
            n = (labels == rid).sum()
            pct = n / len(labels) * 100
            print(f"    {period_name} {rname}: {n:,} days ({pct:.1f}%)")
    
    return {
        "avg_confidence": avg_confidence,
        "decisive_pct": decisive_pct,
        "uncertain_pct": uncertain_pct,
        "transition_rate": transition_rate,
        "avg_regime_length": avg_regime_length,
        "short_runs_pct": short_runs / len(runs_arr) * 100,
        "separation": separation if len(bull_rets) > 10 and len(bear_rets) > 10 else 0,
        "p_value": p_val if len(bull_rets) > 10 and len(bear_rets) > 10 else 1,
    }


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    
    print(SEP)
    print("  REGIME DIAGNOSTIC: S&P 500 vs NIFTY-50")
    print(f"  Why does HMM-GPa underperform Vanilla GP on S&P 500?")
    print(SEP)
    
    # ---- Load S&P 500 index ----
    sp_path = r"C:\Users\EV-Car\Main_Project_2_Review1\data\raw_sp500\SP500_INDEX.csv"
    sp_df = pd.read_csv(sp_path, header=0, skiprows=[1, 2])
    sp_df.rename(columns={"Price": "Date"}, inplace=True)
    
    # ---- Load NIFTY-50 index ----
    # Check both possible locations
    nifty_path = r"C:\Users\EV-Car\Main_Project_2\data\raw\NIFTY50_INDEX.csv"
    if not os.path.exists(nifty_path):
        nifty_path = r"C:\Users\EV-Car\Main_Project_2_Review1\data\raw\NIFTY50_INDEX.csv"
    
    if os.path.exists(nifty_path):
        nifty_df = pd.read_csv(nifty_path)
        # Handle potential MultiIndex header
        if nifty_df.iloc[0].astype(str).str.contains("Ticker|Date").any():
            nifty_df = pd.read_csv(nifty_path, header=0, skiprows=[1, 2])
            nifty_df.rename(columns={nifty_df.columns[0]: "Date"}, inplace=True)
    else:
        nifty_df = None
        print("  WARNING: NIFTY-50 index data not found, skipping comparison")
    
    # ---- Run diagnostics ----
    sp_stats = analyze_regime_quality(sp_df, "S&P 500 (^GSPC)")
    
    if nifty_df is not None:
        nifty_stats = analyze_regime_quality(nifty_df, "NIFTY-50")
    
    # ---- COMPARISON SUMMARY ----
    print(f"\n\n{SEP}")
    print("  DIAGNOSTIC SUMMARY: S&P 500 vs NIFTY-50")
    print(SEP)
    
    if nifty_df is not None:
        headers = ["Metric", "S&P 500", "NIFTY-50", "Verdict"]
        rows = [
            ["Avg Confidence",
             f"{sp_stats['avg_confidence']:.3f}",
             f"{nifty_stats['avg_confidence']:.3f}",
             "Higher = cleaner regimes"],
            ["Decisive (>0.8)",
             f"{sp_stats['decisive_pct']:.1f}%",
             f"{nifty_stats['decisive_pct']:.1f}%",
             "Higher = better for HMM-GP"],
            ["Uncertain (<0.6)",
             f"{sp_stats['uncertain_pct']:.1f}%",
             f"{nifty_stats['uncertain_pct']:.1f}%",
             "Lower = better"],
            ["Transition Rate",
             f"{sp_stats['transition_rate']:.2f}%",
             f"{nifty_stats['transition_rate']:.2f}%",
             "Lower = more stable regimes"],
            ["Avg Regime Length",
             f"{sp_stats['avg_regime_length']:.0f} days",
             f"{nifty_stats['avg_regime_length']:.0f} days",
             "Longer = better"],
            ["Short Runs (<=5d)",
             f"{sp_stats['short_runs_pct']:.0f}%",
             f"{nifty_stats['short_runs_pct']:.0f}%",
             "Lower = less noise"],
            ["Regime Separation",
             f"{sp_stats['separation']:.1f}% gap",
             f"{nifty_stats['separation']:.1f}% gap",
             "Larger = clearer regimes"],
            ["Separation p-value",
             f"{sp_stats['p_value']:.4f}",
             f"{nifty_stats['p_value']:.4f}",
             "Lower = more significant"],
        ]
        
        # Print as table
        col_widths = [max(len(r[j]) for r in [headers] + rows) + 2 for j in range(4)]
        header_str = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        print(f"  {header_str}")
        print(f"  {'-' * len(header_str)}")
        for row in rows:
            row_str = " | ".join(r.ljust(w) for r, w in zip(row, col_widths))
            print(f"  {row_str}")
    
    # ---- DIAGNOSIS ----
    print(f"\n{SEP}")
    print("  ROOT CAUSE DIAGNOSIS")
    print(SEP)
    
    issues = []
    if sp_stats["uncertain_pct"] > 30:
        issues.append("HIGH UNCERTAINTY: S&P 500 regime probabilities are often near 0.5 "
                      "-> soft combination dilutes signal")
    if sp_stats["transition_rate"] > 2:
        issues.append("HIGH TRANSITION RATE: Regimes flip too often "
                      "-> GP formulas can't specialize, noise in combination")
    if sp_stats["avg_regime_length"] < 30:
        issues.append("SHORT REGIME DURATIONS: Avg regime too short for GP to exploit")
    if sp_stats["short_runs_pct"] > 30:
        issues.append("TOO MANY SHORT RUNS: Frequent 1-5 day regime flips = noise")
    if sp_stats["separation"] < 10:
        issues.append("WEAK SEPARATION: Bull/Bear return gap is small "
                      "-> regimes don't carry distinct alpha opportunities")
    if sp_stats["p_value"] > 0.05:
        issues.append("NOT SIGNIFICANT: Regime separation is not statistically significant")
    
    if not issues:
        issues.append("No obvious issues found - deeper investigation needed")
    
    for i, issue in enumerate(issues):
        print(f"  [{i+1}] {issue}")
    
    print(f"\n  Done.")
