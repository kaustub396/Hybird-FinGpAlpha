"""
Phase 2: Baseline Alpha Strategies

Implements standard quantitative factors as baselines.
These are fixed-formula alphas (no learning) that serve as comparison
benchmarks for the GP-discovered alphas in Phase 3.

Each baseline function returns a DataFrame(dates × stocks) of alpha scores.
Higher score = more bullish prediction.

Baselines implemented:
1. Momentum (12-1 month)
2. Short-term Reversal (1-month return, inverted)
3. Mean Reversion (z-score based)
4. Low Volatility
5. Trend (price relative to 200-DMA)
6. Combined Simple (equal-weight combination)
"""

import pandas as pd
import numpy as np
import os
import pickle

PROC_DIR = r"C:\Users\EV-Car\Main_Project_2\data\processed"


def load_panel():
    """Load the cross-sectional panel."""
    panel_path = os.path.join(PROC_DIR, 'panel.pkl')
    with open(panel_path, 'rb') as f:
        panel = pickle.load(f)
    print(f"Panel loaded: {len(list(panel.values())[0])} dates × {len(list(panel.values())[0].columns)} stocks")
    return panel


# ============================================================
# BASELINE 1: Momentum (12-1 month)
# ============================================================
# Classic Jegadeesh & Titman (1993) momentum.
# Signal = cumulative return from month -12 to month -1 (skip last month).
# Rationale: Winners keep winning over 3-12 month horizons.

def momentum_alpha(panel):
    """
    12-1 month momentum: ret_250d minus ret_20d.
    Skipping the most recent month avoids short-term reversal contamination.
    """
    ret_250 = panel['ret_250d']
    ret_20 = panel['ret_20d']
    alpha = ret_250 - ret_20  # 12-month return minus 1-month return
    return alpha


# ============================================================
# BASELINE 2: Short-Term Reversal
# ============================================================
# Stocks that dropped recently tend to bounce back.
# Signal = -1 × (1-month return).
# Rationale: Overreaction in the short term.

def reversal_alpha(panel):
    """
    Short-term reversal: negative of 20-day return.
    Stocks that fell recently are expected to recover.
    """
    alpha = -panel['ret_20d']
    return alpha


# ============================================================
# BASELINE 3: Mean Reversion (Z-Score)
# ============================================================
# Price deviation from 20-day rolling mean, normalized by rolling std.
# Signal = -z_score (negative because extreme positive z → expected to revert down).
# Rationale: Prices oscillate around a mean.

def mean_reversion_alpha(panel):
    """
    Z-score mean reversion: negative of 20-day z-score.
    Extreme positive z-score → overextended → short.
    Extreme negative z-score → oversold → long.
    """
    alpha = -panel['zscore_20']
    return alpha


# ============================================================
# BASELINE 4: Low Volatility
# ============================================================
# Low-volatility anomaly: less volatile stocks earn higher risk-adjusted returns.
# Signal = -volatility (negative because lower vol = higher score).
# Rationale: Documented anomaly across markets and decades.

def low_volatility_alpha(panel):
    """
    Low-volatility factor: negative of 60-day realized volatility.
    Lower volatility stocks get higher alpha scores.
    """
    alpha = -panel['vol_60d']
    return alpha


# ============================================================
# BASELINE 5: Trend (Price vs 200-DMA)
# ============================================================
# Stocks above their 200-day moving average are in uptrend.
# Signal = price / SMA_200 - 1.
# Rationale: Trend persistence is one of the strongest market effects.

def trend_alpha(panel):
    """
    Trend factor: price relative to 200-day moving average.
    Stocks above SMA-200 are scored higher (uptrend).
    """
    alpha = panel['price_to_sma200']
    return alpha


# ============================================================
# BASELINE 6: Combined Simple Alpha
# ============================================================
# Equal-weight combination of all individual baselines.
# Each baseline is z-scored cross-sectionally before combination
# to ensure equal contribution.

def combined_alpha(panel):
    """
    Equal-weight combination of 5 baseline alphas.
    Each is cross-sectionally z-scored before averaging.
    """
    alphas = {
        'momentum': momentum_alpha(panel),
        'reversal': reversal_alpha(panel),
        'mean_rev': mean_reversion_alpha(panel),
        'low_vol': low_volatility_alpha(panel),
        'trend': trend_alpha(panel),
    }
    
    # Cross-sectional z-score each alpha (rank stocks within each day)
    z_scored = {}
    for name, alpha in alphas.items():
        row_mean = alpha.mean(axis=1)
        row_std = alpha.std(axis=1)
        z_scored[name] = alpha.sub(row_mean, axis=0).div(row_std.replace(0, np.nan), axis=0)
    
    # Equal-weight average
    combined = sum(z_scored.values()) / len(z_scored)
    return combined


# ============================================================
# RUN ALL BASELINES
# ============================================================

def run_all_baselines(panel, evaluator, target='fwd_ret_20d', holding_period=20):
    """
    Run and evaluate all baseline strategies.
    
    Parameters
    ----------
    panel : dict
        Cross-sectional panel.
    evaluator : AlphaEvaluator
        Evaluation engine instance.
    target : str
        Forward return target.
    holding_period : int
        Rebalance frequency in trading days.
    
    Returns
    -------
    dict : {strategy_name: evaluation_results}
    """
    strategies = {
        'Momentum (12-1M)': momentum_alpha,
        'Short-Term Reversal': reversal_alpha,
        'Mean Reversion (Z-Score)': mean_reversion_alpha,
        'Low Volatility': low_volatility_alpha,
        'Trend (200-DMA)': trend_alpha,
        'Combined (Equal Wt)': combined_alpha,
    }
    
    all_results = {}
    
    for name, func in strategies.items():
        print(f"\n{'#'*55}")
        print(f"  EVALUATING: {name}")
        print(f"  Target: {target} | Holding: {holding_period}d")
        print(f"{'#'*55}")
        
        alpha_scores = func(panel)
        results = evaluator.evaluate(
            alpha_scores, target=target, holding_period=holding_period, verbose=True
        )
        all_results[name] = results
    
    return all_results


if __name__ == "__main__":
    from evaluation import AlphaEvaluator, compare_alphas
    
    # Load data
    panel = load_panel()
    
    # Initialize evaluator
    evaluator = AlphaEvaluator(panel, transaction_cost=0.001, n_quantiles=5)
    
    # Run baselines with 20-day forward returns (~ 1 month holding)
    print("\n" + "=" * 55)
    print("  PHASE 2: BASELINE EVALUATION")
    print("  Target: 20-day forward returns")
    print("  Rebalance: Every 20 trading days")
    print("  Transaction cost: 10 bps one-way")
    print("=" * 55)
    
    results = run_all_baselines(
        panel, evaluator, target='fwd_ret_20d', holding_period=20
    )
    
    # Comparison table
    print("\n\n")
    print("=" * 80)
    print("  BASELINE COMPARISON TABLE")
    print("=" * 80)
    comparison = compare_alphas(results)
    print(comparison.to_string())
    
    # Save results
    comparison.to_csv(os.path.join(PROC_DIR, '..', 'baseline_results.csv'))
    print(f"\nResults saved to Main_Project_2/data/baseline_results.csv")
    
    # Also evaluate with 60-day target (longer horizon)
    print("\n\n")
    print("=" * 55)
    print("  ADDITIONAL: 60-day Forward Returns")
    print("=" * 55)
    
    results_60d = run_all_baselines(
        panel, evaluator, target='fwd_ret_60d', holding_period=60
    )
    
    print("\n\n")
    print("=" * 80)
    print("  BASELINE COMPARISON TABLE (60-day horizon)")
    print("=" * 80)
    comparison_60d = compare_alphas(results_60d)
    print(comparison_60d.to_string())
    comparison_60d.to_csv(os.path.join(PROC_DIR, '..', 'baseline_results_60d.csv'))
