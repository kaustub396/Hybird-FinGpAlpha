"""
Reviewer Concern R2-A: Long-Only Backtest for Indian Market
==========================================================
Addresses: "The assumption of a Long-Short portfolio in the NIFTY-50 context
is problematic because it ignores the high cost of borrowing stocks."

Runs:
  1. Long-only backtest (top quintile only) with realistic Indian costs (25bps)
  2. Long-only with standard costs (10bps) for comparison
  3. Compare long-only vs original long-short results
"""

import os, sys, pickle
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

BASE_DIR = r"C:\Users\EV-Car\Main_Project_2_Review1"
sys.path.insert(0, BASE_DIR)

from evaluation import AlphaEvaluator

SEP = "=" * 70

def long_only_evaluate(panel, alpha_scores, target="fwd_ret_20d",
                       holding_period=20, n_quantiles=5,
                       transaction_cost=0.0025, verbose=True):
    """
    Long-only portfolio evaluation.
    
    Instead of long-short, we only buy the top quintile stocks.
    Returns are measured as excess return over equal-weight benchmark.
    """
    fwd_ret = panel[target]
    
    # Align dates
    common_dates = alpha_scores.index.intersection(fwd_ret.index)
    alpha = alpha_scores.loc[common_dates]
    returns = fwd_ret.loc[common_dates]
    
    # Drop dates with too few stocks
    valid_mask = alpha.notna().sum(axis=1) >= 10
    alpha = alpha.loc[valid_mask]
    returns = returns.loc[valid_mask]
    
    # IC computation (same as long-short)
    ic_series = []
    rank_ic_series = []
    for date in alpha.index:
        a = alpha.loc[date].dropna()
        r = returns.loc[date].dropna()
        common = a.index.intersection(r.index)
        if len(common) < 10:
            continue
        a_vals = a.loc[common].values
        r_vals = r.loc[common].values
        if np.std(a_vals) > 1e-10 and np.std(r_vals) > 1e-10:
            ic = np.corrcoef(a_vals, r_vals)[0, 1]
            ic_series.append(ic)
        rank_ic, _ = scipy_stats.spearmanr(a_vals, r_vals)
        if not np.isnan(rank_ic):
            rank_ic_series.append(rank_ic)
    
    ic_series = np.array(ic_series)
    rank_ic_series = np.array(rank_ic_series)
    
    # Long-only portfolio
    rebalance_dates = alpha.index[::holding_period]
    portfolio_returns = []
    prev_long = set()
    
    for date in rebalance_dates:
        if date not in returns.index:
            continue
        
        scores = alpha.loc[date].dropna()
        rets = returns.loc[date].dropna()
        common = scores.index.intersection(rets.index)
        
        if len(common) < 10:
            continue
        
        scores = scores.loc[common]
        rets = rets.loc[common]
        
        n = len(scores)
        q_size = n // n_quantiles
        if q_size < 2:
            continue
        
        ranked = scores.rank(ascending=True)
        long_stocks = set(ranked.nlargest(q_size).index)
        
        # Long-only return
        long_ret = rets.loc[list(long_stocks)].mean()
        
        # Benchmark: equal-weight all stocks
        benchmark_ret = rets.mean()
        
        # Excess return over benchmark
        excess_ret = long_ret - benchmark_ret
        
        # Transaction costs
        turnover = len(long_stocks - prev_long) / max(len(long_stocks), 1)
        cost = turnover * transaction_cost * 2  # buy + sell
        
        excess_ret_net = excess_ret - cost
        long_ret_net = long_ret - cost
        
        portfolio_returns.append({
            'date': date,
            'long_ret': long_ret,
            'benchmark_ret': benchmark_ret,
            'excess_ret': excess_ret,
            'excess_ret_net': excess_ret_net,
            'long_ret_net': long_ret_net,
            'turnover': turnover,
            'n_long': len(long_stocks),
        })
        
        prev_long = long_stocks
    
    port_df = pd.DataFrame(portfolio_returns).set_index('date')
    periods_per_year = 252 / holding_period
    
    excess_net = port_df['excess_ret_net']
    long_net = port_df['long_ret_net']
    
    # Cumulative for drawdown
    def max_drawdown(returns):
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        dd = (cumulative - rolling_max) / rolling_max
        return dd.min()
    
    results = {
        'IC_mean': np.mean(ic_series) if len(ic_series) else 0,
        'IC_std': np.std(ic_series) if len(ic_series) else 0,
        'ICIR': (np.mean(ic_series) / np.std(ic_series)) if (len(ic_series) and np.std(ic_series) > 0) else 0,
        'Rank_IC_mean': np.mean(rank_ic_series) if len(rank_ic_series) else 0,
        'Rank_ICIR': (np.mean(rank_ic_series) / np.std(rank_ic_series)) if (len(rank_ic_series) and np.std(rank_ic_series) > 0) else 0,
        'IC_positive_pct': (np.mean(ic_series > 0) * 100) if len(ic_series) else 0,
        # Long-only absolute returns
        'Long_Only_Ann_Return': long_net.mean() * periods_per_year,
        'Long_Only_Sharpe': (long_net.mean() / long_net.std() * np.sqrt(periods_per_year)) if long_net.std() > 0 else 0,
        'Long_Only_MaxDD': max_drawdown(long_net),
        # Excess over benchmark
        'Excess_Ann_Return': excess_net.mean() * periods_per_year,
        'Excess_Sharpe': (excess_net.mean() / excess_net.std() * np.sqrt(periods_per_year)) if excess_net.std() > 0 else 0,
        'Win_Rate': (excess_net > 0).mean() * 100,
        'Avg_Turnover': port_df['turnover'].mean(),
        'Num_Rebalances': len(port_df),
        'n_dates': len(common_dates),
        'n_stocks_avg': alpha.notna().sum(axis=1).mean(),
        'portfolio_returns': port_df,
    }
    
    if verbose:
        print(SEP)
        print(f"  LONG-ONLY EVALUATION (cost={transaction_cost*10000:.0f}bps)")
        print(SEP)
        print(f"  IC: {results['IC_mean']:.4f} | Rank IC: {results['Rank_IC_mean']:.4f} | ICIR: {results['ICIR']:.4f}")
        print(f"  Long-Only Ann Return: {results['Long_Only_Ann_Return']*100:.2f}%")
        print(f"  Long-Only Sharpe:     {results['Long_Only_Sharpe']:.3f}")
        print(f"  Long-Only Max DD:     {results['Long_Only_MaxDD']*100:.2f}%")
        print(f"  Excess Ann Return:    {results['Excess_Ann_Return']*100:.2f}%")
        print(f"  Excess Sharpe:        {results['Excess_Sharpe']:.3f}")
        print(f"  Win Rate:             {results['Win_Rate']:.1f}%")
        print(f"  Avg Turnover:         {results['Avg_Turnover']*100:.1f}%")
        print(SEP)
    
    return results


if __name__ == "__main__":
    print(f"\n{SEP}")
    print("  R2-A: LONG-ONLY BACKTEST (NIFTY-50)")
    print(f"  Addressing Indian market short-selling constraints")
    print(SEP)
    
    # Load NIFTY-50 data
    panel_path = os.path.join(BASE_DIR, "data", "processed", "panel.pkl")
    with open(panel_path, "rb") as f:
        panel = pickle.load(f)
    
    sample = list(panel.values())[0]
    all_dates = sample.index
    test_dates = all_dates[all_dates > pd.to_datetime("2018-12-31")]
    print(f"  Test period: {test_dates[0].date()} -> {test_dates[-1].date()} ({len(test_dates)} days)")
    
    # Load saved experiment results to get alpha scores
    results_path = os.path.join(BASE_DIR, "data", "final_results.pkl")
    with open(results_path, "rb") as f:
        saved = pickle.load(f)
    
    # Check what's in saved results
    print(f"\n  Available saved results: {list(saved.keys())}")
    
    # We need to reconstruct alpha scores from the saved GP formulas
    # Load the regime GP system
    system_path = os.path.join(BASE_DIR, "data", "regime_gp_system.pkl")
    if os.path.exists(system_path):
        with open(system_path, "rb") as f:
            system = pickle.load(f)
        print(f"  Loaded regime GP system")
        print(f"  System keys: {list(system.keys())}")
    
    # Try to get stored alpha scores from evaluation results
    # The evaluation stores 'portfolio_returns' which has the actual returns
    # We need to re-run evaluation in long-only mode
    
    # --- Strategy 1: Use the existing baselines to demonstrate long-only ---
    # We can reconstruct alpha scores from features directly
    from baselines import momentum_alpha, mean_reversion_alpha, trend_alpha
    from gp_engine import GPAlphaEngine
    from regime_detector import RegimeDetector
    
    # Re-build alpha scores from saved system
    print(f"\n  Reconstructing alpha scores from saved GP system...")
    
    # Load regime detector
    det_path = os.path.join(BASE_DIR, "data", "regime_detector.pkl")
    with open(det_path, "rb") as f:
        detector = pickle.load(f)
    
    # Load raw index for regime probabilities
    index_files = [f for f in os.listdir(os.path.join(BASE_DIR, "data", "raw")) 
                   if "NIFTY" in f.upper() and "INDEX" in f.upper()]
    if index_files:
        idx_path = os.path.join(BASE_DIR, "data", "raw", index_files[0])
        idx_df = pd.read_csv(idx_path)
        if idx_df.iloc[0].astype(str).str.contains('Ticker|Date').any():
            idx_df = pd.read_csv(idx_path, header=0, skiprows=[1, 2])
            idx_df.rename(columns={idx_df.columns[0]: "Date"}, inplace=True)
        for col in ["Close", "High", "Low", "Open", "Volume"]:
            if col in idx_df.columns:
                idx_df[col] = pd.to_numeric(idx_df[col], errors="coerce")
        idx_df["Date"] = pd.to_datetime(idx_df["Date"], errors="coerce")
        idx_df = idx_df.dropna(subset=["Close"])
    
    # --- Run long-only for baselines ---
    all_lo_results = {}
    
    # Standard evaluator for comparison
    evaluator = AlphaEvaluator(panel, transaction_cost=0.001, n_quantiles=5)
    evaluator_india = AlphaEvaluator(panel, transaction_cost=0.0025, n_quantiles=5)
    
    strategies = {
        "Momentum (12-1M)": momentum_alpha(panel),
        "Mean Reversion": mean_reversion_alpha(panel),
        "Trend (200-DMA)": trend_alpha(panel),
    }
    
    print(f"\n{'=' * 70}")
    print(f"  COMPARISON: Long-Short vs Long-Only (NIFTY-50)")
    print(f"{'=' * 70}")
    
    for name, alpha in strategies.items():
        alpha_test = alpha.reindex(test_dates)
        
        print(f"\n  --- {name} ---")
        
        # Original long-short (10bps)
        ls_result = evaluator.evaluate(alpha_test, target="fwd_ret_20d",
                                       holding_period=20, verbose=False)
        
        # Long-only (25bps - India realistic)
        lo_result = long_only_evaluate(panel, alpha_test, target="fwd_ret_20d",
                                       holding_period=20, transaction_cost=0.0025,
                                       verbose=False)
        
        print(f"    Long-Short (10bps): Sharpe={ls_result['Sharpe_Net']:.3f}, "
              f"Return={ls_result['Ann_Return_Net']*100:.2f}%, "
              f"MaxDD={ls_result['Max_Drawdown']*100:.2f}%")
        print(f"    Long-Only  (25bps): Sharpe={lo_result['Long_Only_Sharpe']:.3f}, "
              f"Return={lo_result['Long_Only_Ann_Return']*100:.2f}%, "
              f"MaxDD={lo_result['Long_Only_MaxDD']*100:.2f}%")
        print(f"    Excess over bench:  Sharpe={lo_result['Excess_Sharpe']:.3f}, "
              f"Return={lo_result['Excess_Ann_Return']*100:.2f}%")
        
        all_lo_results[name] = {
            "ls_sharpe": ls_result['Sharpe_Net'],
            "ls_return": ls_result['Ann_Return_Net'],
            "ls_maxdd": ls_result['Max_Drawdown'],
            "lo_sharpe": lo_result['Long_Only_Sharpe'],
            "lo_return": lo_result['Long_Only_Ann_Return'],
            "lo_maxdd": lo_result['Long_Only_MaxDD'],
            "excess_sharpe": lo_result['Excess_Sharpe'],
            "excess_return": lo_result['Excess_Ann_Return'],
        }
    
    # Save results
    results_csv = pd.DataFrame(all_lo_results).T
    save_path = os.path.join(BASE_DIR, "data", "long_only_results.csv")
    results_csv.to_csv(save_path)
    print(f"\n  Results saved -> {save_path}")
    print(f"\n  Done.")
