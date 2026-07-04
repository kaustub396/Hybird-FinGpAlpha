"""
Complete Reviewer Experiments: HMM-GPa Long-Only + Full Statistics
================================================================
Reconstructs HMM-GPa alpha scores from saved formula strings,
then runs:
  1. Long-only backtest (R2-A) for HMM-GPa and Vanilla GP
  2. DSR on real results (R1-C)  
  3. Permutation test on GP alphas (R1-B)
  4. Paired IC tests (R1-C)
"""

import os, sys, pickle, time
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

BASE_DIR = r"C:\Users\EV-Car\Main_Project_2_Review1"
sys.path.insert(0, BASE_DIR)

from gp_engine import GPAlphaEngine
from regime_detector import RegimeDetector
from evaluation import AlphaEvaluator
from baselines import momentum_alpha, mean_reversion_alpha, trend_alpha

SEP = "=" * 70


def formula_to_alpha(engine, formula_str, date_mask=None):
    """
    Parse a GP formula string and compute alpha scores.
    Uses DEAP's PrimitiveTree.from_string to reconstruct the individual.
    """
    from deap import gp as deap_gp
    
    # Parse the formula string into a PrimitiveTree
    individual = deap_gp.PrimitiveTree.from_string(formula_str, engine.pset)
    
    # Compile and compute
    func = engine.toolbox.compile(expr=individual)
    
    target_df = engine.panel['fwd_ret_20d']
    if date_mask is not None:
        if isinstance(date_mask, pd.Index):
            target_df = target_df.loc[target_df.index.intersection(date_mask)]
        else:
            target_df = target_df.loc[date_mask]
    
    dates = target_df.index
    stocks = target_df.columns
    
    args = []
    for fname in engine.feature_names:
        feat_df = engine.panel[fname].reindex(index=dates, columns=stocks)
        args.append(feat_df.values)
    
    try:
        alpha = func(*args)
        alpha = np.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
        if np.isscalar(alpha):
            alpha = np.full((len(dates), len(stocks)), alpha)
        return pd.DataFrame(alpha, index=dates, columns=stocks)
    except Exception as e:
        print(f"    Error computing alpha: {e}")
        return pd.DataFrame(0.0, index=dates, columns=stocks)


def long_only_portfolio(alpha_scores, fwd_returns, holding_period=20,
                        n_quantiles=5, transaction_cost=0.0025):
    """
    Build long-only portfolio (top quintile), return metrics.
    """
    common_dates = alpha_scores.index.intersection(fwd_returns.index)
    alpha = alpha_scores.loc[common_dates]
    returns = fwd_returns.loc[common_dates]
    
    valid_mask = alpha.notna().sum(axis=1) >= 10
    alpha = alpha.loc[valid_mask]
    returns = returns.loc[valid_mask]
    
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
        
        long_ret = rets.loc[list(long_stocks)].mean()
        benchmark_ret = rets.mean()
        excess_ret = long_ret - benchmark_ret
        
        turnover = len(long_stocks - prev_long) / max(len(long_stocks), 1)
        cost = turnover * transaction_cost * 2
        
        portfolio_returns.append({
            'date': date,
            'long_ret_net': long_ret - cost,
            'excess_ret_net': excess_ret - cost,
            'turnover': turnover,
        })
        prev_long = long_stocks
    
    if not portfolio_returns:
        return {'Long_Only_Sharpe': 0, 'Long_Only_Return': 0, 'Long_Only_MaxDD': 0,
                'Excess_Sharpe': 0, 'Excess_Return': 0}
    
    port_df = pd.DataFrame(portfolio_returns).set_index('date')
    periods_per_year = 252 / holding_period
    
    long_net = port_df['long_ret_net']
    excess_net = port_df['excess_ret_net']
    
    cumulative = (1 + long_net).cumprod()
    rolling_max = cumulative.expanding().max()
    max_dd = ((cumulative - rolling_max) / rolling_max).min()
    
    return {
        'Long_Only_Sharpe': (long_net.mean() / long_net.std() * np.sqrt(periods_per_year)) if long_net.std() > 0 else 0,
        'Long_Only_Return': long_net.mean() * periods_per_year,
        'Long_Only_MaxDD': max_dd,
        'Excess_Sharpe': (excess_net.mean() / excess_net.std() * np.sqrt(periods_per_year)) if excess_net.std() > 0 else 0,
        'Excess_Return': excess_net.mean() * periods_per_year,
        'Avg_Turnover': port_df['turnover'].mean(),
    }


def deflated_sharpe_ratio(sharpe, n_trials, n_returns, skew=0.0, kurt=3.0):
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)."""
    euler = 0.5772156649
    e_max = np.sqrt(2 * np.log(n_trials)) - \
            (np.log(np.pi) + euler) / (2 * np.sqrt(2 * np.log(n_trials)))
    se = np.sqrt((1 - skew * sharpe + ((kurt - 1) / 4) * sharpe**2) / n_returns)
    if se < 1e-10:
        return {"DSR": 0.5, "E_max_sharpe": e_max}
    z = (sharpe - e_max) / se
    return {"DSR": scipy_stats.norm.cdf(z), "E_max_sharpe": e_max, "z": z}


def permutation_test(alpha_scores, fwd_returns, n_perm=500, seed=42):
    """Permutation test for IC significance."""
    rng = np.random.RandomState(seed)
    common = alpha_scores.index.intersection(fwd_returns.index)
    alpha = alpha_scores.loc[common]
    returns = fwd_returns.loc[common]
    
    def compute_mean_ic(alpha_df, returns_df):
        ics = []
        for date in alpha_df.index:
            a = alpha_df.loc[date].dropna()
            r = returns_df.loc[date].dropna()
            c = a.index.intersection(r.index)
            if len(c) < 10:
                continue
            av, rv = a.loc[c].values, r.loc[c].values
            if np.std(av) > 1e-10 and np.std(rv) > 1e-10:
                ics.append(np.corrcoef(av, rv)[0, 1])
        return np.mean(ics) if ics else 0, ics
    
    real_ic, real_ic_series = compute_mean_ic(alpha, returns)
    real_icir = np.mean(real_ic_series) / np.std(real_ic_series) if len(real_ic_series) > 1 and np.std(real_ic_series) > 0 else 0
    
    perm_ics = []
    dates_arr = np.array(common)
    
    print(f"    Running {n_perm} permutations...", end=" ", flush=True)
    for _ in range(n_perm):
        shuffled = dates_arr[rng.permutation(len(dates_arr))]
        returns_shuffled = returns.copy()
        returns_shuffled.index = shuffled
        returns_shuffled = returns_shuffled.sort_index()
        pic, _ = compute_mean_ic(alpha, returns_shuffled)
        perm_ics.append(pic)
    print("Done.")
    
    perm_ics = np.array(perm_ics)
    p_val = (perm_ics >= real_ic).mean()
    
    return {
        'real_ic': real_ic, 'real_icir': real_icir,
        'perm_ic_mean': perm_ics.mean(), 'perm_ic_std': perm_ics.std(),
        'p_value': p_val, 'n_perm': n_perm,
        'real_ic_series': np.array(real_ic_series),
    }


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    total_start = time.time()
    
    print(f"\n{SEP}")
    print("  COMPLETE REVIEWER EXPERIMENTS")
    print("  R2-A: Long-Only | R1-B: Permutation | R1-C: DSR + Paired Tests")
    print(SEP)
    
    # Load data
    panel_path = os.path.join(BASE_DIR, "data", "processed", "panel.pkl")
    with open(panel_path, "rb") as f:
        panel = pickle.load(f)
    
    system_path = os.path.join(BASE_DIR, "data", "regime_gp_system.pkl")
    with open(system_path, "rb") as f:
        system = pickle.load(f)
    
    sample = list(panel.values())[0]
    all_dates = sample.index
    test_dates = all_dates[all_dates > pd.to_datetime("2018-12-31")]
    fwd_ret = panel["fwd_ret_20d"]
    
    print(f"  Test: {test_dates[0].date()} -> {test_dates[-1].date()} ({len(test_dates)} days)")
    print(f"  Regime system: {system['n_regimes']} regimes, {system['combination']} combination")
    
    # ================================================================
    # RECONSTRUCT HMM-GPa ALPHA SCORES
    # ================================================================
    print(f"\n{SEP}")
    print("  RECONSTRUCTING HMM-GPa ALPHA SCORES FROM SAVED FORMULAS")
    print(SEP)
    
    engine = GPAlphaEngine(panel, random_state=42)
    engine._setup_gp()  # Must call this to initialize DEAP PSET
    detector = system['regime_detector']
    regime_formulas = system['regime_formulas']
    n_regimes = system['n_regimes']
    
    # Get regime probabilities for test period
    # Load raw index
    raw_dir = os.path.join(BASE_DIR, "data", "raw")
    index_files = [f for f in os.listdir(raw_dir) 
                   if "NIFTY" in f.upper() and "INDEX" in f.upper()]
    idx_path = os.path.join(raw_dir, index_files[0])
    idx_df = pd.read_csv(idx_path)
    if idx_df.iloc[0].astype(str).str.contains('Ticker|Date').any():
        idx_df = pd.read_csv(idx_path, header=0, skiprows=[1, 2])
        idx_df.rename(columns={idx_df.columns[0]: "Date"}, inplace=True)
    for col in ["Close", "High", "Low", "Open", "Volume"]:
        if col in idx_df.columns:
            idx_df[col] = pd.to_numeric(idx_df[col], errors="coerce")
    idx_df["Date"] = pd.to_datetime(idx_df["Date"], errors="coerce")
    idx_df = idx_df.dropna(subset=["Close"])
    
    regime_proba = detector.predict_proba(idx_df)
    regime_proba = regime_proba.reindex(test_dates).fillna(1.0 / n_regimes)
    
    stocks = fwd_ret.columns
    
    # Regime-aware combined alpha
    hmm_alpha = pd.DataFrame(0.0, index=test_dates, columns=stocks)
    
    for regime_id, formulas in regime_formulas.items():
        print(f"  Regime {regime_id}: computing {len(formulas)} formulas...")
        regime_alpha = pd.DataFrame(0.0, index=test_dates, columns=stocks)
        
        for i, formula_str in enumerate(formulas):
            try:
                a = formula_to_alpha(engine, formula_str, date_mask=test_dates)
                # Z-score normalize
                row_mean = a.mean(axis=1)
                row_std = a.std(axis=1).replace(0, np.nan)
                z = a.sub(row_mean, axis=0).div(row_std, axis=0).fillna(0)
                regime_alpha += z / len(formulas)
                print(f"    Formula #{i+1}: OK")
            except Exception as e:
                print(f"    Formula #{i+1}: ERROR - {e}")
        
        # Weight by regime probability
        prob_col = regime_proba.iloc[:, regime_id]
        prob_aligned = prob_col.reindex(test_dates).fillna(1.0 / n_regimes)
        hmm_alpha += regime_alpha.mul(prob_aligned, axis=0)
    
    print(f"  HMM-GPa alpha reconstructed: {hmm_alpha.shape}")
    
    # Also build Vanilla GP alpha (use all formulas equally weighted without regime conditioning)
    vanilla_alpha = pd.DataFrame(0.0, index=test_dates, columns=stocks)
    all_formulas = []
    for formulas in regime_formulas.values():
        all_formulas.extend(formulas)
    
    print(f"\n  Building Vanilla-equivalent alpha from {len(all_formulas)} total formulas...")
    for i, formula_str in enumerate(all_formulas):
        try:
            a = formula_to_alpha(engine, formula_str, date_mask=test_dates)
            row_mean = a.mean(axis=1)
            row_std = a.std(axis=1).replace(0, np.nan)
            z = a.sub(row_mean, axis=0).div(row_std, axis=0).fillna(0)
            vanilla_alpha += z / len(all_formulas)
        except:
            pass
    
    # ================================================================
    # R2-A: LONG-ONLY BACKTEST
    # ================================================================
    print(f"\n{SEP}")
    print("  R2-A: LONG-ONLY BACKTEST (NIFTY-50)")
    print(f"  Transaction cost: 25bps (India realistic)")
    print(SEP)
    
    evaluator = AlphaEvaluator(panel, transaction_cost=0.001, n_quantiles=5)
    
    strategies = {
        "HMM-GPa (Ours)": hmm_alpha,
        "Vanilla GP (all formulas)": vanilla_alpha,
        "Momentum (12-1M)": momentum_alpha(panel).reindex(test_dates),
        "Mean Reversion": mean_reversion_alpha(panel).reindex(test_dates),
        "Trend (200-DMA)": trend_alpha(panel).reindex(test_dates),
    }
    
    all_results = {}
    
    for name, alpha in strategies.items():
        print(f"\n  --- {name} ---")
        
        # Long-short (original, 10bps)
        ls = evaluator.evaluate(alpha, target="fwd_ret_20d",
                                holding_period=20, verbose=False)
        
        # Long-only (25bps India)
        lo = long_only_portfolio(alpha, fwd_ret, holding_period=20,
                                 transaction_cost=0.0025)
        
        print(f"    Long-Short (10bps): Sharpe={ls['Sharpe_Net']:.3f}, "
              f"Return={ls['Ann_Return_Net']*100:.2f}%, "
              f"MaxDD={ls['Max_Drawdown']*100:.2f}%")
        print(f"    Long-Only  (25bps): Sharpe={lo['Long_Only_Sharpe']:.3f}, "
              f"Return={lo['Long_Only_Return']*100:.2f}%, "
              f"MaxDD={lo['Long_Only_MaxDD']*100:.2f}%")
        print(f"    Excess over bench:  Sharpe={lo['Excess_Sharpe']:.3f}, "
              f"Return={lo['Excess_Return']*100:.2f}%")
        
        all_results[name] = {
            'ls_sharpe': ls['Sharpe_Net'],
            'ls_return': ls['Ann_Return_Net'],
            'ls_maxdd': ls['Max_Drawdown'],
            'lo_sharpe': lo['Long_Only_Sharpe'],
            'lo_return': lo['Long_Only_Return'],
            'lo_maxdd': lo['Long_Only_MaxDD'],
            'excess_sharpe': lo['Excess_Sharpe'],
            'excess_return': lo['Excess_Return'],
            'ic_mean': ls['IC_mean'],
            'rank_ic': ls['Rank_IC_mean'],
            'icir': ls['ICIR'],
            'ic_series': ls.get('ic_series', []),
        }
    
    # ================================================================
    # R1-B: PERMUTATION TEST
    # ================================================================
    print(f"\n{SEP}")
    print("  R1-B: PERMUTATION TEST (Data-Mining Risk)")
    print(SEP)
    
    perm_results = {}
    for name in ["HMM-GPa (Ours)", "Momentum (12-1M)"]:
        print(f"\n  Permutation test: {name}")
        perm = permutation_test(strategies[name], fwd_ret, n_perm=500)
        perm_results[name] = perm
        
        print(f"    Real Mean IC:     {perm['real_ic']:.4f}")
        print(f"    Permuted IC mean: {perm['perm_ic_mean']:.4f} +/- {perm['perm_ic_std']:.4f}")
        print(f"    p-value:          {perm['p_value']:.4f}")
        sig = "***" if perm['p_value'] < 0.01 else "**" if perm['p_value'] < 0.05 else "*" if perm['p_value'] < 0.10 else "n.s."
        print(f"    Significance:     {sig}")
    
    # ================================================================
    # R1-C: DEFLATED SHARPE RATIO
    # ================================================================
    print(f"\n{SEP}")
    print("  R1-C: DEFLATED SHARPE RATIO")
    print(SEP)
    
    n_trials = 500 * 50 * n_regimes  # population * generations * regimes
    
    for name, res in all_results.items():
        sharpe = res['ls_sharpe']
        # Approximate n_returns
        n_ret = len(test_dates) // 20
        dsr = deflated_sharpe_ratio(sharpe, n_trials, n_ret)
        
        sig = "***" if dsr['DSR'] > 0.99 else "**" if dsr['DSR'] > 0.95 else "*" if dsr['DSR'] > 0.90 else "n.s."
        print(f"  {name:30s}: Sharpe={sharpe:.3f}, DSR={dsr['DSR']:.4f} {sig}, "
              f"E[max]={dsr['E_max_sharpe']:.3f}")
    
    # ================================================================
    # R1-C: PAIRED IC TESTS
    # ================================================================
    print(f"\n{SEP}")
    print("  R1-C: PAIRED IC TESTS (HMM-GPa vs others)")
    print(SEP)
    
    hmm_ic = all_results["HMM-GPa (Ours)"]["ic_series"]
    
    for name in ["Momentum (12-1M)", "Mean Reversion", "Trend (200-DMA)"]:
        other_ic = all_results[name]["ic_series"]
        n = min(len(hmm_ic), len(other_ic))
        if n < 10:
            print(f"  {name}: insufficient data for paired test")
            continue
        
        a, b = np.array(hmm_ic[:n]), np.array(other_ic[:n])
        diff = a - b
        
        t_stat, p_t = scipy_stats.ttest_rel(a, b)
        try:
            w_stat, p_w = scipy_stats.wilcoxon(diff)
        except:
            w_stat, p_w = 0, 1.0
        
        cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
        ir = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
        
        sig_t = "***" if p_t < 0.01 else "**" if p_t < 0.05 else "*" if p_t < 0.10 else "n.s."
        print(f"  HMM-GPa vs {name:20s}: IC_diff={np.mean(diff):.4f}, "
              f"t={t_stat:.3f}, p={p_t:.4f} {sig_t}, "
              f"Cohen's d={cohens_d:.3f}, IR={ir:.3f}")
    
    # ================================================================
    # SUMMARY TABLE
    # ================================================================
    print(f"\n\n{SEP}")
    print("  COMPLETE RESULTS SUMMARY")
    print(SEP)
    
    print(f"\n  {'Strategy':<30} {'LS Sharpe':>10} {'LO Sharpe':>10} {'LS Return':>10} {'LO Return':>10} {'LO MaxDD':>10} {'IC':>8}")
    print(f"  {'-'*90}")
    for name, r in all_results.items():
        print(f"  {name:<30} {r['ls_sharpe']:>10.3f} {r['lo_sharpe']:>10.3f} "
              f"{r['ls_return']*100:>9.2f}% {r['lo_return']*100:>9.2f}% "
              f"{r['lo_maxdd']*100:>9.2f}% {r['ic_mean']:>8.4f}")
    
    # Save everything
    save_data = {
        'all_results': {k: {kk: vv for kk, vv in v.items() if kk != 'ic_series'} 
                        for k, v in all_results.items()},
        'permutation': {k: {kk: vv for kk, vv in v.items() if kk != 'real_ic_series'} 
                        for k, v in perm_results.items()},
    }
    save_path = os.path.join(BASE_DIR, "data", "complete_reviewer_results.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(save_data, f)
    
    # Also save as CSV
    csv_data = []
    for name, r in all_results.items():
        csv_data.append({
            'Strategy': name,
            'LS_Sharpe': f"{r['ls_sharpe']:.3f}",
            'LS_Return': f"{r['ls_return']*100:.2f}%",
            'LS_MaxDD': f"{r['ls_maxdd']*100:.2f}%",
            'LO_Sharpe_25bps': f"{r['lo_sharpe']:.3f}",
            'LO_Return_25bps': f"{r['lo_return']*100:.2f}%",
            'LO_MaxDD': f"{r['lo_maxdd']*100:.2f}%",
            'Excess_Sharpe': f"{r['excess_sharpe']:.3f}",
            'IC': f"{r['ic_mean']:.4f}",
            'RankIC': f"{r['rank_ic']:.4f}",
            'ICIR': f"{r['icir']:.4f}",
        })
    
    csv_path = os.path.join(BASE_DIR, "data", "complete_reviewer_results.csv")
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    
    elapsed = time.time() - total_start
    print(f"\n  Results saved -> {csv_path}")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"\n  Done.")
