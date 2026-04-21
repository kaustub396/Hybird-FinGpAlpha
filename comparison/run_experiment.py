import pandas as pd
import numpy as np
import pickle
import os
import time
import sys

BASE_DIR = r"C:\Users\EV-Car\Main-Project_3"
GP_LIB_DIR = os.path.join(BASE_DIR, "gp")
if GP_LIB_DIR not in sys.path:
    sys.path.insert(0, GP_LIB_DIR)

from gp_engine import GPAlphaEngine
from evaluation import AlphaEvaluator, compare_alphas
from regime_detector import RegimeDetector
from baselines import (
    momentum_alpha, reversal_alpha, mean_reversion_alpha,
    low_volatility_alpha, trend_alpha, combined_alpha
)

PROC_DIR = r"C:\Users\EV-Car\Main_Project_2\data\processed"


def _cross_sectional_zscore(alpha_df):
    row_mean = alpha_df.mean(axis=1)
    row_std = alpha_df.std(axis=1).replace(0, np.nan)
    return alpha_df.sub(row_mean, axis=0).div(row_std, axis=0).fillna(0)


def build_afm_adaptive_alpha(panel):
    """
    AFM-style adaptive fusion alpha using integrated panel AFM features.
    """
    sent_strength = panel['afm_sentiment'].mean(axis=1).abs()
    fund_strength = panel['afm_fundamental'].mean(axis=1).abs()
    alpha_weight = sent_strength / (sent_strength + fund_strength + 1e-8)

    momentum_z = _cross_sectional_zscore(momentum_alpha(panel))
    meanrev_z = _cross_sectional_zscore(mean_reversion_alpha(panel))
    fused_alpha = momentum_z.mul(alpha_weight, axis=0) + meanrev_z.mul(1 - alpha_weight, axis=0)
    return fused_alpha


def build_hmm_regime_gp_alpha(panel, index_df, train_dates, test_dates, gp_params):
    """
    HMM-regime-aware GP Alpha:
    1) Detect regimes using HMM on index
    2) Evolve GP separately per HMM regime
    3) Soft-combine by regime probabilities
    """
    train_end = train_dates[-1]
    train_idx_df = index_df[pd.to_datetime(index_df['Date']) <= pd.to_datetime(train_end)]

    detector = RegimeDetector(n_regimes=2, random_state=42)
    detector.fit(train_idx_df)
    all_labels = detector.predict(index_df)

    common_train = train_dates.intersection(all_labels.index)
    regime_labels_train = all_labels.loc[common_train]

    stocks = list(panel.values())[0].columns
    combined_soft = pd.DataFrame(0.0, index=test_dates, columns=stocks)

    regime_proba = detector.predict_proba(index_df)
    regime_proba_test = regime_proba.reindex(test_dates).fillna(0.5)

    for regime_id in range(2):
        regime_dates = regime_labels_train[regime_labels_train == regime_id].index
        engine = GPAlphaEngine(panel, **gp_params, random_state=242 + regime_id)
        hof = engine.evolve(
            target='fwd_ret_20d', n_gen=50,
            date_mask=regime_dates, verbose=False, elite_size=5
        )
        alpha = engine.compute_alpha(hof[0], date_mask=test_dates)
        z_alpha = _cross_sectional_zscore(alpha)
        prob_col = regime_proba_test.iloc[:, regime_id]
        combined_soft += z_alpha.mul(prob_col, axis=0)

    return combined_soft


def build_power_results_table(results_dict):
    rows = []
    for strategy, res in results_dict.items():
        rows.append({
            'Strategy': strategy,
            'Ann Return': f"{res.get('Ann_Return_Net', 0) * 100:+.2f}%",
            'Sharpe': f"{res.get('Sharpe_Net', 0):.2f}",
            'Rank IC': f"{res.get('Rank_IC_mean', 0):.4f}",
            'Max Drawdown': f"{res.get('Max_Drawdown', 0) * 100:.2f}%",
        })

    table = pd.DataFrame(rows)
    table['__sort'] = table['Strategy'].map({
        'Regime-GP Ensemble (Ours)': 1,
        'Regime-GP Soft (Ours)': 2,
        'Regime-GP Hard (Ours)': 3,
        'Regime-GP Ensemble + HMM (Exp)': 4,
        'GP (AFM-Bear only)': 4,
        'GP (AFM-Bull only)': 5,
        'HMM Regime-GP Alpha': 6,
        'AFM Adaptive Fusion': 7,
        'Vanilla GP': 8,
        'Mean Reversion': 9,
        'Momentum (12-1M)': 10,
        'Low Volatility': 11,
        'Trend (200-DMA)': 12,
    }).fillna(9999)
    table = table.sort_values(['__sort', 'Strategy']).drop(columns=['__sort']).reset_index(drop=True)
    return table


def run_improved_system():
    print("Loading integrated panel (AFM features included)...")
    DATA_DIR = GP_LIB_DIR
    OUTPUT_DIR = os.path.join(BASE_DIR, "comparison")
    with open(os.path.join(DATA_DIR, 'integrated_panel.pkl'), 'rb') as f:
        panel = pickle.load(f)
    index_df = pd.read_csv(os.path.join(PROC_DIR, 'NIFTY50_INDEX.csv'))
    
    panel_dates = list(panel.values())[0].index
    train_end = '2018-12-31'
    train_dates = panel_dates[panel_dates <= pd.to_datetime(train_end)]
    test_dates = panel_dates[panel_dates > pd.to_datetime(train_end)]
    
    print(f"Train: {train_dates[0].date()} to {train_dates[-1].date()} ({len(train_dates)} days)")
    print(f"Test:  {test_dates[0].date()} to {test_dates[-1].date()} ({len(test_dates)} days)")
    
    print("\n" + "=" * 65)
    print("  REGIME DETECTION (2 regimes: AFM-Bear vs AFM-Bull)")
    print("=" * 65)
    
    afm_sent = panel['afm_sentiment'].iloc[:, 0]
    afm_fund = panel['afm_fundamental'].iloc[:, 0]
    fusion = afm_sent * 0.5 + afm_fund * 0.5
    afm_regimes = (fusion > fusion.median()).astype(int)

    class AFMDetector:
        def _get_regime_names(self):
            return ["AFM-Bear", "AFM-Bull"]
        def predict(self, idx_dfs):
            return afm_regimes
        def predict_proba(self, idx_dfs):
            prob = pd.DataFrame({'AFM-Bear': 1 - afm_regimes, 'AFM-Bull': afm_regimes})
            return prob
            
    detector = AFMDetector()
    all_labels = afm_regimes
    
    common_train = train_dates.intersection(all_labels.index)
    regime_labels_train = all_labels.loc[common_train]
    
    common_test = test_dates.intersection(all_labels.index)
    regime_labels_test = all_labels.loc[common_test]
    
    print(f"\nTrain regime split:")
    for i, name in enumerate(detector._get_regime_names()):
        n = (regime_labels_train == i).sum()
        print(f"  {name}: {n} ({n/len(regime_labels_train)*100:.1f}%)")
    
    print(f"\nTest regime split:")
    for i, name in enumerate(detector._get_regime_names()):
        n = (regime_labels_test == i).sum()
        print(f"  {name}: {n} ({n/len(regime_labels_test)*100:.1f}%)")
    
    gp_params = {
        'population_size': 500,
        'tournament_size': 5,
        'max_depth': 6,
        'cx_prob': 0.7,
        'mut_prob': 0.2,
        'parsimony_weight': 0.001,
    }
    
    regime_engines = {}
    regime_alphas = {}
    regime_formulas = {}
    
    regime_names = detector._get_regime_names()
    
    for regime_id in range(2):
        regime_name = regime_names[regime_id]
        regime_dates = regime_labels_train[regime_labels_train == regime_id].index
        
        print(f"\n{'-' * 55}")
        print(f"  GP Evolution: {regime_name} Regime ({len(regime_dates)} dates)")
        print(f"{'-' * 55}")
        
        engine = GPAlphaEngine(panel, **gp_params, random_state=42 + regime_id)
        hof = engine.evolve(
            target='fwd_ret_20d', n_gen=50,
            date_mask=regime_dates, verbose=True, elite_size=5
        )
        
        regime_engines[regime_id] = engine
        regime_alphas[regime_id] = list(hof[:5])
        regime_formulas[regime_id] = [engine.get_formula(ind) for ind in hof[:5]]
        
        print(f"\n  Top 3 {regime_name} formulas:")
        for j in range(min(3, len(hof))):
            print(f"    #{j+1} (IC={hof[j].fitness.values[0]:.4f}): {engine.get_formula(hof[j])}")
    
    print(f"\n{'-' * 55}")
    print(f"  Vanilla GP Baseline")
    print(f"{'-' * 55}")
    
    vanilla_engine = GPAlphaEngine(panel, **gp_params, random_state=141)
    vanilla_hof = vanilla_engine.evolve(
        target='fwd_ret_20d', n_gen=50,
        date_mask=common_train, verbose=True, elite_size=5
    )
    
    evaluator = AlphaEvaluator(panel, transaction_cost=0.001, n_quantiles=5)
    
    regime_proba = detector.predict_proba(index_df)
    regime_proba_test = regime_proba.reindex(test_dates).fillna(0.5)
    
    results = {}
    stocks = list(panel.values())[0].columns
    
    print("\n" + "=" * 55)
    print("  METHOD 1: Soft Regime Combination")
    print("=" * 55)
    
    combined_soft = pd.DataFrame(0.0, index=test_dates, columns=stocks)
    for regime_id in range(2):
        engine = regime_engines[regime_id]
        best_ind = regime_alphas[regime_id][0]
        alpha = engine.compute_alpha(best_ind, date_mask=test_dates)
        row_mean = alpha.mean(axis=1)
        row_std = alpha.std(axis=1).replace(0, np.nan)
        z_alpha = alpha.sub(row_mean, axis=0).div(row_std, axis=0).fillna(0)
        prob_col = regime_proba_test.iloc[:, regime_id]
        combined_soft += z_alpha.mul(prob_col, axis=0)
    
    results['Regime-GP Soft (Ours)'] = evaluator.evaluate(
        combined_soft, target='fwd_ret_20d', holding_period=20, verbose=True
    )
    
    print("\n" + "=" * 55)
    print("  METHOD 2: Hard Regime Combination")
    print("=" * 55)
    
    combined_hard = pd.DataFrame(0.0, index=test_dates, columns=stocks)
    current_regime_test = detector.predict(index_df).reindex(test_dates)
    
    for regime_id in range(2):
        engine = regime_engines[regime_id]
        best_ind = regime_alphas[regime_id][0]
        alpha = engine.compute_alpha(best_ind, date_mask=test_dates)
        row_mean = alpha.mean(axis=1)
        row_std = alpha.std(axis=1).replace(0, np.nan)
        z_alpha = alpha.sub(row_mean, axis=0).div(row_std, axis=0).fillna(0)
        
        mask = current_regime_test == regime_id
        combined_hard.loc[mask] = z_alpha.loc[mask]
    
    results['Regime-GP Hard (Ours)'] = evaluator.evaluate(
        combined_hard, target='fwd_ret_20d', holding_period=20, verbose=True
    )
    
    print("\n" + "=" * 55)
    print("  METHOD 3: Ensemble (Top-3 per Regime, Soft)")
    print("=" * 55)
    
    combined_ensemble = pd.DataFrame(0.0, index=test_dates, columns=stocks)
    for regime_id in range(2):
        engine = regime_engines[regime_id]
        regime_alpha = pd.DataFrame(0.0, index=test_dates, columns=stocks)
        for ind in regime_alphas[regime_id][:3]:
            alpha = engine.compute_alpha(ind, date_mask=test_dates)
            row_mean = alpha.mean(axis=1)
            row_std = alpha.std(axis=1).replace(0, np.nan)
            z_alpha = alpha.sub(row_mean, axis=0).div(row_std, axis=0).fillna(0)
            regime_alpha += z_alpha / 3
        prob_col = regime_proba_test.iloc[:, regime_id]
        combined_ensemble += regime_alpha.mul(prob_col, axis=0)
    
    results['Regime-GP Ensemble (Ours)'] = evaluator.evaluate(
        combined_ensemble, target='fwd_ret_20d', holding_period=20, verbose=True
    )
    
    print("\n" + "=" * 55)
    print("  VANILLA GP (No Regime)")
    print("=" * 55)
    
    vanilla_combined = pd.DataFrame(0.0, index=test_dates, columns=stocks)
    for ind in list(vanilla_hof[:3]):
        alpha = vanilla_engine.compute_alpha(ind, date_mask=test_dates)
        row_mean = alpha.mean(axis=1)
        row_std = alpha.std(axis=1).replace(0, np.nan)
        z_alpha = alpha.sub(row_mean, axis=0).div(row_std, axis=0).fillna(0)
        vanilla_combined += z_alpha / 3
    
    results['Vanilla GP'] = evaluator.evaluate(
        vanilla_combined, target='fwd_ret_20d', holding_period=20, verbose=True
    )

    hmm_gp_alpha = build_hmm_regime_gp_alpha(
        panel=panel,
        index_df=index_df,
        train_dates=train_dates,
        test_dates=test_dates,
        gp_params=gp_params,
    )
    results['HMM Regime-GP Alpha'] = evaluator.evaluate(
        hmm_gp_alpha, target='fwd_ret_20d', holding_period=20, verbose=False
    )

    # Experimental hybrid: keep original Regime-GP Ensemble unchanged, and add
    # a separate strategy that blends it with HMM Regime-GP Alpha.
    ensemble_hmm_hybrid = _cross_sectional_zscore(
        0.5 * combined_ensemble + 0.5 * hmm_gp_alpha
    )
    results['Regime-GP Ensemble + HMM (Exp)'] = evaluator.evaluate(
        ensemble_hmm_hybrid, target='fwd_ret_20d', holding_period=20, verbose=False
    )

    afm_alpha = build_afm_adaptive_alpha(panel).reindex(test_dates)
    results['AFM Adaptive Fusion'] = evaluator.evaluate(
        afm_alpha, target='fwd_ret_20d', holding_period=20, verbose=False
    )
    
    for regime_id in range(2):
        regime_name = regime_names[regime_id]
        engine = regime_engines[regime_id]
        alpha = engine.compute_alpha(regime_alphas[regime_id][0], date_mask=test_dates)
        results[f'GP ({regime_name} only)'] = evaluator.evaluate(
            alpha, target='fwd_ret_20d', holding_period=20, verbose=False
        )
    
    baseline_funcs = {
        'Momentum (12-1M)': momentum_alpha,
        'Mean Reversion': mean_reversion_alpha,
        'Low Volatility': low_volatility_alpha,
        'Trend (200-DMA)': trend_alpha,
    }
    for name, func in baseline_funcs.items():
        alpha = func(panel).reindex(test_dates)
        results[name] = evaluator.evaluate(
            alpha, target='fwd_ret_20d', holding_period=20, verbose=False
        )
    
    print("\n\n" + "=" * 85)
    print("  MAIN COMPARISON TABLE (Out-of-Sample 2019-2025)")
    print("=" * 85)
    comparison = compare_alphas(results)
    print(comparison.to_string())
    
    comparison.to_csv(os.path.join(OUTPUT_DIR, 'integrated_main_results.csv'))

    print("\n\nResults (500 population x 50 generations)")
    power_table = build_power_results_table(results)
    print(power_table.to_string(index=False))
    power_table.to_csv(os.path.join(OUTPUT_DIR, 'integrated_power_results.csv'), index=False)
    
    print("\n\n" + "=" * 65)
    print("  DISCOVERED FORMULAS")
    print("=" * 65)
    for regime_id in range(2):
        regime_name = regime_names[regime_id]
        print(f"\n  > {regime_name} Regime:")
        for j, formula in enumerate(regime_formulas[regime_id][:3]):
            ic = regime_alphas[regime_id][j].fitness.values[0]
            print(f"    #{j+1} (IC={ic:.4f}): {formula}")
    
    print(f"\n  > Vanilla GP:")
    for j, ind in enumerate(list(vanilla_hof[:3])):
        formula = vanilla_engine.get_formula(ind)
        ic = ind.fitness.values[0]
        print(f"    #{j+1} (IC={ic:.4f}): {formula}")
    
    return results


if __name__ == "__main__":
    results = run_improved_system()
