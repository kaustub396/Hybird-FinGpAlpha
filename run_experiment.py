"""
Improved Regime-Aware GP with 2 regimes and better combination.

Key fixes from initial run:
1. Use 2 regimes (High-Vol Bear vs Low-Vol Bull) — 3-regime HMM had Sideways≈Bull
2. Try both hard and soft combination
3. Evaluate ablations properly
"""

import pandas as pd
import numpy as np
import pickle
import os
import time

from regime_detector import RegimeDetector
from gp_engine import GPAlphaEngine
from evaluation import AlphaEvaluator, compare_alphas
from baselines import (
    momentum_alpha, reversal_alpha, mean_reversion_alpha,
    low_volatility_alpha, trend_alpha, combined_alpha
)

PROC_DIR = r"C:\Users\EV-Car\Main_Project_2\data\processed"


def run_improved_system():
    # Load data
    print("Loading data...")
    with open(os.path.join(PROC_DIR, 'panel.pkl'), 'rb') as f:
        panel = pickle.load(f)
    index_df = pd.read_csv(os.path.join(PROC_DIR, 'NIFTY50_INDEX.csv'))
    
    panel_dates = list(panel.values())[0].index
    train_end = '2018-12-31'
    train_dates = panel_dates[panel_dates <= pd.to_datetime(train_end)]
    test_dates = panel_dates[panel_dates > pd.to_datetime(train_end)]
    
    print(f"Train: {train_dates[0].date()} to {train_dates[-1].date()} ({len(train_dates)} days)")
    print(f"Test:  {test_dates[0].date()} to {test_dates[-1].date()} ({len(test_dates)} days)")
    
    # ============================================
    # REGIME DETECTION (2 regimes)
    # ============================================
    print("\n" + "=" * 65)
    print("  REGIME DETECTION (2 regimes: Bear vs Bull)")
    print("=" * 65)
    
    train_idx_df = index_df[pd.to_datetime(index_df['Date']) <= pd.to_datetime(train_end)]
    
    detector = RegimeDetector(n_regimes=2, random_state=42)
    detector.fit(train_idx_df)
    detector.print_summary()
    
    all_labels = detector.predict(index_df)
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
    
    # ============================================
    # GP EVOLUTION PER REGIME
    # ============================================
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
        
        print(f"\n{'─' * 55}")
        print(f"  GP Evolution: {regime_name} Regime ({len(regime_dates)} dates)")
        print(f"{'─' * 55}")
        
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
    
    # ============================================
    # VANILLA GP (no regime)
    # ============================================
    print(f"\n{'─' * 55}")
    print(f"  Vanilla GP Baseline")
    print(f"{'─' * 55}")
    
    vanilla_engine = GPAlphaEngine(panel, **gp_params, random_state=141)
    vanilla_hof = vanilla_engine.evolve(
        target='fwd_ret_20d', n_gen=50,
        date_mask=common_train, verbose=True, elite_size=5
    )
    
    # ============================================
    # BUILD COMBINED ALPHA SIGNALS FOR TEST SET
    # ============================================
    evaluator = AlphaEvaluator(panel, transaction_cost=0.001, n_quantiles=5)
    
    # Get regime probabilities on test set
    regime_proba = detector.predict_proba(index_df)
    regime_proba_test = regime_proba.reindex(test_dates).fillna(0.5)
    
    results = {}
    stocks = list(panel.values())[0].columns
    
    # --- Method 1: Soft combination (probability-weighted) ---
    print("\n" + "=" * 55)
    print("  METHOD 1: Soft Regime Combination")
    print("=" * 55)
    
    combined_soft = pd.DataFrame(0.0, index=test_dates, columns=stocks)
    for regime_id in range(2):
        engine = regime_engines[regime_id]
        # Use top-1 formula only (simplest, most interpretable)
        best_ind = regime_alphas[regime_id][0]
        alpha = engine.compute_alpha(best_ind, date_mask=test_dates)
        # Z-score cross-sectionally
        row_mean = alpha.mean(axis=1)
        row_std = alpha.std(axis=1).replace(0, np.nan)
        z_alpha = alpha.sub(row_mean, axis=0).div(row_std, axis=0).fillna(0)
        # Weight by regime probability
        prob_col = regime_proba_test.iloc[:, regime_id]
        combined_soft += z_alpha.mul(prob_col, axis=0)
    
    results['Regime-GP Soft (Ours)'] = evaluator.evaluate(
        combined_soft, target='fwd_ret_20d', holding_period=20, verbose=True
    )
    
    # --- Method 2: Hard combination (use current regime's alpha) ---
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
    
    # --- Method 3: Ensemble of top-3 per regime, soft-weighted ---
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
    
    # --- Vanilla GP ---
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
    
    # --- Individual regime alphas ---
    for regime_id in range(2):
        regime_name = regime_names[regime_id]
        engine = regime_engines[regime_id]
        alpha = engine.compute_alpha(regime_alphas[regime_id][0], date_mask=test_dates)
        results[f'GP ({regime_name} only)'] = evaluator.evaluate(
            alpha, target='fwd_ret_20d', holding_period=20, verbose=False
        )
    
    # --- Baselines ---
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
    
    # ============================================
    # COMPARISON TABLE
    # ============================================
    print("\n\n" + "=" * 85)
    print("  MAIN COMPARISON TABLE (Out-of-Sample 2019-2025)")
    print("=" * 85)
    comparison = compare_alphas(results)
    print(comparison.to_string())
    
    comparison.to_csv(os.path.join(PROC_DIR, '..', 'main_results_v2.csv'))
    
    # Print formulas
    print("\n\n" + "=" * 65)
    print("  DISCOVERED FORMULAS")
    print("=" * 65)
    for regime_id in range(2):
        regime_name = regime_names[regime_id]
        print(f"\n  ▸ {regime_name} Regime:")
        for j, formula in enumerate(regime_formulas[regime_id][:3]):
            ic = regime_alphas[regime_id][j].fitness.values[0]
            print(f"    #{j+1} (IC={ic:.4f}): {formula}")
    
    print(f"\n  ▸ Vanilla GP:")
    for j, ind in enumerate(list(vanilla_hof[:3])):
        formula = vanilla_engine.get_formula(ind)
        ic = ind.fitness.values[0]
        print(f"    #{j+1} (IC={ic:.4f}): {formula}")
    
    return results


if __name__ == "__main__":
    results = run_improved_system()
