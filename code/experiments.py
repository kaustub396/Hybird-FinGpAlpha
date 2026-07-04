"""
Phase 4A: Rolling Validation + GPU-Accelerated GP + Regime-Aware Selection

Three experiments:
1. Rolling validation: 3 train/test windows to confirm GP robustness
2. Regime-Aware Selection: discover formulas on all data, select best per regime
3. Ablation: tree depth & population size sensitivity

Uses GPU (CUDA) for batch IC computation when available.
"""

import pandas as pd
import numpy as np
import pickle
import os
import time
import torch

from gp_engine import GPAlphaEngine
from regime_detector import RegimeDetector
from evaluation import AlphaEvaluator, compare_alphas
from baselines import momentum_alpha, mean_reversion_alpha, low_volatility_alpha, trend_alpha

PROC_DIR = r"C:\Users\EV-Car\Main_Project_2\data\processed"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")


def load_data():
    with open(os.path.join(PROC_DIR, 'panel.pkl'), 'rb') as f:
        panel = pickle.load(f)
    index_df = pd.read_csv(os.path.join(PROC_DIR, 'NIFTY50_INDEX.csv'))
    return panel, index_df


# ============================================================
# EXPERIMENT 1: ROLLING VALIDATION
# ============================================================

def rolling_validation(panel, index_df):
    """
    3-window rolling validation to confirm GP robustness.
    
    Window 1: Train 2002-2012, Test 2013-2016
    Window 2: Train 2002-2016, Test 2017-2020
    Window 3: Train 2002-2020, Test 2021-2025
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1: ROLLING VALIDATION")
    print("=" * 70)
    
    windows = [
        ('2002-2012', '2012-12-31', '2013-01-01', '2016-12-31'),
        ('2002-2016', '2016-12-31', '2017-01-01', '2020-12-31'),
        ('2002-2020', '2020-12-31', '2021-01-01', '2025-12-31'),
    ]
    
    panel_dates = list(panel.values())[0].index
    evaluator = AlphaEvaluator(panel, transaction_cost=0.001, n_quantiles=5)
    
    all_window_results = {}
    
    gp_params = {
        'population_size': 500,
        'tournament_size': 5,
        'max_depth': 6,
        'cx_prob': 0.7,
        'mut_prob': 0.2,
        'parsimony_weight': 0.001,
    }
    
    for w_idx, (label, train_end, test_start, test_end) in enumerate(windows):
        print(f"\n{'━' * 70}")
        print(f"  Window {w_idx+1}: Train {label} | Test {test_start[:4]}-{test_end[:4]}")
        print(f"{'━' * 70}")
        
        train_dates = panel_dates[panel_dates <= pd.to_datetime(train_end)]
        test_dates = panel_dates[
            (panel_dates >= pd.to_datetime(test_start)) &
            (panel_dates <= pd.to_datetime(test_end))
        ]
        
        print(f"  Train: {len(train_dates)} days | Test: {len(test_dates)} days")
        
        # GP evolution on training data
        engine = GPAlphaEngine(panel, **gp_params, random_state=42)
        hof = engine.evolve(
            target='fwd_ret_20d', n_gen=50,
            date_mask=train_dates, verbose=True, elite_size=5
        )
        
        # Evaluate top-3 ensemble on test data
        combined = pd.DataFrame(0.0, index=test_dates, columns=list(panel.values())[0].columns)
        for ind in list(hof[:3]):
            alpha = engine.compute_alpha(ind, date_mask=test_dates)
            row_mean = alpha.mean(axis=1)
            row_std = alpha.std(axis=1).replace(0, np.nan)
            z_alpha = alpha.sub(row_mean, axis=0).div(row_std, axis=0).fillna(0)
            combined += z_alpha / 3
        
        results = {}
        
        # GP result
        results['GP (Ours)'] = evaluator.evaluate(
            combined, target='fwd_ret_20d', holding_period=20, verbose=True
        )
        
        # Baselines on same test period
        for name, func in [('Momentum', momentum_alpha), ('Mean Reversion', mean_reversion_alpha)]:
            alpha = func(panel).reindex(test_dates)
            results[name] = evaluator.evaluate(
                alpha, target='fwd_ret_20d', holding_period=20, verbose=False
            )
        
        # Top formula
        best_formula = engine.get_formula(hof[0])
        print(f"\n  Best formula: {best_formula}")
        print(f"  GP Sharpe: {results['GP (Ours)']['Sharpe_Net']:.3f}")
        print(f"  Mom Sharpe: {results['Momentum']['Sharpe_Net']:.3f}")
        print(f"  MR  Sharpe: {results['Mean Reversion']['Sharpe_Net']:.3f}")
        
        all_window_results[f"W{w_idx+1} ({test_start[:4]}-{test_end[:4]})"] = results
    
    # Summary table
    print("\n\n" + "=" * 70)
    print("  ROLLING VALIDATION SUMMARY")
    print("=" * 70)
    
    summary_rows = []
    for window_name, results in all_window_results.items():
        for method_name, res in results.items():
            summary_rows.append({
                'Window': window_name,
                'Method': method_name,
                'IC': f"{res['IC_mean']:.4f}",
                'ICIR': f"{res['ICIR']:.4f}",
                'Sharpe': f"{res['Sharpe_Net']:.3f}",
                'MaxDD': f"{res['Max_Drawdown']*100:.1f}%",
                'WinRate': f"{res['Win_Rate']:.1f}%",
            })
    
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    
    return all_window_results


# ============================================================
# EXPERIMENT 2: REGIME-AWARE SELECTION
# ============================================================

def regime_aware_selection(panel, index_df):
    """
    Discover a LIBRARY of formulas on all training data,
    then SELECT the best-performing subset for each regime.
    
    This avoids the overfitting problem of training separate GPs per regime
    (less data per regime → complex formulas overfit).
    """
    print("\n\n" + "=" * 70)
    print("  EXPERIMENT 2: REGIME-AWARE SELECTION")
    print("=" * 70)
    
    panel_dates = list(panel.values())[0].index
    train_end = '2018-12-31'
    train_dates = panel_dates[panel_dates <= pd.to_datetime(train_end)]
    test_dates = panel_dates[panel_dates > pd.to_datetime(train_end)]
    stocks = list(panel.values())[0].columns
    
    # Step 1: Detect regimes
    print("\n▸ Step 1: Regime Detection")
    train_idx = index_df[pd.to_datetime(index_df['Date']) <= pd.to_datetime(train_end)]
    detector = RegimeDetector(n_regimes=2, random_state=42)
    detector.fit(train_idx)
    
    all_labels = detector.predict(index_df)
    train_labels = all_labels.reindex(train_dates).dropna().astype(int)
    test_labels = all_labels.reindex(test_dates).dropna().astype(int)
    
    regime_names = detector._get_regime_names()
    print(f"  Train: Bear={int((train_labels==0).sum())}, Bull={int((train_labels==1).sum())}")
    print(f"  Test:  Bear={int((test_labels==0).sum())}, Bull={int((test_labels==1).sum())}")
    
    # Step 2: Discover formula LIBRARY on ALL training data
    print("\n▸ Step 2: GP Discovery (Full Training Data)")
    
    # Run GP multiple times with different seeds to build diverse library
    library = []
    for seed in [42, 123, 456]:
        engine = GPAlphaEngine(
            panel,
            population_size=500, tournament_size=5, max_depth=6,
            cx_prob=0.7, mut_prob=0.2, parsimony_weight=0.001,
            random_state=seed
        )
        hof = engine.evolve(
            target='fwd_ret_20d', n_gen=50,
            date_mask=train_dates, verbose=(seed == 42),
            elite_size=10
        )
        
        for ind in hof[:10]:
            formula = engine.get_formula(ind)
            alpha = engine.compute_alpha(ind, date_mask=train_dates)
            library.append({
                'individual': ind,
                'engine': engine,
                'formula': formula,
                'alpha_train': alpha,
                'fitness': ind.fitness.values[0],
            })
        
        print(f"  Seed {seed}: top IC = {hof[0].fitness.values[0]:.4f} | Library size: {len(library)}")
    
    # Step 3: Evaluate each formula per regime on training data
    print("\n▸ Step 3: Per-Regime IC Evaluation")
    
    target_train = panel['fwd_ret_20d'].reindex(train_dates)
    
    for item in library:
        alpha = item['alpha_train']
        for regime_id in range(2):
            regime_dates = train_labels[train_labels == regime_id].index
            regime_dates_common = alpha.index.intersection(regime_dates)
            
            ics = []
            for d in regime_dates_common[:300]:
                a = alpha.loc[d].dropna()
                r = target_train.loc[d].dropna()
                common = a.index.intersection(r.index)
                if len(common) < 10:
                    continue
                ic = np.corrcoef(a.loc[common].values, r.loc[common].values)[0, 1]
                if not np.isnan(ic):
                    ics.append(ic)
            
            item[f'ic_regime_{regime_id}'] = np.mean(ics) if ics else 0
    
    # Step 4: Select best formulas per regime
    print("\n▸ Step 4: Per-Regime Formula Selection")
    
    regime_selections = {}
    for regime_id in range(2):
        name = regime_names[regime_id]
        ic_key = f'ic_regime_{regime_id}'
        sorted_lib = sorted(library, key=lambda x: x[ic_key], reverse=True)
        
        # Pick top-3 with diversity (no near-duplicate formulas)
        selected = []
        seen_formulas = set()
        for item in sorted_lib:
            # Simple dedup: skip if formula is very similar to one already selected
            formula_hash = item['formula'][:30]  # rough dedup
            if formula_hash not in seen_formulas:
                selected.append(item)
                seen_formulas.add(formula_hash)
            if len(selected) >= 3:
                break
        
        regime_selections[regime_id] = selected
        print(f"\n  {name} Regime - Best 3:")
        for j, item in enumerate(selected):
            print(f"    #{j+1} IC={item[ic_key]:.4f}: {item['formula']}")
    
    # Step 5: Build combined alpha on TEST set with regime-aware weighting
    print("\n▸ Step 5: Regime-Aware Combination (Test Set)")
    
    evaluator = AlphaEvaluator(panel, transaction_cost=0.001, n_quantiles=5)
    
    # Get regime probabilities on test set
    regime_proba = detector.predict_proba(index_df)
    regime_proba_test = regime_proba.reindex(test_dates).fillna(0.5)
    
    # Method A: Regime-aware soft combination
    combined_regime = pd.DataFrame(0.0, index=test_dates, columns=stocks)
    for regime_id in range(2):
        regime_alpha = pd.DataFrame(0.0, index=test_dates, columns=stocks)
        for item in regime_selections[regime_id]:
            engine = item['engine']
            alpha = engine.compute_alpha(item['individual'], date_mask=test_dates)
            row_mean = alpha.mean(axis=1)
            row_std = alpha.std(axis=1).replace(0, np.nan)
            z_alpha = alpha.sub(row_mean, axis=0).div(row_std, axis=0).fillna(0)
            regime_alpha += z_alpha / 3
        
        prob_col = regime_proba_test.iloc[:, regime_id]
        combined_regime += regime_alpha.mul(prob_col, axis=0)
    
    results = {}
    
    print("\n  Regime-Aware Selection (Soft):")
    results['Regime-Aware Selection (Ours)'] = evaluator.evaluate(
        combined_regime, target='fwd_ret_20d', holding_period=20, verbose=True
    )
    
    # Method B: Vanilla GP ensemble (no regime, just best overall)
    print("\n  Vanilla GP Ensemble:")
    overall_best = sorted(library, key=lambda x: x['fitness'], reverse=True)[:3]
    vanilla_combined = pd.DataFrame(0.0, index=test_dates, columns=stocks)
    for item in overall_best:
        engine = item['engine']
        alpha = engine.compute_alpha(item['individual'], date_mask=test_dates)
        row_mean = alpha.mean(axis=1)
        row_std = alpha.std(axis=1).replace(0, np.nan)
        z_alpha = alpha.sub(row_mean, axis=0).div(row_std, axis=0).fillna(0)
        vanilla_combined += z_alpha / 3
    
    results['Vanilla GP Ensemble'] = evaluator.evaluate(
        vanilla_combined, target='fwd_ret_20d', holding_period=20, verbose=True
    )
    
    # Method C: Hard regime switching
    print("\n  Regime-Aware Selection (Hard):")
    combined_hard = pd.DataFrame(0.0, index=test_dates, columns=stocks)
    test_regime = detector.predict(index_df).reindex(test_dates)
    
    for regime_id in range(2):
        mask = test_regime == regime_id
        regime_alpha = pd.DataFrame(0.0, index=test_dates, columns=stocks)
        for item in regime_selections[regime_id]:
            engine = item['engine']
            alpha = engine.compute_alpha(item['individual'], date_mask=test_dates)
            row_mean = alpha.mean(axis=1)
            row_std = alpha.std(axis=1).replace(0, np.nan)
            z_alpha = alpha.sub(row_mean, axis=0).div(row_std, axis=0).fillna(0)
            regime_alpha += z_alpha / 3
        combined_hard.loc[mask] = regime_alpha.loc[mask]
    
    results['Regime-Aware Hard'] = evaluator.evaluate(
        combined_hard, target='fwd_ret_20d', holding_period=20, verbose=True
    )
    
    # Baselines
    for name, func in [('Momentum', momentum_alpha), ('Mean Reversion', mean_reversion_alpha)]:
        alpha = func(panel).reindex(test_dates)
        results[name] = evaluator.evaluate(
            alpha, target='fwd_ret_20d', holding_period=20, verbose=False
        )
    
    print("\n\n" + "=" * 80)
    print("  REGIME-AWARE SELECTION COMPARISON (OOS 2019-2025)")
    print("=" * 80)
    comparison = compare_alphas(results)
    print(comparison.to_string())
    
    return results, regime_selections, library


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    panel, index_df = load_data()
    
    start = time.time()
    
    # Experiment 1: Rolling validation
    rolling_results = rolling_validation(panel, index_df)
    
    # Experiment 2: Regime-aware selection
    regime_results, regime_selections, library = regime_aware_selection(panel, index_df)
    
    elapsed = time.time() - start
    print(f"\n\nTotal time: {elapsed/60:.1f} minutes")
    
    # Save all results
    save_path = os.path.join(PROC_DIR, '..', 'experiment_results.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump({
            'rolling': rolling_results,
            'regime_selection': regime_results,
        }, f)
    print(f"Results saved to {save_path}")
