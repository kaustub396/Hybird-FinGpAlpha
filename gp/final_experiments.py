"""
Phase 4B: Final Experiments with Improved GP Engine

Key improvements over previous runs:
1. Fitness = ICIR-based (Rank IC consistency), not mean IC
2. Dispersion check rejects near-constant formulas
3. 500-date sampling (not 200) for fitness evaluation
4. Multi-seed runs for robustness
5. Train/Validation split within training period to avoid overfitting

Experiments:
A. Single-window OOS with improved GP
B. Rolling validation (3 windows)  
C. Regime-Aware Selection with improved GP
D. Ablation: GP depth, parsimony weight
"""

import pandas as pd
import numpy as np
import pickle
import os
import time

from gp_engine import GPAlphaEngine
from regime_detector import RegimeDetector
from evaluation import AlphaEvaluator, compare_alphas
from baselines import momentum_alpha, mean_reversion_alpha, trend_alpha

PROC_DIR = r"C:\Users\EV-Car\Main_Project_2\data\processed"


def load_data():
    with open(os.path.join(PROC_DIR, 'panel.pkl'), 'rb') as f:
        panel = pickle.load(f)
    index_df = pd.read_csv(os.path.join(PROC_DIR, 'NIFTY50_INDEX.csv'))
    return panel, index_df


def run_gp_with_validation(panel, train_dates, n_gen=50, n_seeds=3,
                           pop_size=500, max_depth=6, parsimony=0.002):
    """
    Run GP with multiple seeds and select formulas validated on held-out portion.
    
    Split training data into fit (first 80%) and validate (last 20%).
    Only keep formulas whose validation Rank IC > 0.
    """
    # Split training into fit/validate
    split_idx = int(len(train_dates) * 0.8)
    fit_dates = train_dates[:split_idx]
    val_dates = train_dates[split_idx:]
    
    print(f"  Fit: {len(fit_dates)} days | Validate: {len(val_dates)} days")
    
    gp_params = {
        'population_size': pop_size,
        'tournament_size': 5,
        'max_depth': max_depth,
        'cx_prob': 0.7,
        'mut_prob': 0.25,
        'parsimony_weight': parsimony,
    }
    
    all_individuals = []
    
    for seed in range(n_seeds):
        actual_seed = 42 + seed * 111
        engine = GPAlphaEngine(panel, **gp_params, random_state=actual_seed)
        hof = engine.evolve(
            target='fwd_ret_20d', n_gen=n_gen,
            date_mask=fit_dates, verbose=(seed == 0),
            elite_size=10
        )
        
        # Validate each Hall of Fame individual on validation dates
        target_val = panel['fwd_ret_20d'].reindex(val_dates)
        
        for ind in hof[:10]:
            alpha_val = engine.compute_alpha(ind, date_mask=val_dates)
            
            # Compute validation Rank IC
            val_ics = []
            for d in val_dates[:200]:
                if d not in alpha_val.index or d not in target_val.index:
                    continue
                a = alpha_val.loc[d].dropna()
                r = target_val.loc[d].dropna()
                common = a.index.intersection(r.index)
                if len(common) < 10:
                    continue
                from scipy import stats as scipy_stats
                rank_ic, _ = scipy_stats.spearmanr(a.loc[common].values, r.loc[common].values)
                if not np.isnan(rank_ic):
                    val_ics.append(rank_ic)
            
            val_rank_ic = np.mean(val_ics) if val_ics else 0
            val_icir = (np.mean(val_ics) / np.std(val_ics)) if (val_ics and np.std(val_ics) > 0) else 0
            
            all_individuals.append({
                'individual': ind,
                'engine': engine,
                'formula': engine.get_formula(ind),
                'train_fitness': ind.fitness.values[0],
                'val_rank_ic': val_rank_ic,
                'val_icir': val_icir,
                'tree_size': len(ind),
                'seed': actual_seed,
            })
        
        print(f"  Seed {actual_seed}: top fitness={hof[0].fitness.values[0]:.4f}")
    
    # Filter: keep only formulas with positive validation Rank IC
    valid = [x for x in all_individuals if x['val_rank_ic'] > 0.005]
    
    if len(valid) < 3:
        # If too few pass validation, take top by val_rank_ic regardless
        valid = sorted(all_individuals, key=lambda x: x['val_rank_ic'], reverse=True)[:5]
    
    # Sort by validation ICIR (best consistency)
    valid.sort(key=lambda x: x['val_icir'], reverse=True)
    
    print(f"\n  Validated formulas: {len(valid)} (of {len(all_individuals)} total)")
    for j, item in enumerate(valid[:5]):
        print(f"    #{j+1} val_IC={item['val_rank_ic']:.4f} val_ICIR={item['val_icir']:.4f} "
              f"size={item['tree_size']}: {item['formula'][:80]}")
    
    return valid


def build_ensemble_alpha(panel, validated_formulas, dates, top_n=3):
    """Build equal-weight ensemble of top validated formulas."""
    stocks = list(panel.values())[0].columns
    combined = pd.DataFrame(0.0, index=dates, columns=stocks)
    
    for item in validated_formulas[:top_n]:
        engine = item['engine']
        alpha = engine.compute_alpha(item['individual'], date_mask=dates)
        row_mean = alpha.mean(axis=1)
        row_std = alpha.std(axis=1).replace(0, np.nan)
        z_alpha = alpha.sub(row_mean, axis=0).div(row_std, axis=0).fillna(0)
        combined += z_alpha / top_n
    
    return combined


def main():
    panel, index_df = load_data()
    panel_dates = list(panel.values())[0].index
    evaluator = AlphaEvaluator(panel, transaction_cost=0.001, n_quantiles=5)
    
    start_time = time.time()
    
    # ============================================================
    # EXPERIMENT A: Main OOS (Train 2002-2018, Test 2019-2025)
    # ============================================================
    print("=" * 70)
    print("  EXPERIMENT A: MAIN OUT-OF-SAMPLE TEST")
    print("  Train: 2002-2018 | Test: 2019-2025")
    print("=" * 70)
    
    train_end = '2018-12-31'
    train_dates = panel_dates[panel_dates <= pd.to_datetime(train_end)]
    test_dates = panel_dates[panel_dates > pd.to_datetime(train_end)]
    
    validated = run_gp_with_validation(panel, train_dates, n_gen=50, n_seeds=3)
    
    # Build test alpha
    gp_alpha = build_ensemble_alpha(panel, validated, test_dates, top_n=3)
    
    results_a = {}
    print("\n  GP Ensemble (Improved):")
    results_a['GP (Ours)'] = evaluator.evaluate(
        gp_alpha, target='fwd_ret_20d', holding_period=20, verbose=True
    )
    
    # Baselines
    for name, func in [('Momentum', momentum_alpha), ('Mean Reversion', mean_reversion_alpha),
                        ('Trend (200-DMA)', trend_alpha)]:
        alpha = func(panel).reindex(test_dates)
        results_a[name] = evaluator.evaluate(
            alpha, target='fwd_ret_20d', holding_period=20, verbose=False
        )
    
    print("\n" + "=" * 80)
    print("  EXPERIMENT A RESULTS")
    print("=" * 80)
    comp_a = compare_alphas(results_a)
    print(comp_a.to_string())
    
    # ============================================================
    # EXPERIMENT B: Rolling Validation (3 windows)
    # ============================================================
    print("\n\n" + "=" * 70)
    print("  EXPERIMENT B: ROLLING VALIDATION")
    print("=" * 70)
    
    windows = [
        ('W1', '2012-12-31', '2013-01-01', '2016-12-31'),
        ('W2', '2016-12-31', '2017-01-01', '2020-12-31'),
        ('W3', '2020-12-31', '2021-01-01', '2025-12-31'),
    ]
    
    rolling_rows = []
    
    for label, tr_end, te_start, te_end in windows:
        print(f"\n{'━' * 55}")
        print(f"  {label}: Train→{tr_end} | Test {te_start}→{te_end}")
        print(f"{'━' * 55}")
        
        tr_dates = panel_dates[panel_dates <= pd.to_datetime(tr_end)]
        te_dates = panel_dates[
            (panel_dates >= pd.to_datetime(te_start)) &
            (panel_dates <= pd.to_datetime(te_end))
        ]
        
        validated_w = run_gp_with_validation(panel, tr_dates, n_gen=50, n_seeds=2)
        gp_alpha_w = build_ensemble_alpha(panel, validated_w, te_dates, top_n=3)
        
        res_gp = evaluator.evaluate(
            gp_alpha_w, target='fwd_ret_20d', holding_period=20, verbose=True
        )
        
        res_mom = evaluator.evaluate(
            momentum_alpha(panel).reindex(te_dates),
            target='fwd_ret_20d', holding_period=20, verbose=False
        )
        res_mr = evaluator.evaluate(
            mean_reversion_alpha(panel).reindex(te_dates),
            target='fwd_ret_20d', holding_period=20, verbose=False
        )
        
        for name, res in [('GP', res_gp), ('Momentum', res_mom), ('Mean Rev', res_mr)]:
            rolling_rows.append({
                'Window': label,
                'Method': name,
                'IC': f"{res['IC_mean']:.4f}",
                'Rank_IC': f"{res['Rank_IC_mean']:.4f}",
                'ICIR': f"{res['ICIR']:.4f}",
                'Sharpe': f"{res['Sharpe_Net']:.3f}",
                'MaxDD': f"{res['Max_Drawdown']*100:.1f}%",
                'AnnRet': f"{res['Ann_Return_Net']*100:.1f}%",
            })
    
    print("\n\n" + "=" * 80)
    print("  ROLLING VALIDATION SUMMARY")
    print("=" * 80)
    rolling_df = pd.DataFrame(rolling_rows)
    print(rolling_df.to_string(index=False))
    
    # ============================================================
    # EXPERIMENT C: Regime-Aware Selection
    # ============================================================
    print("\n\n" + "=" * 70)
    print("  EXPERIMENT C: REGIME-AWARE SELECTION")
    print("=" * 70)
    
    train_dates_c = panel_dates[panel_dates <= pd.to_datetime('2018-12-31')]
    test_dates_c = panel_dates[panel_dates > pd.to_datetime('2018-12-31')]
    stocks = list(panel.values())[0].columns
    
    # Regime detection
    train_idx = index_df[pd.to_datetime(index_df['Date']) <= pd.to_datetime('2018-12-31')]
    detector = RegimeDetector(n_regimes=2, random_state=42)
    detector.fit(train_idx)
    
    all_labels = detector.predict(index_df)
    train_labels = all_labels.reindex(train_dates_c).dropna().astype(int)
    test_labels = all_labels.reindex(test_dates_c).dropna().astype(int)
    
    regime_names = detector._get_regime_names()
    print(f"  Test: Bear={int((test_labels==0).sum())}, Bull={int((test_labels==1).sum())}")
    
    # Get validated formulas from Experiment A (already computed)
    # Evaluate each per regime on training data
    target_train = panel['fwd_ret_20d'].reindex(train_dates_c)
    
    for item in validated:
        engine = item['engine']
        alpha_train = engine.compute_alpha(item['individual'], date_mask=train_dates_c)
        
        for regime_id in range(2):
            regime_dates = train_labels[train_labels == regime_id].index
            common = alpha_train.index.intersection(regime_dates)
            
            ics = []
            for d in common[:300]:
                a = alpha_train.loc[d].dropna()
                r = target_train.loc[d].dropna()
                c = a.index.intersection(r.index)
                if len(c) < 10:
                    continue
                from scipy import stats as scipy_stats
                rank_ic, _ = scipy_stats.spearmanr(a.loc[c].values, r.loc[c].values)
                if not np.isnan(rank_ic):
                    ics.append(rank_ic)
            item[f'regime_{regime_id}_ic'] = np.mean(ics) if ics else 0
    
    # Select best per regime
    regime_proba_test = detector.predict_proba(index_df).reindex(test_dates_c).fillna(0.5)
    
    combined_regime = pd.DataFrame(0.0, index=test_dates_c, columns=stocks)
    
    for regime_id in range(2):
        ic_key = f'regime_{regime_id}_ic'
        best_for_regime = sorted(validated, key=lambda x: x.get(ic_key, 0), reverse=True)[:3]
        
        print(f"\n  Best for {regime_names[regime_id]}:")
        for j, item in enumerate(best_for_regime):
            print(f"    #{j+1} regime_IC={item.get(ic_key,0):.4f}: {item['formula'][:70]}")
        
        regime_alpha = pd.DataFrame(0.0, index=test_dates_c, columns=stocks)
        for item in best_for_regime:
            engine = item['engine']
            alpha = engine.compute_alpha(item['individual'], date_mask=test_dates_c)
            row_mean = alpha.mean(axis=1)
            row_std = alpha.std(axis=1).replace(0, np.nan)
            z_alpha = alpha.sub(row_mean, axis=0).div(row_std, axis=0).fillna(0)
            regime_alpha += z_alpha / 3
        
        prob_col = regime_proba_test.iloc[:, regime_id]
        combined_regime += regime_alpha.mul(prob_col, axis=0)
    
    results_c = {}
    
    print("\n  Regime-Aware Selection:")
    results_c['Regime-Aware GP (Ours)'] = evaluator.evaluate(
        combined_regime, target='fwd_ret_20d', holding_period=20, verbose=True
    )
    
    print("\n  Vanilla GP (from Exp A):")
    results_c['Vanilla GP'] = results_a['GP (Ours)']
    
    for name, func in [('Momentum', momentum_alpha), ('Mean Reversion', mean_reversion_alpha)]:
        alpha = func(panel).reindex(test_dates_c)
        results_c[name] = evaluator.evaluate(
            alpha, target='fwd_ret_20d', holding_period=20, verbose=False
        )
    
    print("\n" + "=" * 80)
    print("  REGIME-AWARE SELECTION COMPARISON")
    print("=" * 80)
    comp_c = compare_alphas(results_c)
    print(comp_c.to_string())
    
    # ============================================================
    # SAVE ALL RESULTS
    # ============================================================
    elapsed = time.time() - start_time
    print(f"\n\nTotal time: {elapsed/60:.1f} minutes")
    
    # Save
    all_results = {
        'experiment_a': results_a,
        'rolling': rolling_rows,
        'experiment_c': results_c,
        'discovered_formulas': [
            {'formula': v['formula'], 'val_ic': v['val_rank_ic'], 
             'val_icir': v['val_icir'], 'size': v['tree_size']}
            for v in validated[:10]
        ],
    }
    
    save_path = os.path.join(PROC_DIR, '..', 'final_results.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    main()
