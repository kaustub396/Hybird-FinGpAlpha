import sys
sys.path.insert(0, r"C:\Users\EV-Car\Main_Project_2")

import pandas as pd
import numpy as np
import pickle
import os

from regime_detector import RegimeDetector
from gp_engine import GPAlphaEngine
from evaluation import AlphaEvaluator, compare_alphas
from baselines import (
    momentum_alpha, mean_reversion_alpha,
    low_volatility_alpha, trend_alpha
)

PROC_DIR = r"C:\Users\EV-Car\Main_Project_2\data\processed"


def run_original_baseline():
    with open(os.path.join(PROC_DIR, 'panel.pkl'), 'rb') as f:
        panel = pickle.load(f)
    index_df = pd.read_csv(os.path.join(PROC_DIR, 'NIFTY50_INDEX.csv'))

    panel_dates = list(panel.values())[0].index
    train_end = '2018-12-31'
    train_dates = panel_dates[panel_dates <= pd.to_datetime(train_end)]
    test_dates = panel_dates[panel_dates > pd.to_datetime(train_end)]

    print(f"Train: {train_dates[0].date()} to {train_dates[-1].date()} ({len(train_dates)} days)")
    print(f"Test:  {test_dates[0].date()} to {test_dates[-1].date()} ({len(test_dates)} days)")

    train_idx_df = index_df[pd.to_datetime(index_df['Date']) <= pd.to_datetime(train_end)]

    detector = RegimeDetector(n_regimes=2, random_state=42)
    detector.fit(train_idx_df)

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

    evaluator = AlphaEvaluator(panel, transaction_cost=0.001, n_quantiles=5)

    regime_proba = detector.predict_proba(index_df)
    regime_proba_test = regime_proba.reindex(test_dates).fillna(0.5)

    results = {}
    stocks = list(panel.values())[0].columns

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

    results['Original GP (HMM Regime)'] = evaluator.evaluate(
        combined_soft, target='fwd_ret_20d', holding_period=20, verbose=True
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
    print("  ORIGINAL GP BASELINE RESULTS (Out-of-Sample 2019-2025)")
    print("=" * 85)
    comparison = compare_alphas(results)
    print(comparison.to_string())

    comparison.to_csv(r"C:\Users\EV-Car\Main-Project_3\gp\original_baseline_results.csv")
    print("\nSaved to Main-Project_3/gp/original_baseline_results.csv")

    return results


if __name__ == "__main__":
    results = run_original_baseline()
