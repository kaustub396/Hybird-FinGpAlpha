"""
Cross-Market Generalisation -- Phase 3: Full HMM-GP Alpha Experiment on S&P 500

Re-trains the ENTIRE pipeline from scratch on S&P 500 data:
  1. HMM regime detection on ^GSPC index  (train: 2007-2018)
  2. GP evolution per regime with ICIR fitness
  3. Out-of-sample evaluation (2019-2025)
  4. Baseline comparisons

This demonstrates that the METHODOLOGY generalises,
not that the NIFTY-50 formulas transfer.

All paths point to data/raw_sp500 and data/processed_sp500
-- original NIFTY-50 data is never touched.
"""

import os
import sys
import pickle
import time
import pandas as pd
import numpy as np
from copy import deepcopy

# ---- Paths ------------------------------------------------------------------
BASE_DIR = r"C:\Users\EV-Car\Main_Project_2_Review1"
SP500_RAW_DIR  = os.path.join(BASE_DIR, "data", "raw_sp500")
SP500_PROC_DIR = os.path.join(BASE_DIR, "data", "processed_sp500")
RESULTS_DIR    = os.path.join(BASE_DIR, "data", "results_sp500")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---- Add project root to path so we can import existing modules -------------
sys.path.insert(0, BASE_DIR)

from regime_detector import RegimeDetector
from gp_engine import GPAlphaEngine
from evaluation import AlphaEvaluator, compare_alphas
from baselines import (momentum_alpha, reversal_alpha,
                       mean_reversion_alpha, low_volatility_alpha,
                       trend_alpha, combined_alpha)


# =============================================================================
# Load S&P 500 data
# =============================================================================
def load_sp500_data():
    """Load S&P 500 panel and index data."""
    print("Loading S&P 500 data...")

    # Panel (features)
    panel_path = os.path.join(SP500_PROC_DIR, "panel_sp500.pkl")
    with open(panel_path, "rb") as f:
        panel = pickle.load(f)

    sample_df = list(panel.values())[0]
    n_dates = len(sample_df)
    n_stocks = len(sample_df.columns)
    print(f"  Panel loaded: {n_dates} dates x {n_stocks} stocks")
    print(f"  Date range: {sample_df.index[0].date()} -> {sample_df.index[-1].date()}")

    # Index data for regime detection
    # yfinance saves MultiIndex headers: row0=colnames, row1=Ticker, row2=Date/NaN
    index_path = os.path.join(SP500_RAW_DIR, "SP500_INDEX.csv")
    index_df = pd.read_csv(index_path, header=0, skiprows=[1, 2])
    # The first column is named 'Price' but it is actually the Date
    index_df.rename(columns={"Price": "Date"}, inplace=True)
    # Coerce all numeric columns
    for col in ["Close", "High", "Low", "Open", "Volume"]:
        if col in index_df.columns:
            index_df[col] = pd.to_numeric(index_df[col], errors="coerce")
    index_df = index_df.dropna(subset=["Close"])
    print(f"  Index data: {len(index_df)} rows")

    return panel, index_df


# =============================================================================
# Main experiment
# =============================================================================
def run_sp500_experiment():
    """Run the full HMM-GP Alpha experiment on S&P 500."""

    total_start = time.time()

    panel, index_df = load_sp500_data()

    # ---- Train / Test split (same as NIFTY-50 paper) ----
    train_end = "2018-12-31"
    panel_dates = list(panel.values())[0].index
    train_dates = panel_dates[panel_dates <= pd.to_datetime(train_end)]
    test_dates  = panel_dates[panel_dates > pd.to_datetime(train_end)]

    # Filter to 2007+ to match paper (avoid pre-2007 data where some stocks are missing)
    train_dates = train_dates[train_dates >= pd.to_datetime("2007-01-01")]

    print(f"\n  Train: {train_dates[0].date()} -> {train_dates[-1].date()} ({len(train_dates)} days)")
    print(f"  Test:  {test_dates[0].date()} -> {test_dates[-1].date()} ({len(test_dates)} days)")

    # =================================================================
    # STEP 1: Regime Detection on S&P 500 Index
    # =================================================================
    print("\n" + "=" * 65)
    print("  STEP 1: HMM Regime Detection on S&P 500 (^GSPC)")
    print("=" * 65)

    # Use 2-regime model (Bull/Bear) as in the paper
    detector = RegimeDetector(n_regimes=2, random_state=42)

    # Train HMM only on training period index data
    idx_df = index_df.copy()
    if "Date" not in idx_df.columns:
        idx_df = idx_df.reset_index()
        idx_df.rename(columns={idx_df.columns[0]: "Date"}, inplace=True)
    idx_df["Date"] = pd.to_datetime(idx_df["Date"])

    # Filter index to training period
    train_idx = idx_df[idx_df["Date"] <= pd.to_datetime(train_end)].copy()
    detector.fit(train_idx)
    detector.print_summary()

    # Get regime labels for ALL dates (train + test)
    all_labels = detector.predict(idx_df)

    # Align with panel dates
    train_common = train_dates.intersection(all_labels.index)
    regime_labels = all_labels.loc[train_common]

    regime_names = detector._get_regime_names()
    print(f"\n  Training regime distribution:")
    for i, name in enumerate(regime_names):
        n = (regime_labels == i).sum()
        print(f"    {name}: {n} days ({n/len(regime_labels)*100:.1f}%)")

    # =================================================================
    # STEP 2: GP Evolution per Regime
    # =================================================================
    print("\n" + "=" * 65)
    print("  STEP 2: Regime-Conditioned GP Evolution")
    print("=" * 65)

    gp_params = {
        "population_size": 500,
        "tournament_size": 5,
        "max_depth": 6,
        "cx_prob": 0.7,
        "mut_prob": 0.2,
        "parsimony_weight": 0.001,
    }
    top_k = 3

    regime_engines = {}
    regime_alphas = {}
    regime_formulas = {}

    for regime_id in range(2):
        rname = regime_names[regime_id]
        regime_dates = regime_labels[regime_labels == regime_id].index

        if len(regime_dates) < 100:
            print(f"\n  Skipping {rname}: only {len(regime_dates)} dates")
            continue

        print(f"\n{'-' * 55}")
        print(f"  Regime: {rname} ({len(regime_dates)} training days)")
        print(f"{'-' * 55}")

        engine = GPAlphaEngine(
            panel,
            **gp_params,
            random_state=42 + regime_id
        )

        hof = engine.evolve(
            target="fwd_ret_20d",
            n_gen=50,
            date_mask=regime_dates,
            verbose=True,
            elite_size=top_k
        )

        regime_engines[regime_id] = engine
        regime_alphas[regime_id] = list(hof[:top_k])
        regime_formulas[regime_id] = [
            engine.get_formula(ind) for ind in hof[:top_k]
        ]

        print(f"\n  Top {top_k} formulas for {rname}:")
        for j, formula in enumerate(regime_formulas[regime_id]):
            ic = hof[j].fitness.values[0]
            print(f"    #{j+1} (fitness={ic:.4f}): {formula}")

    # =================================================================
    # STEP 3: Vanilla GP Baseline
    # =================================================================
    print(f"\n{'-' * 55}")
    print(f"  Vanilla GP Baseline (no regime conditioning)")
    print(f"{'-' * 55}")

    vanilla_engine = GPAlphaEngine(
        panel,
        **gp_params,
        random_state=42 + 99
    )

    vanilla_hof = vanilla_engine.evolve(
        target="fwd_ret_20d",
        n_gen=50,
        date_mask=train_common,
        verbose=True,
        elite_size=top_k
    )

    vanilla_alphas = list(vanilla_hof[:top_k])

    print(f"\n  Top {top_k} vanilla GP formulas:")
    for j, ind in enumerate(vanilla_alphas):
        formula = vanilla_engine.get_formula(ind)
        ic = ind.fitness.values[0]
        print(f"    #{j+1} (fitness={ic:.4f}): {formula}")

    # =================================================================
    # STEP 4: Out-of-Sample Evaluation (2019-2025)
    # =================================================================
    print("\n\n" + "=" * 65)
    print("  OUT-OF-SAMPLE EVALUATION (2019-2025) -- S&P 500")
    print("=" * 65)

    evaluator = AlphaEvaluator(panel, transaction_cost=0.001, n_quantiles=5)
    results = {}

    # ---- Regime-Aware GP (soft combination) ----
    print("\n" + "#" * 55)
    print("  EVALUATING: HMM-GP Alpha (Regime-Aware)")
    print("#" * 55)

    regime_proba = detector.predict_proba(idx_df)
    regime_proba = regime_proba.reindex(test_dates).fillna(0.5)

    target_df = panel["fwd_ret_20d"]
    stocks = target_df.columns
    combined_alpha_scores = pd.DataFrame(0.0, index=test_dates, columns=stocks)

    for regime_id, alphas in regime_alphas.items():
        engine = regime_engines[regime_id]
        # Equal-weight the top_k formulas within each regime
        regime_alpha = pd.DataFrame(0.0, index=test_dates, columns=stocks)
        for ind in alphas:
            a = engine.compute_alpha(ind, date_mask=test_dates)
            row_mean = a.mean(axis=1)
            row_std = a.std(axis=1).replace(0, np.nan)
            z = a.sub(row_mean, axis=0).div(row_std, axis=0).fillna(0)
            regime_alpha += z / len(alphas)

        # Weight by regime probability (soft combination)
        prob_col = regime_proba.iloc[:, regime_id]
        prob_aligned = prob_col.reindex(test_dates).fillna(1.0 / 2)
        combined_alpha_scores += regime_alpha.mul(prob_aligned, axis=0)

    results["HMM-GP Alpha (Ours)"] = evaluator.evaluate(
        combined_alpha_scores, target="fwd_ret_20d",
        holding_period=20, verbose=True
    )

    # ---- Vanilla GP ----
    print("\n" + "#" * 55)
    print("  EVALUATING: Vanilla GP (No Regime)")
    print("#" * 55)

    vanilla_combined = pd.DataFrame(0.0, index=test_dates, columns=stocks)
    for ind in vanilla_alphas:
        a = vanilla_engine.compute_alpha(ind, date_mask=test_dates)
        row_mean = a.mean(axis=1)
        row_std = a.std(axis=1).replace(0, np.nan)
        z = a.sub(row_mean, axis=0).div(row_std, axis=0).fillna(0)
        vanilla_combined += z / len(vanilla_alphas)

    results["Vanilla GP"] = evaluator.evaluate(
        vanilla_combined, target="fwd_ret_20d",
        holding_period=20, verbose=True
    )

    # ---- Baselines ----
    baseline_funcs = {
        "Momentum (12-1M)": momentum_alpha,
        "Mean Reversion": mean_reversion_alpha,
        "Trend (200-DMA)": trend_alpha,
    }

    for name, func in baseline_funcs.items():
        print(f"\n  Baseline: {name}")
        alpha = func(panel)
        alpha_test = alpha.reindex(test_dates)
        results[name] = evaluator.evaluate(
            alpha_test, target="fwd_ret_20d",
            holding_period=20, verbose=True
        )

    # =================================================================
    # STEP 5: Comparison Table
    # =================================================================
    print("\n\n" + "=" * 80)
    print("  S&P 500 CROSS-MARKET GENERALISATION RESULTS")
    print("=" * 80)

    comparison = compare_alphas(results)
    print(comparison.to_string())

    # Save results
    comparison.to_csv(os.path.join(RESULTS_DIR, "sp500_main_results.csv"))
    print(f"\n  Results saved -> {RESULTS_DIR}/sp500_main_results.csv")

    # ---- Save discovered formulas ----
    formulas_path = os.path.join(RESULTS_DIR, "sp500_discovered_formulas.txt")
    with open(formulas_path, "w") as f:
        f.write("S&P 500 -- Discovered Formulas by Regime\n")
        f.write("=" * 65 + "\n\n")
        for regime_id, formulas in regime_formulas.items():
            rname = regime_names[regime_id]
            f.write(f"{rname} Regime:\n")
            for j, formula in enumerate(formulas):
                ic = regime_alphas[regime_id][j].fitness.values[0]
                f.write(f"  #{j+1} (fitness={ic:.4f}): {formula}\n")
            f.write("\n")
        f.write("Vanilla GP (no regime):\n")
        for j, ind in enumerate(vanilla_alphas):
            formula = vanilla_engine.get_formula(ind)
            ic = ind.fitness.values[0]
            f.write(f"  #{j+1} (fitness={ic:.4f}): {formula}\n")

    print(f"  Formulas saved -> {formulas_path}")

    elapsed = time.time() - total_start
    print(f"\n  Total experiment time: {elapsed:.0f} seconds ({elapsed/60:.1f} minutes)")
    print("=" * 65)

    return results, regime_formulas


if __name__ == "__main__":
    results, formulas = run_sp500_experiment()
