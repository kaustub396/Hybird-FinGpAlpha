"""
Phase 3C: Regime-Aware Formulaic Alpha Discovery

This is the CORE NOVEL METHOD of the paper.

Architecture:
1. RegimeDetector identifies market regimes (Bear/Sideways/Bull) from NIFTY-50 index
2. GP evolves SEPARATE formula populations per regime
3. Dynamic Combination Layer weights regime-specific alphas by regime probability

Key Insight:
Standard alpha mining assumes stationarity — one formula works always.
Real markets switch between regimes where different strategies dominate.
By conditioning GP search on regimes, we discover regime-specific alphas
that are more robust and adaptive.

Novelty vs Prior Work:
- Alpha², QuantFactor, AlphaForge: single GP/RL search, no regime awareness
- This work: regime-conditioned search space = regime-specific formula discovery

Usage:
    from regime_gp import RegimeAwareAlphaDiscovery
    system = RegimeAwareAlphaDiscovery(panel, index_df)
    system.fit(target='fwd_ret_20d', n_gen=50)
    alpha_scores = system.predict(target='fwd_ret_20d')
"""

import pandas as pd
import numpy as np
import pickle
import os
import time
from copy import deepcopy

from regime_detector import RegimeDetector
from gp_engine import GPAlphaEngine
from evaluation import AlphaEvaluator, compare_alphas


class RegimeAwareAlphaDiscovery:
    """
    Main system: Regime-Aware Formulaic Alpha Discovery.
    
    Parameters
    ----------
    panel : dict
        Cross-sectional panel {feature: DataFrame(dates × stocks)}.
    index_df : DataFrame
        NIFTY-50 index OHLC data for regime detection.
    n_regimes : int
        Number of market regimes.
    gp_params : dict
        Parameters for GPAlphaEngine (population_size, max_depth, etc.).
    top_k : int
        Number of top formulas to keep per regime.
    combination : str
        How to combine regime-specific alphas:
        'hard' = use only current regime's alphas
        'soft' = weight by regime probability
        'learned' = optimize weights on validation data
    random_state : int
        Random seed.
    """
    
    def __init__(self, panel, index_df, n_regimes=3, gp_params=None,
                 top_k=3, combination='soft', random_state=42):
        
        self.panel = panel
        self.index_df = index_df
        self.n_regimes = n_regimes
        self.top_k = top_k
        self.combination = combination
        self.random_state = random_state
        
        # Default GP parameters
        self.gp_params = gp_params or {
            'population_size': 500,
            'tournament_size': 5,
            'max_depth': 6,
            'cx_prob': 0.7,
            'mut_prob': 0.2,
            'parsimony_weight': 0.001,
        }
        
        # Components (set during fit)
        self.regime_detector = None
        self.regime_engines = {}      # {regime_id: GPAlphaEngine}
        self.regime_alphas = {}       # {regime_id: [top_k individuals]}
        self.regime_formulas = {}     # {regime_id: [formula strings]}
        self.regime_weights = {}      # {regime_id: [weights for top_k]}
        self.regime_eval_results = {} # {regime_id: [eval results]}
        
        # Vanilla GP baseline (for ablation)
        self.vanilla_engine = None
        self.vanilla_alphas = None
    
    def fit(self, target='fwd_ret_20d', n_gen=50, train_end=None, verbose=True):
        """
        Fit the complete regime-aware alpha discovery system.
        
        Parameters
        ----------
        target : str
            Forward return target.
        n_gen : int
            GP generations per regime.
        train_end : str or None
            End date for training (dates after this are held out for testing).
            If None, uses all data for training.
        verbose : bool
            Print progress.
        
        Returns
        -------
        self
        """
        start_time = time.time()
        
        if verbose:
            print("=" * 65)
            print("  REGIME-AWARE FORMULAIC ALPHA DISCOVERY")
            print("  Training Phase")
            print("=" * 65)
        
        # ============================================
        # STEP 1: Detect Regimes
        # ============================================
        if verbose:
            print("\n▸ Step 1: Regime Detection")
        
        self.regime_detector = RegimeDetector(
            n_regimes=self.n_regimes,
            random_state=self.random_state
        )
        
        # Use only training data for regime detection if train_end specified
        if train_end is not None:
            train_idx = self.index_df[
                pd.to_datetime(self.index_df['Date']) <= pd.to_datetime(train_end)
            ]
            self.regime_detector.fit(train_idx)
        else:
            self.regime_detector.fit(self.index_df)
        
        if verbose:
            self.regime_detector.print_summary()
        
        # Get regime labels for panel dates
        all_labels = self.regime_detector.predict(self.index_df)
        
        # Restrict to training period
        panel_dates = list(self.panel.values())[0].index
        if train_end is not None:
            train_dates = panel_dates[panel_dates <= pd.to_datetime(train_end)]
        else:
            train_dates = panel_dates
        
        # Align regime labels with panel dates
        common_dates = train_dates.intersection(all_labels.index)
        regime_labels = all_labels.loc[common_dates]
        
        if verbose:
            print(f"\n  Training dates: {len(common_dates)}")
            regime_names = self.regime_detector._get_regime_names()
            for i, name in enumerate(regime_names):
                n = (regime_labels == i).sum()
                print(f"    {name}: {n} dates ({n/len(regime_labels)*100:.1f}%)")
        
        # ============================================
        # STEP 2: Regime-Conditioned GP Evolution
        # ============================================
        if verbose:
            print("\n▸ Step 2: Regime-Conditioned GP Evolution")
        
        regime_names = self.regime_detector._get_regime_names()
        
        for regime_id in range(self.n_regimes):
            regime_name = regime_names[regime_id]
            regime_dates = regime_labels[regime_labels == regime_id].index
            
            if len(regime_dates) < 100:
                if verbose:
                    print(f"\n  ⚠ Skipping {regime_name}: only {len(regime_dates)} dates")
                continue
            
            if verbose:
                print(f"\n{'─' * 55}")
                print(f"  Regime: {regime_name} ({len(regime_dates)} dates)")
                print(f"{'─' * 55}")
            
            # Create GP engine for this regime
            engine = GPAlphaEngine(
                self.panel,
                **self.gp_params,
                random_state=self.random_state + regime_id
            )
            
            # Evolve on regime-specific dates
            hof = engine.evolve(
                target=target,
                n_gen=n_gen,
                date_mask=regime_dates,
                verbose=verbose,
                elite_size=self.top_k
            )
            
            self.regime_engines[regime_id] = engine
            self.regime_alphas[regime_id] = list(hof[:self.top_k])
            self.regime_formulas[regime_id] = [
                engine.get_formula(ind) for ind in hof[:self.top_k]
            ]
            
            if verbose:
                print(f"\n  Top {self.top_k} formulas for {regime_name}:")
                for j, formula in enumerate(self.regime_formulas[regime_id]):
                    ic = hof[j].fitness.values[0]
                    print(f"    #{j+1} (IC={ic:.4f}): {formula}")
        
        # ============================================
        # STEP 3: Compute Regime-Specific Weights
        # ============================================
        if verbose:
            print(f"\n▸ Step 3: Computing Combination Weights")
        
        self._compute_weights(target, common_dates, regime_labels)
        
        # ============================================
        # STEP 4: Vanilla GP Baseline (for ablation)
        # ============================================
        if verbose:
            print(f"\n{'─' * 55}")
            print(f"  Vanilla GP Baseline (no regime conditioning)")
            print(f"{'─' * 55}")
        
        self.vanilla_engine = GPAlphaEngine(
            self.panel,
            **self.gp_params,
            random_state=self.random_state + 99
        )
        
        vanilla_hof = self.vanilla_engine.evolve(
            target=target,
            n_gen=n_gen,
            date_mask=common_dates,
            verbose=verbose,
            elite_size=self.top_k
        )
        
        self.vanilla_alphas = list(vanilla_hof[:self.top_k])
        
        if verbose:
            print(f"\n  Top {self.top_k} vanilla GP formulas:")
            for j, ind in enumerate(self.vanilla_alphas):
                formula = self.vanilla_engine.get_formula(ind)
                ic = ind.fitness.values[0]
                print(f"    #{j+1} (IC={ic:.4f}): {formula}")
        
        elapsed = time.time() - start_time
        if verbose:
            print(f"\n{'=' * 65}")
            print(f"  Training complete in {elapsed:.0f} seconds")
            print(f"{'=' * 65}")
        
        return self
    
    def _compute_weights(self, target, dates, regime_labels):
        """
        Compute combination weights for each regime's top-K alphas.
        
        Uses IC on training data as weight (higher IC = more weight).
        """
        target_df = self.panel[target]
        
        for regime_id, alphas in self.regime_alphas.items():
            engine = self.regime_engines[regime_id]
            weights = []
            
            for ind in alphas:
                alpha_scores = engine.compute_alpha(ind, date_mask=dates)
                
                # Compute IC on training dates
                ics = []
                for date in dates[:500]:  # Sample for speed
                    if date not in alpha_scores.index or date not in target_df.index:
                        continue
                    a = alpha_scores.loc[date].dropna()
                    r = target_df.loc[date].dropna()
                    common = a.index.intersection(r.index)
                    if len(common) < 10:
                        continue
                    ic = np.corrcoef(a.loc[common].values, r.loc[common].values)[0, 1]
                    if not np.isnan(ic):
                        ics.append(ic)
                
                mean_ic = np.mean(ics) if ics else 0
                weights.append(max(mean_ic, 0))  # Clip negative IC to 0
            
            # Normalize weights
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]
            else:
                weights = [1.0 / len(weights)] * len(weights)
            
            self.regime_weights[regime_id] = weights
    
    def predict(self, target='fwd_ret_20d', date_mask=None):
        """
        Generate combined alpha scores using regime-aware combination.
        
        Parameters
        ----------
        target : str
            For date alignment.
        date_mask : array-like or None
            Dates to predict on. If None, uses all panel dates.
        
        Returns
        -------
        DataFrame : Combined alpha scores (dates × stocks).
        """
        target_df = self.panel[target]
        if date_mask is not None:
            if isinstance(date_mask, pd.Index):
                dates = target_df.index.intersection(date_mask)
            else:
                dates = date_mask
        else:
            dates = target_df.index
        
        stocks = target_df.columns
        
        # Get regime probabilities for all dates
        regime_proba = self.regime_detector.predict_proba(self.index_df)
        regime_proba = regime_proba.reindex(dates).fillna(1.0 / self.n_regimes)
        
        # Initialize combined alpha
        combined = pd.DataFrame(0.0, index=dates, columns=stocks)
        
        for regime_id, alphas in self.regime_alphas.items():
            engine = self.regime_engines[regime_id]
            weights = self.regime_weights[regime_id]
            
            # Compute weighted sum of regime-specific alphas
            regime_alpha = pd.DataFrame(0.0, index=dates, columns=stocks)
            for ind, w in zip(alphas, weights):
                alpha_scores = engine.compute_alpha(ind, date_mask=dates)
                # Cross-sectional z-score
                row_mean = alpha_scores.mean(axis=1)
                row_std = alpha_scores.std(axis=1).replace(0, np.nan)
                z_scored = alpha_scores.sub(row_mean, axis=0).div(row_std, axis=0)
                z_scored = z_scored.fillna(0)
                regime_alpha += w * z_scored
            
            # Weight by regime probability
            if self.combination == 'soft':
                regime_prob_col = regime_proba.iloc[:, regime_id]
                regime_prob_aligned = regime_prob_col.reindex(dates).fillna(
                    1.0 / self.n_regimes
                )
                combined += regime_alpha.mul(regime_prob_aligned, axis=0)
            elif self.combination == 'hard':
                # Hard assignment: only use current regime's alpha
                current_regime = self.regime_detector.predict(self.index_df)
                current_regime = current_regime.reindex(dates)
                mask = current_regime == regime_id
                combined.loc[mask] += regime_alpha.loc[mask]
            else:
                combined += regime_alpha / self.n_regimes
        
        return combined
    
    def predict_vanilla(self, date_mask=None):
        """Generate alpha scores from vanilla GP (for ablation comparison)."""
        target_df = self.panel['fwd_ret_20d']
        if date_mask is not None:
            dates = target_df.index.intersection(date_mask) if isinstance(date_mask, pd.Index) else date_mask
        else:
            dates = target_df.index
        
        stocks = target_df.columns
        combined = pd.DataFrame(0.0, index=dates, columns=stocks)
        
        if self.vanilla_alphas is None:
            return combined
        
        for ind in self.vanilla_alphas:
            alpha_scores = self.vanilla_engine.compute_alpha(ind, date_mask=dates)
            row_mean = alpha_scores.mean(axis=1)
            row_std = alpha_scores.std(axis=1).replace(0, np.nan)
            z_scored = alpha_scores.sub(row_mean, axis=0).div(row_std, axis=0).fillna(0)
            combined += z_scored / len(self.vanilla_alphas)
        
        return combined

    def export_gp_signal(
        self,
        target='fwd_ret_20d',
        date_mask=None,
        method='mean_rank',
        quantile=0.1,
        normalize=True,
        resample_freq=None,
    ):
        """
        Export a scalar time-series GP signal from cross-sectional alpha scores.

        Parameters
        ----------
        target : str
            Forward return target used for date alignment in predict().
        date_mask : array-like or None
            Optional prediction dates.
        method : str
            Aggregation method:
            - 'mean_rank': mean cross-sectional rank, scaled to [-1, 1]
            - 'top_bottom_spread': top-quantile minus bottom-quantile spread
        quantile : float
            Tail quantile used when method='top_bottom_spread'.
        normalize : bool
            If True, z-score normalize the resulting signal over time.
        resample_freq : str or None
            Optional pandas frequency (e.g., 'W', 'M') for resampling.

        Returns
        -------
        Series
            Indexed by date with name 'gp_signal'.
        """
        alpha_df = self.predict(target=target, date_mask=date_mask)
        if alpha_df.empty:
            return pd.Series(dtype=float, name='gp_signal')

        if method == 'mean_rank':
            rank_pct = alpha_df.rank(axis=1, pct=True)
            signal = rank_pct.mean(axis=1) * 2.0 - 1.0
        elif method == 'top_bottom_spread':
            q = float(quantile)
            if q <= 0 or q >= 0.5:
                raise ValueError("quantile must be in (0, 0.5) for top_bottom_spread")
            top = alpha_df.quantile(1.0 - q, axis=1)
            bottom = alpha_df.quantile(q, axis=1)
            signal = top - bottom
        else:
            raise ValueError("method must be one of: 'mean_rank', 'top_bottom_spread'")

        signal = signal.sort_index()

        if resample_freq is not None:
            signal = signal.resample(resample_freq).mean()

        if normalize:
            std = signal.std()
            if std and not np.isclose(std, 0.0):
                signal = (signal - signal.mean()) / std
            else:
                signal = signal - signal.mean()

        signal.name = 'gp_signal'
        return signal
    
    def full_evaluation(self, evaluator, target='fwd_ret_20d',
                        holding_period=20, test_dates=None, verbose=True):
        """
        Complete evaluation: regime-GP vs vanilla-GP vs baselines.
        
        Parameters
        ----------
        evaluator : AlphaEvaluator
        target : str
        holding_period : int
        test_dates : Index or None
            If specified, evaluate only on these dates.
        verbose : bool
        
        Returns
        -------
        dict : {method_name: evaluation_results}
        """
        results = {}
        
        # 1. Regime-Aware GP (Full Model)
        if verbose:
            print("\n" + "#" * 55)
            print("  EVALUATING: Regime-Aware GP (Full Model)")
            print("#" * 55)
        
        regime_alpha = self.predict(target=target, date_mask=test_dates)
        results['Regime-Aware GP (Ours)'] = evaluator.evaluate(
            regime_alpha, target=target, holding_period=holding_period, verbose=verbose
        )
        
        # 2. Vanilla GP (Ablation: no regime conditioning)
        if verbose:
            print("\n" + "#" * 55)
            print("  EVALUATING: Vanilla GP (No Regime)")
            print("#" * 55)
        
        vanilla_alpha = self.predict_vanilla(date_mask=test_dates)
        results['Vanilla GP'] = evaluator.evaluate(
            vanilla_alpha, target=target, holding_period=holding_period, verbose=verbose
        )
        
        # 3. Individual regime alphas (for analysis)
        regime_names = self.regime_detector._get_regime_names()
        for regime_id, alphas in self.regime_alphas.items():
            name = regime_names[regime_id]
            if verbose:
                print(f"\n  Regime-specific: {name}")
            
            engine = self.regime_engines[regime_id]
            best_ind = alphas[0]
            alpha_scores = engine.compute_alpha(best_ind, date_mask=test_dates)
            results[f'GP ({name} only)'] = evaluator.evaluate(
                alpha_scores, target=target, holding_period=holding_period,
                verbose=verbose
            )
        
        return results
    
    def get_discovered_formulas(self):
        """Return all discovered formulas organized by regime."""
        regime_names = self.regime_detector._get_regime_names()
        summary = {}
        
        for regime_id, formulas in self.regime_formulas.items():
            name = regime_names[regime_id]
            summary[name] = []
            for j, formula in enumerate(formulas):
                ind = self.regime_alphas[regime_id][j]
                summary[name].append({
                    'rank': j + 1,
                    'formula': formula,
                    'fitness_ic': ind.fitness.values[0],
                    'tree_size': len(ind),
                    'weight': self.regime_weights[regime_id][j],
                })
        
        return summary
    
    def print_discovered_formulas(self):
        """Pretty-print all discovered formulas."""
        summary = self.get_discovered_formulas()
        
        print("\n" + "=" * 65)
        print("  DISCOVERED FORMULAS BY REGIME")
        print("=" * 65)
        
        for regime_name, formulas in summary.items():
            print(f"\n  ▸ {regime_name} Regime:")
            for f in formulas:
                print(f"    #{f['rank']} (IC={f['fitness_ic']:.4f}, "
                      f"size={f['tree_size']}, w={f['weight']:.3f})")
                print(f"       {f['formula']}")
    
    def save(self, path):
        """Save complete system to disk."""
        save_data = {
            'regime_detector': self.regime_detector,
            'regime_formulas': self.regime_formulas,
            'regime_weights': self.regime_weights,
            'n_regimes': self.n_regimes,
            'combination': self.combination,
            'gp_params': self.gp_params,
        }
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"System saved to {path}")
    
    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)


# ============================================================
# MAIN: Full Training + Evaluation Pipeline
# ============================================================

if __name__ == "__main__":
    import os
    from baselines import (
        momentum_alpha, reversal_alpha, mean_reversion_alpha,
        low_volatility_alpha, trend_alpha, combined_alpha
    )
    
    PROC_DIR = r"C:\Users\EV-Car\Main_Project_2\data\processed"
    
    # Load data
    print("Loading data...")
    with open(os.path.join(PROC_DIR, 'panel.pkl'), 'rb') as f:
        panel = pickle.load(f)
    
    index_df = pd.read_csv(os.path.join(PROC_DIR, 'NIFTY50_INDEX.csv'))
    print(f"Panel: {len(list(panel.values())[0])} dates × {len(list(panel.values())[0].columns)} stocks")
    
    # ============================================
    # TRAIN/TEST SPLIT
    # ============================================
    # Train: 2002-2018 | Test: 2019-2025
    # This gives ~16 years training, ~6 years out-of-sample testing
    train_end = '2018-12-31'
    
    panel_dates = list(panel.values())[0].index
    train_dates = panel_dates[panel_dates <= pd.to_datetime(train_end)]
    test_dates = panel_dates[panel_dates > pd.to_datetime(train_end)]
    
    print(f"Train: {train_dates[0].date()} to {train_dates[-1].date()} ({len(train_dates)} days)")
    print(f"Test:  {test_dates[0].date()} to {test_dates[-1].date()} ({len(test_dates)} days)")
    
    # ============================================
    # FIT THE SYSTEM
    # ============================================
    system = RegimeAwareAlphaDiscovery(
        panel=panel,
        index_df=index_df,
        n_regimes=3,
        gp_params={
            'population_size': 500,
            'tournament_size': 5,
            'max_depth': 6,
            'cx_prob': 0.7,
            'mut_prob': 0.2,
            'parsimony_weight': 0.001,
        },
        top_k=3,
        combination='soft',
        random_state=42
    )
    
    system.fit(
        target='fwd_ret_20d',
        n_gen=50,
        train_end=train_end,
        verbose=True
    )
    
    # Print discovered formulas
    system.print_discovered_formulas()
    
    # ============================================
    # EVALUATE ON TEST SET
    # ============================================
    evaluator = AlphaEvaluator(panel, transaction_cost=0.001, n_quantiles=5)
    
    print("\n\n" + "=" * 65)
    print("  OUT-OF-SAMPLE EVALUATION (2019-2025)")
    print("=" * 65)
    
    # Our method + ablations
    results = system.full_evaluation(
        evaluator, target='fwd_ret_20d',
        holding_period=20, test_dates=test_dates, verbose=True
    )
    
    # Add baselines
    baseline_funcs = {
        'Momentum (12-1M)': momentum_alpha,
        'Mean Reversion': mean_reversion_alpha,
        'Low Volatility': low_volatility_alpha,
        'Trend (200-DMA)': trend_alpha,
        'Combined Baseline': combined_alpha,
    }
    
    for name, func in baseline_funcs.items():
        print(f"\n  Baseline: {name}")
        alpha = func(panel)
        # Restrict to test dates
        alpha_test = alpha.reindex(test_dates)
        results[name] = evaluator.evaluate(
            alpha_test, target='fwd_ret_20d', holding_period=20, verbose=False
        )
    
    # ============================================
    # COMPARISON TABLE
    # ============================================
    print("\n\n" + "=" * 80)
    print("  MAIN COMPARISON TABLE (Out-of-Sample)")
    print("=" * 80)
    comparison = compare_alphas(results)
    print(comparison.to_string())
    
    # Save results
    comparison.to_csv(os.path.join(PROC_DIR, '..', 'main_results.csv'))
    system.save(os.path.join(PROC_DIR, '..', 'regime_gp_system.pkl'))
    
    print("\n\nResults saved to Main_Project_2/data/main_results.csv")
    print("System saved to Main_Project_2/data/regime_gp_system.pkl")
