"""
Phase 3A: Market Regime Detection

Detects market regimes (Bull / Bear / Sideways) from the NIFTY-50 index
using a Gaussian Hidden Markov Model (HMM).

The detected regimes are used to condition the GP alpha mining:
- Different GP populations evolve different formulas per regime
- The dynamic combination layer uses regime-dependent weights

This is a key component of the paper's novelty: regime-aware alpha discovery.

Usage:
    from regime_detector import RegimeDetector
    detector = RegimeDetector(n_regimes=3)
    detector.fit(index_data)
    labels = detector.predict(index_data)
"""

import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

PROC_DIR = r"C:\Users\EV-Car\Main_Project_2\data\processed"


class RegimeDetector:
    """
    Hidden Markov Model-based market regime detector.
    
    Parameters
    ----------
    n_regimes : int
        Number of hidden states (regimes). Default 3.
    features : list
        Index features to use for regime detection.
    lookback : int
        Lookback for feature computation (only used if raw OHLC provided).
    n_iter : int
        Number of EM iterations for HMM fitting.
    random_state : int
        Random seed for reproducibility.
    """
    
    def __init__(self, n_regimes=3, n_iter=200, random_state=42):
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.regime_stats = None
        self.regime_order = None  # Maps HMM state → semantic label
    
    def _compute_index_features(self, index_df):
        """
        Compute features from NIFTY-50 index OHLC for regime detection.
        
        Features chosen to capture:
        - Trend direction (returns at multiple scales)
        - Volatility level (realized vol)
        - Drawdown depth
        - Momentum strength
        
        Parameters
        ----------
        index_df : DataFrame
            Must have columns: Date, Open, High, Low, Close
        
        Returns
        -------
        DataFrame with regime features, NaN rows dropped.
        """
        df = index_df.copy()
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
        df = df.sort_index()
        
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        features = pd.DataFrame(index=df.index)
        
        # 1. Returns at multiple horizons
        features['ret_5d'] = close.pct_change(5)
        features['ret_20d'] = close.pct_change(20)
        features['ret_60d'] = close.pct_change(60)
        
        # 2. Realized volatility (20-day)
        daily_ret = close.pct_change()
        features['vol_20d'] = daily_ret.rolling(20).std() * np.sqrt(252)
        
        # 3. Drawdown from 60-day high
        rolling_max_60 = close.rolling(60).max()
        features['drawdown_60d'] = (close - rolling_max_60) / rolling_max_60
        
        # 4. Trend strength: price relative to 50-day and 200-day SMA
        sma_50 = close.rolling(50).mean()
        sma_200 = close.rolling(200).mean()
        features['price_to_sma50'] = (close - sma_50) / sma_50
        features['sma50_to_sma200'] = (sma_50 - sma_200) / sma_200  # Golden/death cross
        
        # 5. Average True Range (normalized) — captures volatility regime
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': (high - close.shift(1)).abs(),
            'lc': (low - close.shift(1)).abs()
        }).max(axis=1)
        atr_14 = tr.rolling(14).mean()
        features['atr_norm'] = atr_14 / close  # Normalized by price level
        
        # Drop NaN rows (need 200 days of history)
        features = features.dropna()
        
        self.feature_names = list(features.columns)
        return features
    
    def fit(self, index_df):
        """
        Fit HMM on NIFTY-50 index data.
        
        Parameters
        ----------
        index_df : DataFrame
            NIFTY-50 index OHLC data.
        
        Returns
        -------
        self
        """
        features = self._compute_index_features(index_df)
        
        # Standardize features for HMM (important for numerical stability)
        self.feature_means = features.mean()
        self.feature_stds = features.std()
        X = (features - self.feature_means) / self.feature_stds
        
        # Fit Gaussian HMM
        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type='full',
            n_iter=self.n_iter,
            random_state=self.random_state,
            verbose=False
        )
        self.model.fit(X.values)
        
        # Predict regimes
        raw_labels = self.model.predict(X.values)
        
        # Order regimes semantically: sort by mean return (Bull=highest, Bear=lowest)
        regime_returns = {}
        for r in range(self.n_regimes):
            mask = raw_labels == r
            regime_returns[r] = features.loc[features.index[mask], 'ret_20d'].mean()
        
        # Sort regimes: 0=Bear (lowest return), 1=Sideways, 2=Bull (highest return)
        sorted_regimes = sorted(regime_returns.keys(), key=lambda r: regime_returns[r])
        self.regime_order = {old: new for new, old in enumerate(sorted_regimes)}
        
        # Compute regime statistics
        ordered_labels = np.array([self.regime_order[r] for r in raw_labels])
        self.regime_stats = self._compute_regime_stats(features, ordered_labels)
        
        # Store training data dates for reference
        self.train_dates = features.index
        self.train_labels = pd.Series(ordered_labels, index=features.index, name='regime')
        
        return self
    
    def predict(self, index_df):
        """
        Predict regime labels for given index data.
        
        Parameters
        ----------
        index_df : DataFrame
            NIFTY-50 index OHLC data.
        
        Returns
        -------
        Series : regime labels indexed by date.
        """
        features = self._compute_index_features(index_df)
        X = (features - self.feature_means) / self.feature_stds
        
        raw_labels = self.model.predict(X.values)
        ordered_labels = np.array([self.regime_order[r] for r in raw_labels])
        
        return pd.Series(ordered_labels, index=features.index, name='regime')
    
    def predict_proba(self, index_df):
        """
        Predict regime probabilities for given index data.
        
        Returns
        -------
        DataFrame : regime probabilities (dates × regimes).
        """
        features = self._compute_index_features(index_df)
        X = (features - self.feature_means) / self.feature_stds
        
        raw_proba = self.model.predict_proba(X.values)
        
        # Reorder columns to match semantic ordering
        inv_order = {v: k for k, v in self.regime_order.items()}
        reordered = np.zeros_like(raw_proba)
        for new_idx in range(self.n_regimes):
            old_idx = inv_order[new_idx]
            reordered[:, new_idx] = raw_proba[:, old_idx]
        
        regime_names = self._get_regime_names()
        return pd.DataFrame(reordered, index=features.index, columns=regime_names)
    
    def _get_regime_names(self):
        if self.n_regimes == 2:
            return ['Bear', 'Bull']
        elif self.n_regimes == 3:
            return ['Bear', 'Sideways', 'Bull']
        else:
            return [f'Regime_{i}' for i in range(self.n_regimes)]
    
    def _compute_regime_stats(self, features, labels):
        """Compute summary statistics for each regime."""
        stats = {}
        regime_names = self._get_regime_names()
        
        for r in range(self.n_regimes):
            mask = labels == r
            regime_feat = features.loc[features.index[mask]]
            stats[regime_names[r]] = {
                'count': mask.sum(),
                'pct': mask.sum() / len(labels) * 100,
                'avg_ret_20d': regime_feat['ret_20d'].mean(),
                'avg_vol_20d': regime_feat['vol_20d'].mean(),
                'avg_drawdown': regime_feat['drawdown_60d'].mean(),
                'avg_trend': regime_feat['price_to_sma50'].mean(),
            }
        return stats
    
    def print_summary(self):
        """Print regime detection summary."""
        print("=" * 60)
        print("  REGIME DETECTION SUMMARY")
        print("=" * 60)
        print(f"  Number of regimes: {self.n_regimes}")
        print(f"  Training period: {self.train_dates[0].date()} to {self.train_dates[-1].date()}")
        print(f"  Total trading days: {len(self.train_dates)}")
        print(f"  HMM converged: {self.model.monitor_.converged}")
        print(f"  Log-likelihood: {self.model.score(np.zeros((1, len(self.feature_names)))):.0f}")
        print()
        
        regime_names = self._get_regime_names()
        
        print(f"  {'Regime':<12} {'Days':>6} {'Pct':>7} {'Avg Ret':>10} {'Avg Vol':>10} {'Avg DD':>10} {'Avg Trend':>10}")
        print("  " + "-" * 57)
        
        for name in regime_names:
            s = self.regime_stats[name]
            print(f"  {name:<12} {s['count']:>6} {s['pct']:>6.1f}% "
                  f"{s['avg_ret_20d']*100:>9.2f}% {s['avg_vol_20d']*100:>9.1f}% "
                  f"{s['avg_drawdown']*100:>9.2f}% {s['avg_trend']*100:>9.2f}%")
        
        print()
        
        # Transition matrix
        trans = self.model.transmat_
        # Reorder transition matrix
        inv_order = {v: k for k, v in self.regime_order.items()}
        reordered_trans = np.zeros_like(trans)
        for i in range(self.n_regimes):
            for j in range(self.n_regimes):
                reordered_trans[i][j] = trans[inv_order[i]][inv_order[j]]
        
        print("  Transition Matrix:")
        print("  {:<12}".format("From / To"), end="")
        for name in regime_names:
            print(f" {name:>10}", end="")
        print()
        
        for i, name in enumerate(regime_names):
            print(f"  {name:<12}", end="")
            for j in range(self.n_regimes):
                print(f" {reordered_trans[i][j]:>10.3f}", end="")
            print()
        
        print("=" * 60)
    
    def get_regime_dates(self, regime_label=None):
        """
        Get date ranges for each regime.
        
        Parameters
        ----------
        regime_label : int or None
            If specified, return dates for that regime only.
        
        Returns
        -------
        dict or DatetimeIndex
        """
        if regime_label is not None:
            return self.train_labels[self.train_labels == regime_label].index
        
        regime_names = self._get_regime_names()
        return {
            name: self.train_labels[self.train_labels == i].index
            for i, name in enumerate(regime_names)
        }
    
    def save(self, path):
        """Save detector to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Regime detector saved to {path}")
    
    @staticmethod
    def load(path):
        """Load detector from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)


def select_n_regimes(index_df, candidates=[2, 3, 4], random_state=42):
    """
    Select optimal number of regimes using BIC.
    
    Lower BIC = better model (penalizes complexity).
    """
    print("Selecting optimal number of regimes...")
    print(f"  Candidates: {candidates}")
    
    results = {}
    for n in candidates:
        detector = RegimeDetector(n_regimes=n, random_state=random_state)
        features = detector._compute_index_features(index_df)
        X = (features - features.mean()) / features.std()
        
        model = GaussianHMM(
            n_components=n,
            covariance_type='full',
            n_iter=200,
            random_state=random_state
        )
        model.fit(X.values)
        
        bic = -2 * model.score(X.values) * len(X) + n * (n - 1 + 2 * X.shape[1] + X.shape[1] * (X.shape[1] + 1) / 2) * np.log(len(X))
        aic = -2 * model.score(X.values) * len(X) + 2 * n * (n - 1 + 2 * X.shape[1] + X.shape[1] * (X.shape[1] + 1) / 2)
        
        results[n] = {'BIC': bic, 'AIC': aic, 'LL': model.score(X.values) * len(X)}
        print(f"  n={n}: BIC={bic:.0f}, AIC={aic:.0f}, LL={results[n]['LL']:.0f}")
    
    best_n = min(results, key=lambda n: results[n]['BIC'])
    print(f"\n  Best by BIC: n_regimes = {best_n}")
    return best_n, results


if __name__ == "__main__":
    # Load NIFTY-50 index data
    index_path = os.path.join(PROC_DIR, 'NIFTY50_INDEX.csv')
    index_df = pd.read_csv(index_path)
    print(f"Index data: {len(index_df)} rows, {index_df.columns.tolist()}")
    
    # Select optimal number of regimes
    best_n, selection_results = select_n_regimes(index_df)
    
    # Fit with best number of regimes
    print(f"\nFitting regime detector with n_regimes={best_n}...")
    detector = RegimeDetector(n_regimes=best_n)
    detector.fit(index_df)
    detector.print_summary()
    
    # Also try n=3 if best was different (for comparison)
    if best_n != 3:
        print("\n\nAlso fitting with n_regimes=3 for comparison...")
        detector3 = RegimeDetector(n_regimes=3)
        detector3.fit(index_df)
        detector3.print_summary()
    
    # Save the detector
    save_path = os.path.join(PROC_DIR, '..', 'regime_detector.pkl')
    detector.save(save_path)
    
    # Show regime distribution over time
    labels = detector.train_labels
    regime_names = detector._get_regime_names()
    
    print("\nRegime Distribution by Decade:")
    for decade_start in [2002, 2007, 2012, 2017, 2022]:
        decade_end = decade_start + 5
        mask = (labels.index.year >= decade_start) & (labels.index.year < decade_end)
        sub = labels[mask]
        if len(sub) == 0:
            continue
        print(f"\n  {decade_start}-{decade_end}:")
        for i, name in enumerate(regime_names):
            pct = (sub == i).sum() / len(sub) * 100
            print(f"    {name}: {pct:.1f}%")
