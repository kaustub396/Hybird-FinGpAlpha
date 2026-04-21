"""
Evaluation Engine for Cross-Sectional Alpha Strategies.

This module provides the standard quantitative evaluation framework used
throughout the project. It evaluates alpha factors by:

1. Computing Information Coefficient (IC) and Rank IC
2. Building long-short portfolios from alpha scores
3. Measuring risk-adjusted returns (Sharpe, Drawdown, etc.)

All metrics follow standard quantitative finance conventions as used in
Alpha², QuantFactor, AlphaForge, and similar alpha mining papers.

Usage:
    from evaluation import AlphaEvaluator
    evaluator = AlphaEvaluator(panel)
    results = evaluator.evaluate(alpha_scores, target='fwd_ret_20d')
"""

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats


class AlphaEvaluator:
    """
    Evaluates alpha factors on a cross-sectional stock panel.
    
    Parameters
    ----------
    panel : dict
        Dictionary of {feature_name: DataFrame(dates × stocks)}.
        Must contain forward return targets (fwd_ret_5d, fwd_ret_20d, fwd_ret_60d).
    transaction_cost : float
        One-way transaction cost as fraction (default 0.001 = 10 bps).
    n_quantiles : int
        Number of portfolio quantiles (default 5 for quintiles).
    """
    
    def __init__(self, panel, transaction_cost=0.001, n_quantiles=5):
        self.panel = panel
        self.transaction_cost = transaction_cost
        self.n_quantiles = n_quantiles
        
        # Extract close prices for portfolio construction
        if 'Close' in panel:
            self.close = panel['Close']
        
    def evaluate(self, alpha_scores, target='fwd_ret_20d', holding_period=20,
                 verbose=True):
        """
        Full evaluation of an alpha factor.
        
        Parameters
        ----------
        alpha_scores : DataFrame (dates × stocks)
            The alpha signal. Higher = more bullish.
        target : str
            Forward return column to evaluate against.
        holding_period : int
            Holding period in trading days for portfolio evaluation.
        verbose : bool
            Print results.
            
        Returns
        -------
        dict : All evaluation metrics.
        """
        fwd_ret = self.panel[target]
        
        # Align dates
        common_dates = alpha_scores.index.intersection(fwd_ret.index)
        alpha = alpha_scores.loc[common_dates]
        returns = fwd_ret.loc[common_dates]
        
        # Drop dates where we don't have enough stocks
        valid_mask = alpha.notna().sum(axis=1) >= 10
        alpha = alpha.loc[valid_mask]
        returns = returns.loc[valid_mask]
        
        # 1. Information Coefficient metrics
        ic_metrics = self._compute_ic(alpha, returns)
        
        # 2. Portfolio metrics
        portfolio_metrics, portfolio_returns = self._compute_portfolio(
            alpha, returns, holding_period
        )
        
        # 3. Combine all metrics
        results = {**ic_metrics, **portfolio_metrics}
        results['portfolio_returns'] = portfolio_returns
        results['n_dates'] = len(common_dates)
        results['n_stocks_avg'] = alpha.notna().sum(axis=1).mean()
        
        if verbose:
            self._print_results(results)
        
        return results
    
    def _compute_ic(self, alpha, returns):
        """
        Compute Information Coefficient metrics.
        
        IC = Pearson correlation between alpha and forward returns (per date)
        Rank IC = Spearman rank correlation (more robust)
        ICIR = mean(IC) / std(IC) — measures consistency
        """
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
            
            # Pearson IC
            if np.std(a_vals) > 1e-10 and np.std(r_vals) > 1e-10:
                ic = np.corrcoef(a_vals, r_vals)[0, 1]
                ic_series.append(ic)
            
            # Spearman Rank IC
            rank_ic, _ = scipy_stats.spearmanr(a_vals, r_vals)
            if not np.isnan(rank_ic):
                rank_ic_series.append(rank_ic)
        
        ic_series = np.array(ic_series)
        rank_ic_series = np.array(rank_ic_series)
        
        return {
            'IC_mean': np.mean(ic_series) if len(ic_series) else 0,
            'IC_std': np.std(ic_series) if len(ic_series) else 0,
            'ICIR': (np.mean(ic_series) / np.std(ic_series)) if (len(ic_series) and np.std(ic_series) > 0) else 0,
            'Rank_IC_mean': np.mean(rank_ic_series) if len(rank_ic_series) else 0,
            'Rank_IC_std': np.std(rank_ic_series) if len(rank_ic_series) else 0,
            'Rank_ICIR': (np.mean(rank_ic_series) / np.std(rank_ic_series)) if (len(rank_ic_series) and np.std(rank_ic_series) > 0) else 0,
            'IC_positive_pct': (np.mean(ic_series > 0) * 100) if len(ic_series) else 0,
            'ic_series': ic_series,
            'rank_ic_series': rank_ic_series,
        }
    
    def _compute_portfolio(self, alpha, returns, holding_period):
        """
        Build and evaluate a long-short portfolio.
        
        Strategy:
        - Each rebalance date: rank stocks by alpha
        - Long top quantile, short bottom quantile
        - Equal weight within each leg
        - Rebalance every `holding_period` trading days
        """
        rebalance_dates = alpha.index[::holding_period]
        
        portfolio_returns = []
        prev_long = set()
        prev_short = set()
        
        for i, date in enumerate(rebalance_dates):
            if date not in returns.index:
                continue
            
            scores = alpha.loc[date].dropna()
            rets = returns.loc[date].dropna()
            common = scores.index.intersection(rets.index)
            
            if len(common) < 10:
                continue
            
            scores = scores.loc[common]
            rets = rets.loc[common]
            
            # Quantile split
            n = len(scores)
            q_size = n // self.n_quantiles
            if q_size < 2:
                continue
            
            ranked = scores.rank(ascending=True)
            long_stocks = set(ranked.nlargest(q_size).index)
            short_stocks = set(ranked.nsmallest(q_size).index)
            
            # Long-short return (equal weighted)
            long_ret = rets.loc[list(long_stocks)].mean()
            short_ret = rets.loc[list(short_stocks)].mean()
            ls_ret = long_ret - short_ret
            
            # Transaction costs
            turnover_long = len(long_stocks - prev_long) / max(len(long_stocks), 1)
            turnover_short = len(short_stocks - prev_short) / max(len(short_stocks), 1)
            avg_turnover = (turnover_long + turnover_short) / 2
            cost = avg_turnover * self.transaction_cost * 2  # Buy + sell
            
            ls_ret_net = ls_ret - cost
            
            portfolio_returns.append({
                'date': date,
                'long_ret': long_ret,
                'short_ret': short_ret,
                'ls_ret_gross': ls_ret,
                'ls_ret_net': ls_ret_net,
                'turnover': avg_turnover,
                'n_long': len(long_stocks),
                'n_short': len(short_stocks),
            })
            
            prev_long = long_stocks
            prev_short = short_stocks
        
        if not portfolio_returns:
            return self._empty_portfolio_metrics(), pd.DataFrame()
        
        port_df = pd.DataFrame(portfolio_returns).set_index('date')
        
        # Compute portfolio-level metrics
        net_returns = port_df['ls_ret_net']
        gross_returns = port_df['ls_ret_gross']
        
        # Annualization factor (based on holding period)
        periods_per_year = 252 / holding_period
        
        metrics = {
            'Ann_Return_Gross': gross_returns.mean() * periods_per_year,
            'Ann_Return_Net': net_returns.mean() * periods_per_year,
            'Sharpe_Gross': (gross_returns.mean() / gross_returns.std() * np.sqrt(periods_per_year)) if gross_returns.std() > 0 else 0,
            'Sharpe_Net': (net_returns.mean() / net_returns.std() * np.sqrt(periods_per_year)) if net_returns.std() > 0 else 0,
            'Max_Drawdown': self._max_drawdown(net_returns),
            'Win_Rate': (net_returns > 0).mean() * 100,
            'Avg_Turnover': port_df['turnover'].mean(),
            'Num_Rebalances': len(port_df),
            'Long_Avg_Ret': port_df['long_ret'].mean() * periods_per_year,
            'Short_Avg_Ret': port_df['short_ret'].mean() * periods_per_year,
        }
        
        return metrics, port_df
    
    def _max_drawdown(self, returns):
        """Compute maximum drawdown from a return series."""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()
    
    def _empty_portfolio_metrics(self):
        return {
            'Ann_Return_Gross': 0, 'Ann_Return_Net': 0,
            'Sharpe_Gross': 0, 'Sharpe_Net': 0,
            'Max_Drawdown': 0, 'Win_Rate': 0,
            'Avg_Turnover': 0, 'Num_Rebalances': 0,
            'Long_Avg_Ret': 0, 'Short_Avg_Ret': 0,
        }
    
    def _print_results(self, results):
        """Pretty-print evaluation results."""
        print("=" * 55)
        print("  ALPHA EVALUATION RESULTS")
        print("=" * 55)
        print(f"  Dates evaluated:    {results['n_dates']}")
        print(f"  Avg stocks/date:    {results['n_stocks_avg']:.1f}")
        print("-" * 55)
        print("  INFORMATION COEFFICIENT")
        print(f"    IC (mean):        {results['IC_mean']:.4f}")
        print(f"    IC (std):         {results['IC_std']:.4f}")
        print(f"    ICIR:             {results['ICIR']:.4f}")
        print(f"    Rank IC (mean):   {results['Rank_IC_mean']:.4f}")
        print(f"    Rank ICIR:        {results['Rank_ICIR']:.4f}")
        print(f"    IC > 0:           {results['IC_positive_pct']:.1f}%")
        print("-" * 55)
        print("  PORTFOLIO (Long-Short)")
        print(f"    Ann Return (net): {results['Ann_Return_Net']*100:.2f}%")
        print(f"    Sharpe (net):     {results['Sharpe_Net']:.3f}")
        print(f"    Max Drawdown:     {results['Max_Drawdown']*100:.2f}%")
        print(f"    Win Rate:         {results['Win_Rate']:.1f}%")
        print(f"    Avg Turnover:     {results['Avg_Turnover']*100:.1f}%")
        print("=" * 55)


def compare_alphas(results_dict):
    """
    Create a comparison table across multiple alpha strategies.
    
    Parameters
    ----------
    results_dict : dict
        {strategy_name: evaluation_results}
    
    Returns
    -------
    DataFrame : Comparison table.
    """
    metrics_to_show = [
        'IC_mean', 'Rank_IC_mean', 'ICIR', 'Rank_ICIR',
        'Ann_Return_Net', 'Sharpe_Net', 'Max_Drawdown', 'Win_Rate', 'Avg_Turnover'
    ]
    
    rows = []
    for name, res in results_dict.items():
        row = {'Strategy': name}
        for m in metrics_to_show:
            val = res.get(m, 0)
            if m in ['Ann_Return_Net', 'Max_Drawdown', 'Avg_Turnover']:
                row[m] = f"{val*100:.2f}%"
            elif m == 'Win_Rate':
                row[m] = f"{val:.1f}%"
            else:
                row[m] = f"{val:.4f}"
        rows.append(row)
    
    df = pd.DataFrame(rows).set_index('Strategy')
    return df
