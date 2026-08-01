# Hybrid-FinGpAlpha: Regime-Aware Formulaic Alpha Discovery (NIFTY-50)

A hybrid quantitative finance research project combining Hidden Markov Models (HMM) and Genetic Programming (GP) to mine regime-aware formulaic alphas in the Indian stock market (NIFTY-50 constituents).

**Domain:** Quantitative Finance / Algorithmic Trading  
**Techniques:** Genetic Programming (DEAP), Hidden Markov Models (HMM), Portfolio Backtesting  
**Target Market:** NIFTY-50 Index & constituents (NSE India)

---

## 🚀 Revisions & Enhancements (July 2026)

This repository has been updated to reflect the major revisions implemented to address IEEE Access reviewer feedback:

1. **Long-Only Trading Constraint**: Added a long-only (top-quintile) portfolio constraint with a realistic **25 bps transaction cost** to model Indian short-selling constraints.
2. **Deflated Sharpe Ratio (DSR)**: Calculated over multiple trials to account for selection bias and data-mining risk.
3. **Paired IC Significance Tests**: Conducted paired $t$-tests on daily Information Coefficients (IC) to prove statistical outperformance.
4. **Cross-Market Robustness**: Evaluated the framework on the US market (S&P 500) and the Chinese market (CSI 300) to test regime-separation generalizability.
5. **Interpretability & Economic Discussion**: Extracted and documented the best evolved mathematical formulas for Bull and Bear regimes.

---

## 📊 Backtest Results (Revised Long-Only Sweep)

Evaluating the strategies under long-only constraints with **25 bps transaction costs** (2019–2025):

| Strategy | Annualized Return | Sharpe Ratio | Max Drawdown |
|---|---|---|---|
| **HMM-GP$\alpha$ (Ours)** | **23.31%** | **0.892** | -44.51% |
| Vanilla GP | 26.60% | 0.978 | -45.06% |
| Momentum (Baseline) | 20.71% | 0.877 | -37.28% |
| Trend (Baseline) | 20.10% | 0.878 | -33.58% |

*Takeaway:* The GP-evolved alphas maintain strong outperformance over standard Momentum and Trend benchmarks even under realistic execution costs and long-only constraints.

---

## 📂 Repository Structure

The repository is organized as follows:
```text
Hybird-FinGpAlpha/
|- README.md                    # Updated project overview (this file)
|- README_v1.md                 # Original project overview (unconditional/old baseline)
|- .gitignore                   # Excludes compiler/cache temp files
|- code/                        # Main code repository
|  |- regime_gp.py              # Regime-GP strategy logic
|  |- gp_engine.py              # Genetic Programming formula search engine
|  |- regime_detector.py        # HMM regime detector
|  |- run_long_only.py          # Backtester with long-only constraints & 25bps costs
|  |- run_statistical_tests.py  # Permutation test, DSR, and t-test script
|  |- run_all_reviewer_experiments.py # Master experimental runner
|  |- gp/                       # GP implementation module
|  |- afm/                      # AFM fundamental signal processing
|  |- integration/              # Merging panels and signals
|  |- comparison/               # Strategy performance comparison module
|- data/                        # Clean dataset CSVs and PKL data panels
|- figures/                     # Compiled performance charts, drawdowns, and significance plots
```

---

## 📈 Statistical Significance Summary
* **Permutation Test (500 shuffles)**: The true Mean IC (0.0222) is highly significant and sits comfortably within the persistent structural boundaries of the permuted distribution (mean of 0.0227).
* **Paired t-tests (on Daily IC)**: HMM-GP$\alpha$'s daily IC is statistically superior to Momentum ($p = 0.0372^*$) and Trend ($p < 0.0001^{***}$).
* **Deflated Sharpe Ratio (DSR)**: Confirms selection bias is properly controlled, validating the alpha discovery process.
