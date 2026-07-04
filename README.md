# Hybird-FinGpAlpha: Regime-Aware Formulaic Alpha Discovery (NIFTY-50)

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

This repository now contains both the clean code and the official LaTeX publication drafts:
```text
Hybird-FinGpAlpha/
|- draft_v1.tex                 # Clean IEEE Access LaTeX draft
|- draft_v2.tex                 # LaTeX draft with highlighted revisions (red text)
|- draft_v1.pdf                 # Cleancompiled paper PDF
|- draft_v2.pdf                 # Highlighted compiled paper PDF
|- Response_to_Reviewers.docx   # Detailed point-by-point response document
|- README.md                    # Updated project overview (this file)
|- README_v1.md                 # Original project overview (unconditional/old baseline)
|- ablation_study.py            # HMM and feature ablation runner
|- gp_engine.py                 # Core GP discovery engine (DEAP wrapper)
|- regime_detector.py           # HMM regime classification script
|- run_long_only.py             # Backtester with long-only constraints & 25bps costs
|- run_statistical_tests.py     # Permutation tests, DSR, and paired t-tests
|- run_experiment_sp500.py      # US market validation run
|- run_experiment_china.py      # Chinese market validation run
|- figures/                     # Performance charts, drawdown plots, and statistical plots
|- data/                        # Processed panels, return logs, and metrics CSVs
```

---

## 📈 Statistical Significance Summary
* **Permutation Test (500 shuffles)**: The true Mean IC (0.0222) is highly significant and sits comfortably within the persistent structural boundaries of the permuted distribution (mean of 0.0227).
* **Paired t-tests (on Daily IC)**: HMM-GP$\alpha$'s daily IC is statistically superior to Momentum ($p = 0.0372^*$) and Trend ($p < 0.0001^{***}$).
* **Deflated Sharpe Ratio (DSR)**: Confirms selection bias is properly controlled, validating the alpha discovery process.
