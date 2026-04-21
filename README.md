# Hybird-FinGpAlpha

Hybrid financial alpha research project combining:

- GP-based formula discovery from price features
- HMM regime detection (Bull/Bear)
- AFM-derived sentiment/fundamental fusion signals
- Integrated portfolio backtesting and comparison vs classic baselines

## What this project does

1. Reuses GP pipeline components (feature + evaluator stack) for equity alpha generation.
2. Builds an **integrated panel** by merging:
   - Market feature panel (from Main_Project_2 pipeline artifacts)
   - AFM sentiment/fundamental/regime signals (`final_df.csv`, `phase3_fused_signal.csv`)
3. Trains/evaluates multiple variants:
   - Regime-GP Soft / Hard / Ensemble
   - HMM Regime-GP Alpha
   - Regime-GP + HMM experimental blend
   - AFM-only variants
   - Baselines (Momentum, Mean Reversion, Low Volatility, Trend)

## Dataset / Inputs used

- `final_df.csv` - annual AFM/fundamental fusion summary
- `phase3_fused_signal.csv` - fused signal series
- `gp/integrated_panel.pkl` - merged panel used for model/backtest
- Price/feature panel dependency loaded from:
  - `C:\Users\EV-Car\Main_Project_2\data\processed\panel.pkl` (as referenced in code)

## Main files

- `macro_micro_panel.py` - creates `gp/integrated_panel.pkl`
- `comparison/run_experiment.py` - integrated experiment runner
- `run_original_baseline.py` - original HMM regime GP baseline
- `gp/final_experiments.py` - core GP experiment flow
- `gp/integrated_main_results.csv` - main integrated results
- `gp/integrated_power_results.csv` - focused power comparison

## Results obtained

From `gp/integrated_main_results.csv`:

| Strategy | Ann Return (Net) | Sharpe (Net) | Rank IC Mean | Max Drawdown |
|---|---:|---:|---:|---:|
| Regime-GP Ensemble + HMM (Exp) | **8.41%** | **0.5518** | **0.0346** | -19.62% |
| Regime-GP Ensemble (Ours) | 7.68% | 0.5593 | 0.0386 | -24.40% |
| HMM Regime-GP Alpha | 7.49% | 0.5244 | 0.0303 | **-18.17%** |
| Mean Reversion | 4.46% | 0.2981 | 0.0221 | -20.19% |
| Momentum (12-1M) | -2.57% | -0.1369 | 0.0059 | -40.65% |

From `gp/integrated_power_results.csv`:

- Regime-GP variants are consistently ahead of vanilla GP and most classic baselines on risk-adjusted return.
- Best standalone return among listed strategies: **Regime-GP Ensemble (7.68%)**.

## How to run

1. Build integrated panel:

```bash
python macro_micro_panel.py
```

2. Run integrated comparison:

```bash
python comparison/run_experiment.py
```

3. Run original HMM baseline:

```bash
python run_original_baseline.py
```

## Notes

- This repo contains research/backtest outputs, not investment advice.
- Some scripts use absolute Windows paths; adjust paths if running on another machine.
