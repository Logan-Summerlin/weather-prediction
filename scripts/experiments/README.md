# Experiments

One-off research and diagnostic scripts, kept separate from the production
pipeline to avoid accidental coupling.

## benchmarking/

Model training, evaluation, and cross-city comparison experiments.

| Script | Description |
|---|---|
| `run_chi_advanced_benchmark.py` | Advanced NN models (FeatureAttention, MOSCorrection, RegimeConditional) for CHI |
| `run_phl_advanced_benchmark.py` | Advanced NN models for PHL |
| `run_chi_phl_unified_benchmark.py` | E-series + Unified synthesis (U0-U5) for CHI and PHL |
| `run_real_data_benchmark.py` | Benchmark against real NWS MOS forecasts for CHI/PHL |
| `run_cross_city_benchmark_comparison.py` | Cross-city Brier score comparison (NYC, CHI, PHL) |
| `run_promotion_evaluation_v2.py` | Multi-city promotion readiness evaluation (v2) |

## trading/

Backtest and trading strategy experiments.

| Script | Description |
|---|---|
| `run_real_kalshi_backtest.py` | EV-gated backtest using real Kalshi pre-settlement prices |
| `run_unified_trading_backtest.py` | Trading backtest using Unified-series contract probabilities |
| `run_trading_strategy_sweep.py` | Parameter sweep over EV threshold, Kelly fraction, max contracts |
