# ARCHIVE

This folder stores legacy code that is retained for historical reproducibility but is **not part of the active operational pipeline**.

## Archived items

- `legacy_runners/run_kalshi_real_backtest.py`
  - Archived because it is an older in-sample-heavy runner superseded by the current benchmark workflow (`scripts/run_e0_e8_best_model_benchmark.py`) and current OOS-focused workflow (`run_kalshi_real_oos.py`).
  - Keeping the file in-repo preserves historical methodology while preventing accidental use as the primary path.
