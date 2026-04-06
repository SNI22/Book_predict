# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Conda environment: `book_predict`. Always prefix Python commands with `conda run -n book_predict`.

Key packages: LightGBM 4.6, pandas 3.0, numpy 2.4, scikit-learn 1.8, pyarrow 23, tqdm, joblib.

## Run Commands

```bash
# Full run (GPU, 30M row cap recommended for 32GB RAM)
conda run -n book_predict python -u train_lgbm.py \
  --device gpu --gpu-safe --max-train-rows 30000000 \
  --output-dir artifacts_lgbm

# Smoke test (fast, 5K items)
conda run -n book_predict python -u train_lgbm.py \
  --max-items 5000 --panel-days 400 --output-dir artifacts_lgbm_smoke

# Cluster (SLURM)
conda run -n book_predict python -u train_lgbm.py \
  --device gpu --gpu-safe --n-jobs ${SLURM_CPUS_PER_TASK:-16} \
  --max-train-rows 30000000 --output-dir artifacts_lgbm_30m_gpu_safe
```

Dataset paths default to `../book_predict/dataset_large/`. Override with `--txn-path` and `--meta-path`.

## Architecture

All training logic lives in `src/book_predict/trainer_lgbm.py`. `train_lgbm.py` is a thin CLI wrapper that parses args, sets up logging, and calls `run_training(config)`.

### Pipeline stages

**1. Load** (`load_and_aggregate_sales`) — reads both CSVs, aggregates 32M transactions to item-day level (QTY sum, XSJE sum, store_count nunique), computes per-item primary_store and primary_channel (mode by frequency), joins item metadata. Only needed metadata columns are loaded (`usecols` excludes DESCRIPTION, ISBN, UN_NUMBER).

**2. Build** (`build_featured_panel`) — the panel (357K items × 730 days = ~265M rows) never fits in RAM at once. It is built in 20K-item chunks: cross-join → feature engineering → dtype quantization → write parquet to `/tmp` → free. Categorical strings are encoded to int16 during this phase. The raw sales DataFrame is deleted after all chunks are written (RSS drops from ~11GB to ~0.6GB).

**3. Train per horizon** (`train_for_horizon`) — for each horizon (default: 15d, 30d):
- Scan parquet chunks to determine 70/15/15 time-based train/valid/test splits
- Stream rows into on-disk staging CSVs (avoids the giant in-memory concat that caused OOM)
- Train LightGBM from staged files with GPU→CPU automatic fallback
- Stream eval chunks for metrics; write `test_predictions.csv` incrementally

### Key design decisions

- **Targets** are cumulative forward QTY sums: `target_Nd = sum of next N days of sales`.
- **Baseline** is `roll_mean_28 * horizon` (28-day rolling mean extrapolated forward).
- **Categorical encoding**: string→int16 codes built once from the full item set, saved to `cat_mappings.joblib`. LightGBM receives column indices via `categorical_feature=`.
- **Subsampling** (`--max-train-rows`): random row-level sampling applied during staging. Noted limitation: breaks temporal density per item.
- **GPU-safe mode** (`--gpu-safe`): sets `max_bin=255`, `max_cat_threshold=64`, `max_cat_codes=255` to avoid LightGBM GPU bin-size failures on high-cardinality categoricals.

### Feature groups (`ALL_FEATURES` = `CATEGORICAL_FEATURES` + `NUMERIC_FEATURES`)

| Group | Features |
|-------|---------|
| Categoricals (int16) | ITEM_CATEORY_CODE, ITEM_CATEORY, BPDNAME, DLNAME, primary_store, primary_channel |
| Lags | lag_1, lag_7, lag_14, lag_28, lag_365 |
| Rolling sums/means | roll_sum/mean 7/14/28, roll_mean_28_yoy |
| Counters | nonzero_days_28, store_count_28, days_since_sale |
| Revenue | avg_revenue_per_unit_28, LIST_PRICE_PER_UNIT, DLNUM |
| Calendar | day_of_week, day_of_month, month, week_of_year, is_month_start, is_month_end, item_age_days |

`is_month_start` and `is_month_end` have shown zero feature importance across runs — candidates for removal.

### LightGBM hyperparameters (current)

`objective=regression_l1`, `num_leaves=255`, `learning_rate=0.05`, `min_child_samples=200`, `feature_fraction=0.8`, `bagging_fraction=0.8`, `bagging_freq=5`, `lambda_l1=0.1`, `lambda_l2=1.0`, `num_boost_round=1000`, `early_stopping=50`.

30d horizon previously hit `best_iter=999` — may benefit from higher `num_boost_round`.

### Outputs (`artifacts_*/`)

Per horizon: `model.lgb`, `model_meta.joblib` (features + cat_mappings), `metrics.csv`, `test_predictions.csv`, `feature_importance.csv`. Top-level: `training_summary.csv`, timestamped log.

## Memory Budget (32GB machine)

| Phase | RSS |
|-------|-----|
| Load transactions | ~6.5 GB |
| Build chunks | ~8–11 GB |
| After build (parquet on disk) | ~0.6 GB |
| Read-back for training | ~17–19 GB |
| Train (30M rows) | ~8–10 GB |

Always use `--max-train-rows 30000000` on 32GB machines. The `/tmp` parquet spill is cleaned up automatically after training.
