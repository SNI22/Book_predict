# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Conda environment: `book_predict`. Always prefix Python commands with `conda run -n book_predict`.

Key packages: LightGBM 4.6 (CUDA build via conda-forge), pandas, numpy, pyarrow, joblib, tqdm.

Dataset paths default to `../book_predict/dataset_large/`. Override with `--txn-path` and `--meta-path`.

## Top-level scripts and what they do

| Script | Purpose |
|--------|---------|
| `train_lgbm.py` | Thin CLI wrapper. Parses args, sets up logging, calls `run_training(config)` in `src/book_predict/trainer_lgbm.py`. Trains one global LightGBM per horizon. |
| `train_segmented.py` | "Plan B" trainer: one LightGBM per item segment (regular/medium/seasonal/sparse/cold). Reads existing parquet chunks; does NOT re-build the panel. Per-segment hyperparameters live at the top of this file. |
| `enrich_panel.py` | Non-destructive post-processor: reads chunks from `--in-dir`, writes augmented chunks to `--out-dir`. Adds Chinese-holiday features, `discount_ratio_28`, `store_demand_lag1_mean`. Phase-1 feature engineering. |
| `diagnose_segments.py` | Loads chunks + an existing trained model, reports per-segment MAE/WAPE. Does not retrain. |
| `train_sales_model.py` | Legacy HGBR residual-blend trainer on the small `dataset/` CSV. Not the focus. |

## Three workflows

```bash
# A. Standard training (single global model per horizon)
conda run -n book_predict python -u train_lgbm.py \
    --device cuda --gpu-safe --max-train-rows 30000000 \
    --output-dir artifacts_lgbm

# B. Build-once, iterate on features and segmented training
# 1) Build chunks only (skip training)
conda run -n book_predict python -u train_lgbm.py \
    --device gpu --gpu-safe --output-dir artifacts_lgbm/run_x --build-only
# 2) (optional) Enrich chunks with new features
conda run -n book_predict python -u enrich_panel.py \
    --in-dir  artifacts_lgbm/run_x/<ts>/chunks \
    --out-dir artifacts_lgbm_enriched/horizon_30d/chunks --io-workers 4
# 3) Train segmented against original or enriched chunks
conda run -n book_predict python -u train_segmented.py \
    --chunks-dir artifacts_lgbm_enriched/horizon_30d/chunks \
    --horizon 30 --output-dir artifacts_lgbm_segmented/horizon_30d \
    --device gpu --gpu-safe --io-workers 4

# C. Smoke test (fast, ~2 min)
conda run -n book_predict python -u train_lgbm.py \
    --max-items 5000 --panel-days 400 --output-dir artifacts_lgbm_smoke
```

`KEEP_CHUNKS=1` env var on a normal `train_lgbm.py` run also retains chunks (under `<output-dir>/chunks/`) instead of `--build-only`'s skip-training behavior.

## Architecture

All `train_lgbm.py` logic lives in `src/book_predict/trainer_lgbm.py`. Pipeline:

**1. Load** (`load_and_aggregate_sales`) — reads both CSVs, aggregates ~32M transactions to item-day level (QTY sum, XSJE sum, store_count nunique), computes per-item `primary_store` / `primary_channel` (mode by frequency via vectorized `groupby+size+drop_duplicates`, NOT `lambda mode()` — that was hours-slow), joins metadata.

**2. Build** (`build_featured_panel`) — the panel (357K items × 730–1095 days ≈ 265–397M rows) never fits in RAM. Built in 20K-item chunks: cross-join → feature engineering → dtype quantization (int16 cats, float16 ratios, float32 sums) → parquet to `/tmp` → free. Categorical strings are encoded to int16 codes here; mapping saved to `cat_mappings.joblib`. The raw sales DataFrame is deleted after all chunks are written (RSS drops from ~11GB to ~0.6GB).

**3. Train per horizon** (`train_for_horizon`) — for each horizon:
- Scan parquet chunks → determine 70/15/15 time-based splits
- Stream rows into on-disk staging CSVs (avoids the giant in-memory concat that caused OOM)
- Temporal subsampling: rows within `--recent-days` of train_end are kept at full rate; older rows fill the remaining `--max-train-rows` budget
- Train LightGBM with CUDA→OpenCL→CPU fallback
- Stream eval chunks for metrics; write `test_predictions.csv` incrementally

**Multi-GPU**: when `--device cuda` and multiple horizons, each horizon auto-assigns to a different GPU and trains in parallel. Override with `--gpu-device-id N`.

### Segmented trainer (`train_segmented.py`)

Reads parquet chunks (does NOT rebuild the panel), labels each item via `compute_item_segments` from `src/book_predict/segmentation.py`, then trains one booster per segment. Each segment streams chunks once, filters to its items, and splits by date. Inference routes each test row to its segment's booster; metrics are aggregated globally + per segment.

Per-segment hyperparameters and `num_boost_round` caps are top-of-file constants (`BASE_PARAMS`, `SEGMENT_PARAMS`, `SEGMENT_NUM_BOOST`). Sparse/cold use `objective=tweedie` (zero-inflated). Cold has heavier `lambda_l2`.

Saves: one `model_<segment>.lgb` per segment, `segment_metrics.csv`, `test_predictions_all.csv`, `routing_meta.joblib`, `item_segments.csv`.

### Segmentation (`src/book_predict/segmentation.py`)

`SegmentationConfig` defaults — applied to last 90 days of activity:
- `cold`: history < 90 days OR nonzero days < 5
- `seasonal`: history ≥ 365 days AND (yoy autocorr ≥ 0.30 OR max-monthly / mean-monthly ≥ 2.0). Overrides frequency tier.
- `regular`: nonzero days ≥ 60
- `medium`: 30 ≤ nonzero days < 60
- `sparse`: 5 ≤ nonzero days < 30

### Enrichment (`enrich_panel.py`)

Adds new columns to existing chunks without modifying the trainer:
- `holiday_type` int16 (0=none, 1–7 = spring/qingming/labor/dragon/mid-autumn/national/new-year). Mainland China dates 2022–2027 hardcoded.
- `days_to_holiday`, `days_since_holiday` float16, capped at 30
- `discount_ratio_28` float16 = `avg_revenue_per_unit_28 / LIST_PRICE_PER_UNIT`, clipped [0, 2]
- `store_demand_lag1_mean` float32 = mean `lag_1` across items sharing `primary_store` on the same date

`train_segmented.py`'s `_discover_features` auto-picks them up: int16 small-range → categorical, others → numeric. No trainer changes needed when enriching.

## Key design decisions

- **Targets**: cumulative forward QTY sums (`target_Nd = sum of next N days of sales`).
- **Baseline**: `roll_mean_28 * horizon`. Reported alongside model WAPE for lift comparison.
- **Categorical encoding**: string → int16 codes built once from the full item set, saved to `cat_mappings.joblib`. LightGBM receives column indices via `categorical_feature=`. `-1` is the "unknown/missing" code; LightGBM logs a warning that converts it to NaN — this is intended (silenced by `_SilentLgbLogger` in `train_segmented.py`).
- **GPU-safe mode** (`--gpu-safe`): sets `max_bin=255`, `max_cat_threshold=64`, `max_cat_codes=255` to avoid LightGBM GPU bin-size failures on high-cardinality categoricals (BPDNAME ~3K, DLNAME ~20K levels).
- **Subsampling** (`--max-train-rows`): random row-level sampling during staging, with `--recent-days` carve-out. Breaks temporal density per item — known limitation.

## Feature groups

`ALL_FEATURES` = `CATEGORICAL_FEATURES` + `NUMERIC_FEATURES` (defined in `trainer_lgbm.py`):

| Group | Features |
|-------|---------|
| Categoricals (int16) | ITEM_CATEORY_CODE, ITEM_CATEORY, BPDNAME, DLNAME, UN_NUMBER, primary_store, primary_channel |
| Lags | lag_1, lag_7, lag_14, lag_28, lag_91, lag_182, lag_365 |
| Rolling sums | roll_sum_7, roll_sum_14, roll_sum_28, roll_sum_91 |
| Rolling means | roll_mean_7, roll_mean_14, roll_mean_28, roll_mean_91, roll_mean_28_yoy |
| Ratios | velocity_ratio (7d/28d trend), yoy_ratio |
| Category-level | category_roll_mean_28, item_share_of_category, category_yoy_ratio |
| Counters | nonzero_days_28, store_count_28, days_since_sale |
| Revenue | avg_revenue_per_unit_28, LIST_PRICE_PER_UNIT, DLNUM |
| Calendar | day_of_week, day_of_month, month, week_of_year, item_age_days |

## Memory budget (32 GB machine)

| Phase | RSS |
|-------|-----|
| Load transactions | ~6.5 GB |
| Build chunks (730d) | ~8–11 GB |
| Build chunks (1095d / 3yr) | ~12–16 GB |
| After build (parquet on disk) | ~0.6 GB |
| Read-back for training (`train_lgbm.py`) | ~17–19 GB |
| Per-segment training (sequential) | ~6–8 GB peak |

For full `train_lgbm.py` on 32 GB use `--max-train-rows 30000000`. For `train_segmented.py`, keep `--parallel-segments 1` on 32 GB — concurrent segments stack RSS. `/tmp` parquet spill is cleaned up automatically after training (set `KEEP_CHUNKS=1` or use `--build-only` to retain).

## Outputs

Per `train_lgbm.py` run: `<output-dir>/<timestamp>/horizon_<H>d/{model.lgb, model_meta.joblib, metrics.csv, test_predictions.csv, feature_importance.csv}` plus top-level `training_summary.csv` and timestamped log.

Per `train_segmented.py` run: `<output-dir>/{model_<segment>.lgb, segment_metrics.csv, test_predictions_<segment>.csv, test_predictions_all.csv, overall_metrics.json, item_segments.csv, routing_meta.joblib}`.

## Notes for future edits

- `is_month_start` and `is_month_end` were removed from features (zero importance across runs). Don't add them back.
- The pandas/LightGBM warning "negative value in categorical features, will convert it to NaN" is **expected** (`-1` codes for unknown categoricals → NaN branch in trees). Already filtered in `train_segmented.py` via `_SilentLgbLogger`.
- `cold` segment WAPE has been the dominant remaining drag (~1.4–1.6). Tweedie + heavier reg helped only marginally; the next big lever is a two-stage `P(sale>0) × E[sale|sale>0]` model.
- `train_segmented.py` does not currently train on enriched chunks any differently — it just sees more columns. If you add features that need special handling (e.g., monotone constraints on holidays), wire it through `_build_params` or a per-feature config there.
