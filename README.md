# Book Sales Forecasting

Two trainers for book sales forecasting by `INVENTORY_ITEM_ID`.

## Trainers

### LightGBM (`train_lgbm.py`) — recommended
- Predicts cumulative forward sales directly on `dataset_large` (~32.3M txns, 357K items, 7 years)
- Multi-GPU CUDA-accelerated (4× RTX A4000, 16GB each)
- Best results (3yr panel, 30M rows, temporal sampling): **15d WAPE=0.652, 30d WAPE=0.604** vs baselines 0.792 / 0.773

### HGBR (`train_sales_model.py`) — legacy
- Rolling 28d mean + HistGradientBoostingRegressor residual blend on `dataset/` (~1.64M rows)

---

## Quick Start

```bash
# Full run — CUDA, 3yr panel, multi-GPU (recommended)
conda run -n book_predict python -u train_lgbm.py --txn-path ./dataset_large/TMPNXJ202603271.csv --meta-path ./dataset_large/TMPNXJ202603272.csv --output-dir artifacts_lgbm_3yr_cuda --horizons 15 30 --min-history-days 10 --panel-days 1095 --max-train-rows 30000000 --device cuda --gpu-safe --n-jobs -1 --build-workers 16 --random-state 42

# Smoke test (fast, ~2 min)
conda run -n book_predict python -u train_lgbm.py --txn-path ./dataset_large/TMPNXJ202603271.csv --meta-path ./dataset_large/TMPNXJ202603272.csv --output-dir artifacts_lgbm_smoke --max-items 2000 --panel-days 400 --device cuda --gpu-safe --n-jobs -1
```

Each run creates a timestamped subfolder: `<output-dir>/YYYYMMDD_HHMMSS/`

---

## Datasets

### `dataset_large/` (primary)
- `TMPNXJ202603271.csv` — transactions (~32.3M rows, 2019–2026): `MDHM, XSRQ, INVENTORY_ITEM_ID, XSPC, QTY, XSJE`
- `TMPNXJ202603272.csv` — item metadata (~1.16M items): `INVENTORY_ITEM_ID, LIST_PRICE_PER_UNIT, ITEM_CATEORY_CODE, ITEM_CATEORY, UN_NUMBER, BPDNAME, DLNUM, DLNAME`

### `dataset/` (legacy)
- Single CSV `TMPNXJ20260322.csv` (~1.64M rows, 203K items, 2024–2026)

---

## Features

| Group | Features |
|-------|---------|
| Categoricals (int16) | ITEM_CATEORY_CODE, ITEM_CATEORY, BPDNAME, DLNAME, UN_NUMBER, primary_store, primary_channel |
| Lags | lag_1, lag_7, lag_14, lag_28, lag_91, lag_182, lag_365 |
| Rolling sums | roll_sum_7, roll_sum_14, roll_sum_28, roll_sum_91 |
| Rolling means | roll_mean_7, roll_mean_14, roll_mean_28, roll_mean_91, roll_mean_28_yoy |
| Ratios | velocity_ratio (7d/28d trend), yoy_ratio (fixed: 0 when <1yr history) |
| Category-level | category_roll_mean_28, item_share_of_category, category_yoy_ratio |
| Counters | nonzero_days_28, store_count_28, days_since_sale |
| Revenue | avg_revenue_per_unit_28, LIST_PRICE_PER_UNIT, DLNUM |
| Calendar | day_of_month, month, quarter, week_of_year, is_weekend, item_age_days |

**Targets:** cumulative forward QTY sums — `target_15d`, `target_30d`

---

## Pipeline

```
1. Load          load_and_aggregate_sales()
                 32M txns → item-day aggregation → join metadata

2. Build         build_featured_panel()
                 357K items × 1095 days in 20K-item chunks
                 Each chunk: cross-join → features → int16 cats → parquet → free
                 RSS: ~8–11GB during build, drops to ~0.6GB after

3. Train         train_for_horizon() × N horizons (parallel, one GPU each)
                 - Scan chunks → determine 70/15/15 time splits
                 - Temporal subsampling: keep all recent (last 365d), sample older rows
                 - Stage train/valid to on-disk CSV (avoids giant in-memory concat)
                 - Train LightGBM with CUDA→GPU→CPU fallback
                 - Stream eval chunks for metrics; write test_predictions.csv
```

---

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--txn-path` | `../book_predict/dataset_large/…` | Transactions CSV |
| `--meta-path` | `../book_predict/dataset_large/…` | Item metadata CSV |
| `--output-dir` | `artifacts_lgbm` | Base output dir (runs go into `<dir>/YYYYMMDD_HHMMSS/`) |
| `--horizons` | `15 30` | Forecast horizons in days |
| `--min-history-days` | `10` | Min unique sale dates per item |
| `--panel-days` | `730` | Days of history in dense panel |
| `--max-items` | all | Cap on items (by activity) — use for quick runs |
| `--max-train-rows` | all | Cap on training rows per horizon |
| `--recent-days` | `365` | Rows within this many days of train_end kept at full rate; older rows fill remaining budget |
| `--device` | `gpu` | LightGBM device: `cuda` (recommended), `gpu` (OpenCL), `cpu` |
| `--gpu-safe` | off | Sets `max_bin=255`, `max_cat_threshold=64`, `max_cat_codes=255` |
| `--gpu-device-id` | auto | Pin all horizons to one GPU; default auto-assigns horizon[i]→GPU[i] |
| `--n-jobs` | `-1` | LightGBM CPU threads |
| `--build-workers` | `1` | Process workers for panel build (increase carefully — RAM usage scales) |
| `--random-state` | `42` | Random seed |

---

## Multi-GPU

When `--device cuda` (or `gpu`) with multiple horizons, each horizon is automatically assigned to a different GPU and trained in parallel:
- horizon[0] (15d) → GPU 0
- horizon[1] (30d) → GPU 1

Use `--gpu-device-id N` to force a specific GPU and disable auto-assignment.

**CUDA build required** — install via conda-forge (CPU-only is the PyPI default):
```bash
conda install -n book_predict -c conda-forge lightgbm=4.6.0=cuda_py_4 -y
```

---

## Memory Budget (32GB machine, 3yr panel)

| Phase | RSS |
|-------|-----|
| Load transactions | ~6.5 GB |
| Build chunks | ~8–11 GB |
| After build | ~0.6 GB |
| Train (30M rows, temporal sampled) | ~8–10 GB |

Always use `--max-train-rows 30000000` on 32GB machines. Parquet spill in `/tmp` is cleaned up automatically.

---

## Outputs

Each run: `<output-dir>/YYYYMMDD_HHMMSS/`

```
YYYYMMDD_HHMMSS/
  train_YYYYMMDD_HHMMSS.log     — full timestamped log with RSS tracking
  training_summary.csv          — one row per horizon
  horizon_15d/
    model.lgb                   — LightGBM model
    model_meta.joblib           — feature list + cat_mappings for inference
    metrics.csv                 — valid/test MAE, RMSE, WAPE vs baseline
    test_predictions.csv
    feature_importance.csv
  horizon_30d/                  — same structure
```

---

## Results History

| Run | Panel | Rows | 15d WAPE | 30d WAPE | Notes |
|-----|-------|------|----------|----------|-------|
| artifacts_lgbm_30m_gpu_safe | 2yr | 30M | 0.682 | 0.634 | baseline run |
| artifacts_lgbm_3yr/20260406_002504 | 3yr | 30M | 0.652 | 0.604 | +feature expansion, yoy_ratio P1 fix, temporal sampling |

---

## Item Segmentation (Plan B — diagnostic)

Tags every item with one of `regular / medium / sparse / seasonal / cold` based
on the last 90 days of activity, plus a YoY autocorrelation / monthly-skew
seasonality test on items with ≥ 365 days history. Lets us answer:

- Which segment dominates the test population?
- Where is WAPE actually bad — sparse tail, or seasonal items?
- Should we add segment as a LightGBM feature for a model-level lift?

### Files
- `src/book_predict/segmentation.py` — `compute_item_segments`,
  `enrich_panel_with_segments`, `per_segment_metrics`, `SegmentationConfig`.
- `diagnose_segments.py` — CLI: scores existing model on parquet chunks and
  reports per-segment MAE / WAPE.

### Run

The diagnostic needs the on-disk parquet chunks produced during training.
Re-run training once with `KEEP_CHUNKS=1` so they're preserved into
`<output-dir>/chunks/`:

```bash
# 1. Train and keep the chunk parquet files
KEEP_CHUNKS=1 conda run -n book_predict python -u train_lgbm.py \
    --device cuda --gpu-safe \
    --max-train-rows 30000000 \
    --output-dir artifacts_lgbm

# 2. Run per-segment diagnostic for the 30d model
conda run -n book_predict python -u diagnose_segments.py \
    --chunks-dir artifacts_lgbm/chunks \
    --model artifacts_lgbm/horizon_30d/model.lgb \
    --horizon 30 \
    --output artifacts_lgbm/segment_report_30d.csv

# 3. Same for 15d
conda run -n book_predict python -u diagnose_segments.py \
    --chunks-dir artifacts_lgbm/chunks \
    --model artifacts_lgbm/horizon_15d/model.lgb \
    --horizon 15 \
    --output artifacts_lgbm/segment_report_15d.csv
```

Outputs:
- `artifacts_lgbm/item_segments.csv` — one row per item with segment label and stats.
- `artifacts_lgbm/segment_report_<h>d.csv` — per-segment MAE / WAPE / n_rows.

### Decision rules (`SegmentationConfig` defaults)

| Segment | Rule |
|---------|------|
| `cold` | history < 90 days OR nonzero days in last 90 < 5 |
| `seasonal` | history ≥ 365 days AND (corr(QTY_t, QTY_{t-365}) ≥ 0.30 OR max(monthly_QTY)/mean(monthly_QTY) ≥ 2.0) |
| `regular` | nonzero days in last 90 ≥ 60 |
| `medium` | 30 ≤ nonzero days in last 90 < 60 |
| `sparse` | 5 ≤ nonzero days in last 90 < 30 |

`seasonal` overrides the frequency tier; `cold` overrides everything.

### Next step (only if diagnostic shows a clear lift opportunity)

Wire `enrich_panel_with_segments(...)` into `_build_chunk_features` and add
`SEGMENT_FEATURES` to `ALL_FEATURES` so a single LightGBM model can split on
segment. Re-train and compare lift against the current baseline.

---

## Segmented Trainer (`train_segmented.py`) — Plan B

Trains one LightGBM booster per segment (regular / medium / seasonal / sparse /
cold) with per-segment objectives + hyperparameters. Sparse + cold use
`tweedie` (zero-inflated). Routing meta + per-segment models are saved
together for inference.

```bash
# Reuses an existing chunks dir (from --build-only or KEEP_CHUNKS=1)
conda run -n book_predict python -u train_segmented.py \
    --chunks-dir artifacts_lgbm/<run>/chunks \
    --horizon 30 \
    --output-dir artifacts_lgbm_segmented/horizon_30d \
    --device gpu --gpu-safe --io-workers 4
```

Per-segment params live at the top of `train_segmented.py`. Current run
(30d) yields **OVERALL WAPE ~0.60**, with `cold` (WAPE ~1.6) the dominant
remaining drag.

---

## Build-only mode

Skip training and only persist the panel chunks. Useful when iterating on
feature engineering or training on a different machine.

```bash
conda run -n book_predict python -u train_lgbm.py \
    --device gpu --gpu-safe \
    --output-dir artifacts_lgbm/run_x \
    --build-only
# → chunks land under artifacts_lgbm/run_x/<ts>/chunks/
```

---

## Panel Enrichment (`enrich_panel.py`)

Non-destructive post-processor: reads existing parquet chunks from
`--in-dir` and writes augmented chunks to `--out-dir`. Adds:

| Column | Type | Meaning |
|--------|------|---------|
| `holiday_type` | int16 | 0=none, 1=spring, 2=qingming, 3=labor, 4=dragon, 5=mid-autumn, 6=national, 7=new-year |
| `days_to_holiday` | float16 | Days until next holiday, capped at 30 |
| `days_since_holiday` | float16 | Days since last holiday, capped at 30 |
| `discount_ratio_28` | float16 | `avg_revenue_per_unit_28 / LIST_PRICE_PER_UNIT`, clipped to [0, 2] |
| `store_demand_lag1_mean` | float32 | Mean `lag_1` across items sharing the same `primary_store` on the same date |

Mainland China holidays hardcoded for 2022–2027.

```bash
conda run -n book_predict python -u enrich_panel.py \
    --in-dir  artifacts_lgbm/run_x/<ts>/chunks \
    --out-dir artifacts_lgbm_enriched/horizon_30d/chunks \
    --io-workers 4

# Then train against the enriched chunks
conda run -n book_predict python -u train_segmented.py \
    --chunks-dir artifacts_lgbm_enriched/horizon_30d/chunks \
    --horizon 30 \
    --output-dir artifacts_lgbm_segmented_v2/horizon_30d
```

Skip the cross-item store aggregation with `--skip-store-demand` if RAM is
tight. Phase 2 (stockout-corrected rolling means) is not yet implemented.
