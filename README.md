# Book Sales Forecasting

Two trainers for book sales forecasting by `INVENTORY_ITEM_ID`.

## Trainers

### LightGBM (`train_lgbm.py`) — recommended
- Predicts targets directly on `dataset_large` (~32.3M txns, 357K items, 7 years)
- GPU-accelerated (RTX 5070)
- Features: lags (1/7/14/28/91/182/365d), rolling sums/means (7/14/28/91d), YoY ratio, velocity ratio, category-level rolling features, item share of category, store count, revenue-derived, calendar, categoricals (UN_NUMBER, BPDNAME, DLNAME, etc.)
- Best results (3yr panel, 30M row cap): **15d WAPE=0.652, 30d WAPE=0.604** vs baselines 0.792 / 0.773

### HGBR (`train_sales_model.py`) — legacy
- Rolling 28d mean + HistGradientBoostingRegressor residual blend on `dataset/` (~1.64M rows)

## Datasets

### `dataset/` (original)
- Single CSV: `TMPNXJ20260322.csv` (~1.64M rows, 203K items, 2024–2026)
- Schema: `INVENTORY_ITEM_ID, ISBN, DESCRIPTION, LIST_PRICE_PER_UNIT, ITEM_CATEORY_CODE, ITEM_CATEORY, UN_NUMBER, BPDNAME, DLNUM, DLNAME, XSRQ, QTY`
- Includes negative `QTY` (returns)

### `dataset_large/` (extended)
- `TMPNXJ202603271.csv` — transactions (~32.3M rows, 2019–2026)
  - Schema: `MDHM, XSRQ, INVENTORY_ITEM_ID, XSPC, QTY, XSJE`
- `TMPNXJ202603272.csv` — item metadata (~1.16M items)
  - Schema: `INVENTORY_ITEM_ID, ISBN, DESCRIPTION, LIST_PRICE_PER_UNIT, ITEM_CATEORY_CODE, ITEM_CATEORY, UN_NUMBER, BPDNAME, DLNUM, DLNAME`

## Run

```bash
# LightGBM — recommended full run (3yr panel, GPU, 30M row cap)
conda run -n book_predict python -u train_lgbm.py \
  --txn-path ./dataset_large/TMPNXJ202603271.csv \
  --meta-path ./dataset_large/TMPNXJ202603272.csv \
  --output-dir artifacts_lgbm_3yr \
  --horizons 15 30 \
  --min-history-days 10 \
  --panel-days 1095 \
  --max-train-rows 30000000 \
  --device gpu --gpu-safe \
  --n-jobs -1 --build-workers 16 \
  --random-state 42

# LightGBM — smoke test
conda run -n book_predict python -u train_lgbm.py \
  --max-items 5000 --panel-days 400 --output-dir artifacts_lgbm_smoke

# LightGBM — GPU-safe mode (cluster / SLURM)
conda run -n book_predict python -u train_lgbm.py \
  --device gpu --gpu-safe --n-jobs ${SLURM_CPUS_PER_TASK:-16} \
  --max-train-rows 30000000 --output-dir artifacts_lgbm_30m_gpu_safe

# HGBR — legacy
conda run -n book_predict python train_sales_model.py
```

### CLI flags (LightGBM)

| Flag | Default | Description |
|------|---------|-------------|
| `--txn-path` | `../book_predict/dataset_large/TMPNXJ202603271.csv` | Transactions CSV |
| `--meta-path` | `../book_predict/dataset_large/TMPNXJ202603272.csv` | Item metadata CSV |
| `--output-dir` | `artifacts_lgbm` | Output directory |
| `--horizons` | `15 30` | Forecast horizons in days |
| `--min-history-days` | `10` | Min unique sale dates per item |
| `--panel-days` | `730` | Days of history in dense panel |
| `--max-items` | all | Cap on items (by activity) |
| `--max-train-rows` | all | Cap on training rows per horizon — subsamples if exceeded |
| `--device` | `gpu` | LightGBM device (`gpu` or `cpu`) |
| `--n-jobs` | `-1` | LightGBM threads (-1 = all) |
| `--max-bin` | LightGBM default | Histogram bins per numeric feature; lower values are more GPU-friendly |
| `--max-cat-threshold` | LightGBM default | Limits categorical split search complexity |
| `--max-cat-codes` | unlimited | Caps per-feature categorical codes (top-frequency kept, rest mapped to unknown) |
| `--gpu-safe` | off | Applies GPU-safe defaults: `max_bin=255`, `max_cat_threshold=64`, `max_cat_codes=255` (unless overridden) |
| `--random-state` | `42` | Random seed |

## Logging

All output is written to both console and a timestamped log file in the output directory (`train_YYYYMMDD_HHMMSS.log`). The log file flushes each line, so it captures progress even if the process is OOM-killed. RSS memory usage is logged at each chunk and training phase.

## Memory Management (LightGBM Trainer)

The full dataset (357K items x 730 days = ~265M rows) does not fit in 32GB RAM. The trainer uses a staged pipeline so the full panel is never in memory at once.

### Build phase

```
Raw sales (~6.5GB RSS)
    │
    for each chunk (20K items):
    │  1. Cross-join items × dates → dense panel
    │  2. Compute features in-place
    │  3. Drop intermediates (QTY, XSJE, store_count)
    │  4. Quantize (means→float16, calendar→int8)
    │  5. Write to /tmp as parquet
    │  6. del + gc.collect()
    │
    ▼
  del sales → RSS drops to 0.7GB
  Only parquet files on disk remain
```

Observed: RSS stays flat at ~12.7GB throughout all 18 chunks, drops to 0.7GB after freeing sales.

### Train phase

```
for each horizon:
    │  Read parquet chunks one at a time
    │  Apply dropna per chunk (remove warmup rows)
    │  Stream rows into staged train/valid CSV files
    │  Subsample if --max-train-rows set
    │
    ▼
  train_for_horizon() → LightGBM (GPU with automatic CPU fallback on GPU bin failures)
    │
    ▼
  Stream eval chunks for metrics + test predictions
```

Without `--max-train-rows`, the filtered panel (~250M rows) still exceeds 32GB. Use `--max-train-rows 30000000` to cap at ~30M rows — sufficient for LightGBM to learn well.

### Peak memory

| Phase | RSS | Notes |
|-------|-----|-------|
| Load transactions | ~6.5 GB | 32M txn rows |
| Build chunks (int16 cats) | ~8–11 GB | raw sales + 1 chunk |
| After build | ~0.6 GB | parquet on disk |
| Read-back (141M rows, int16 cats + col pruning) | ~17–19 GB | close to 32GB limit |
| Train (30M rows) | ~8–10 GB | filtered + subsampled |

### Dtype budget

| Column type | dtype | Bytes |
|------------|-------|-------|
| Small bounded counters (`nonzero_days_28`, `store_count_28`) | float16 | 2 |
| Lags, sums, revenue | float32 | 4 |
| Calendar (day_of_week, month, etc.) | int8 | 1 |
| `week_of_year`, `item_age_days` | float16 (safe-clipped before cast) | 2 |
| `DLNUM`, `days_since_sale`, rolling means | float32 | 4 |
| Categoricals (int16-encoded) | int16 | 2 |

### Categorical encoding

String categoricals (BPDNAME, DLNAME, etc.) are mapped to int16 codes during panel build to reduce memory from ~60 bytes/cell to 2 bytes. Global mappings are built once from raw data, applied per chunk, and saved to `cat_mappings.joblib` for inference. LightGBM receives `categorical_feature` as column indices and handles the int codes natively.

With `--max-cat-codes`, very high-cardinality features are truncated to top-frequency levels to reduce GPU bin pressure. Unknown/rare levels map to `-1`.

## Outputs

### LightGBM (`artifacts_lgbm/`)
- `training_summary.csv`
- `train_YYYYMMDD_HHMMSS.log` — full training log with memory stats
- `horizon_Xd/model.lgb` — LightGBM model
- `horizon_Xd/model_meta.joblib` — feature list metadata
- `horizon_Xd/metrics.csv` — valid/test MAE, RMSE, WAPE
- `horizon_Xd/test_predictions.csv`
- `horizon_Xd/feature_importance.csv`

### HGBR (`artifacts/`)
- `training_summary.csv`
- `horizon_Xd/model.joblib`
- `horizon_Xd/metrics.csv`
- `horizon_Xd/test_predictions.csv`
