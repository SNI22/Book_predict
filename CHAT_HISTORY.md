# Chat History — LightGBM Direct Trainer

**Date:** 2026-03-29 → 2026-03-30

---

## Context

Original trainer (`~/Documents/book_predict/`, branch `hgbr_residual_blend`) used a 28-day rolling mean as a base + HistGradientBoostingRegressor on residuals, blended at ≤10%. Results were poor (WAPE >1.0 on 15d validation on the small `dataset/` CSV). This repo switches to LightGBM predicting cumulative forward sales directly on `dataset_large` (~32.3M transactions, 357K items, 7 years of history). GPU-accelerated on RTX 5070 (12GB VRAM).

---

## Smoke Test Results (5K items, 400 panel days)

| Horizon | WAPE (test) | Baseline WAPE (test) |
|---------|-------------|----------------------|
| 15d     | 0.6552      | 0.7360               |
| 30d     | 0.5909      | 0.6637               |

Full run not yet completed at this point — OOM during panel build → fixed with chunked approach → GPU driver crashed during retry.

---

## Early Bugs Fixed

1. **Slow aggregation** — per-group `lambda mode()` on 32M rows took hours. Replaced with vectorized item-level mode: `groupby + size + drop_duplicates` to get `primary_store` and `primary_channel`.
2. **Slow dense panel** — Python for-loop over 357K items to build cross-join was prohibitively slow. Replaced with a single cross-join + left-join + `groupby ffill/bfill`.
3. **OOM on panel build** — full cross-join (357K items × 730 days = ~260M rows) exceeded 32GB RAM all at once. Solved with chunked panel build (20K items per chunk).

---

## Memory Optimization — Iteration 1: Fused Chunk Pipeline (2026-03-30)

**Problem:** Even after chunking, chunks were appended to a Python list and then `pd.concat`'d at the end — the final concat held all 260M rows in memory simultaneously.

**Solution:** Fused panel build + feature engineering into a single function `_build_chunk_features`. Each chunk is computed and immediately written to parquet on `/tmp`, then freed via `del + gc.collect()`. The full panel is never in memory at once.

Key mechanics:
- `_build_chunk_features(chunk_items, frame, ...)` — builds cross-join, computes all features in-place, drops intermediates (`XSJE`, `store_count`, `QTY`), returns only `ALL_FEATURES + targets + keys`
- `build_featured_panel()` — iterates chunks, writes each to `tmp_dir/chunk_NNNN.parquet`, frees immediately
- `load_filtered_panel()` — reads parquet files back one at a time, applies `dropna` per horizon (removes warmup rows where lags/targets are NaN), concatenates only usable rows

**Result:** Build phase RSS stayed flat at ~12.7GB for all 18 chunks.

---

## Memory Optimization — Iteration 2: Dtype Quantization (2026-03-30)

Applied during chunk build to reduce per-row size:

| Column type | Before | After | Bytes saved |
|-------------|--------|-------|-------------|
| Rolling means, ratios, counts | float64 | float16 | −6 bytes/cell |
| Calendar (day_of_week, month, etc.) | int64 | int8 | −7 bytes/cell |
| Lags, sums, revenue | float64 | float32 | −4 bytes/cell |
| `item_age_days`, `week_of_year`, `DLNUM` | float64 | float16 | −6 bytes/cell |

**float16 overflow caveat:** `roll_sum_28` and lags can exceed 65504 (float16 max) for popular items — kept those as float32. LightGBM bins internally so float16 precision loss is negligible for means/ratios.

**Bugs hit:**
- `IntCastingNaNError` on `week_of_year`, `item_age_days`, `DLNUM` — contained NaN, couldn't cast directly to int. Fixed: use float16 (handles NaN), and `pd.to_numeric(..., errors="coerce").astype("float16")` for DLNUM.
- `float16` overflow warnings on `roll_sum_28` → kept sums/lags as float32.

---

## Memory Optimization — Iteration 3: Parquet Spill-to-Disk + Per-Horizon Cycle (2026-03-30)

Full lifecycle:

```
Load sales (~6.5GB RSS)
    │
    for each 20K-item chunk:
    │  1. Cross-join items × dates → dense panel
    │  2. Compute all features in-place
    │  3. Drop intermediates (XSJE, store_count, QTY)
    │  4. Quantize dtypes
    │  5. Write to /tmp as parquet
    │  6. del chunk_panel + gc.collect()
    │
    ▼
  del sales → RSS drops to ~0.7GB
  Only parquet files remain on disk

  for each horizon:
    │  Read parquet chunks one at a time
    │  dropna per chunk (remove warmup rows)
    │  pd.concat filtered rows
    │
    ▼
  train LightGBM on GPU
    │
    ▼
  del panel + gc.collect() → freed before next horizon
    │
    ▼
  shutil.rmtree(tmp_dir)  ← parquet files cleaned up
```

**Logging added:** Timestamped log file (`train_YYYYMMDD_HHMMSS.log`) with line-buffered writes and RSS tracking (`/proc/{pid}/status VmRSS`) at every chunk and phase. Survives OOM kills.

**First full run result (2026-03-30):**
- Build: 18 chunks, 265M rows, RSS flat at 12.7GB → 0.7GB after freeing sales. ✓
- Read-back: OOM — `load_filtered_panel` on ~250M raw rows (~141M after dropna) still exceeded 32GB.
- Fix added: `--max-train-rows` flag to subsample during read-back (30M rows recommended).

---

## Memory Optimization — Iteration 4: Categorical Int16 Encoding (2026-03-30)

**Problem:** 6 categorical columns (BPDNAME, DLNAME, ITEM_CATEORY, ITEM_CATEORY_CODE, primary_store, primary_channel) stored as Python strings (~60 bytes each). With 141M rows × 6 cols = ~50GB from categoricals alone.

**Solution:** Build a global `string → int16` mapping before chunking, apply during chunk build.

**Implementation:**
1. Before iterating chunks, collect unique values per categorical column from the aggregated sales frame:
   ```python
   cat_mappings: dict[str, dict[str, int]] = {}
   for col in CATEGORICAL_FEATURES:
       uniques = frame[col].dropna().unique()
       cat_mappings[col] = {v: i for i, v in enumerate(uniques)}
   ```
2. In `_build_chunk_features`, map and cast:
   ```python
   for col in CATEGORICAL_FEATURES:
       panel[col] = panel[col].map(cat_mappings[col]).fillna(-1).astype("int16")
   ```
3. Mappings saved to `cat_mappings.joblib` for inference.
4. LightGBM receives `categorical_feature=cat_feature_indices` (column indices into `ALL_FEATURES`) and handles int-coded categoricals natively.

**Cardinalities:**
| Column | Unique values |
|--------|--------------|
| ITEM_CATEORY_CODE | ~50 |
| ITEM_CATEORY | ~50 |
| primary_channel | ~5 |
| DLNAME | ~12K |
| BPDNAME | ~17K |
| primary_store | ~21K |

---

## Addendum — 2026-03-30 (Done By Mr. ChatGPT)

This section records follow-up changes made after the Claude work above.

### 1. Progress Bars Added

Progress bars were added around:
- chunk build/write in `build_featured_panel()`
- chunk read/filter in `load_filtered_panel()`

Implementation detail:
- Uses `tqdm.auto.tqdm` when available
- Falls back cleanly to normal iteration if `tqdm` is not installed

This was added to make long build/load phases observable without changing the training logic.

### 2. Explicit CPU/GPU CLI Selection

The CLI now supports:

```bash
--device gpu
--device cpu
```

This is wired through `train_lgbm.py` into `LGBMTrainerConfig.device`.

### 3. Automatic GPU → CPU Fallback

Problem observed:
- LightGBM GPU training failed on high-cardinality categorical features with errors such as:
  - `bin size 527 cannot run on GPU`

Fix added by Mr. ChatGPT:
- If LightGBM training raises a GPU-side `LightGBMError`, the trainer now logs the GPU failure and automatically retries that horizon on CPU.
- This fallback happens inside the training path, so the same CLI command can still be launched with `--device gpu`; if GPU is valid it stays on GPU, otherwise it falls back to CPU for that horizon.

Result:
- GPU remains the preferred path
- CPU retry is automatic when GPU training is rejected by LightGBM
- Actual device used is printed in the per-horizon training output

### 4. Streaming Horizon Training To Remove `pd.concat` OOM

Problem observed after chunked parquet build:
- build phase was stable
- read-back still OOM'd after all filtered chunks were loaded
- root cause was the final in-memory `pd.concat(filtered_chunks)` on ~141M usable rows

Fix added by Mr. ChatGPT:
- removed the horizon-level giant DataFrame concat path
- horizon training now runs in streaming stages:
  1. scan filtered chunk files to determine usable split dates
  2. stream rows into on-disk `train` / `valid` staging files
  3. train LightGBM from staged files instead of a monolithic pandas frame
  4. stream filtered chunks again for validation/test prediction and metrics
  5. write `test_predictions.csv` incrementally

Why this matters:
- the old path held many filtered DataFrames plus one newly concatenated DataFrame at the same time
- the new path avoids the big in-RAM assembly step entirely
- memory usage should now be dominated by one filtered chunk at a time plus LightGBM dataset construction

### 5. Float16 Overflow Guard Tightened

Observed warning:
- `RuntimeWarning: overflow encountered in cast`

Likely cause:
- some rolling means were still being downcast to `float16`
- popular items can push even means above `65504`

Fix added by Mr. ChatGPT:
- kept `roll_mean_7`, `roll_mean_14`, `roll_mean_28`, and `roll_mean_28_yoy` as `float32`
- left only small bounded counters such as `nonzero_days_28` and `store_count_28` as `float16`

**Result:** Build-phase RSS dropped from 12.7GB → 8.1–11.3GB. Post-build: 0.6GB.

Second run with int16 cats:
- Build: 18 chunks, 265M rows, RSS 8.1–11.3GB → 0.6GB. ✓
- Read-back: reached 19GB at chunk 18/18 (141M rows) — close but OOM.

---

## Memory Optimization — Iteration 5: Parquet Column Pruning (2026-03-30)

**Problem:** Each parquet chunk stores targets for *all* horizons (e.g., `target_15d` and `target_30d`). When loading for a specific horizon, the other target column is loaded unnecessarily.

**Solution:** `load_filtered_panel()` passes `columns=load_cols` to `pd.read_parquet()`:
```python
load_cols = ["XSRQ", "INVENTORY_ITEM_ID"] + ALL_FEATURES + [target_col]
chunk = pd.read_parquet(f, columns=load_cols)
```

This skips the unused `target_Xd` column during read-back (~0.5–1GB saving for 141M rows).

**Status:** Implemented. Ready for full test run combining int16 cats + column pruning.

---

## Memory Budget Summary

| Phase | RSS | Notes |
|-------|-----|-------|
| Load transactions | ~6.5 GB | 32M txn rows |
| Build chunks (string cats) | ~12.7 GB | old — before int16 encoding |
| Build chunks (int16 cats) | ~8.1–11.3 GB | current |
| After build | ~0.6 GB | only parquet files on disk |
| Read-back (141M rows, int16 + col pruning) | ~17–19 GB | estimated — to be confirmed |
| Train (30M rows subsampled) | ~8–10 GB | confirmed working |

Target: fit 141M rows in <28GB so training has headroom for LightGBM internal copy.

---

## Optimization Decision Log

| Approach | Considered | Chosen | Reason |
|----------|-----------|--------|--------|
| Subsample `--max-train-rows` | Yes | No (for full run) | Random sampling breaks time density per item — not a time-period |
| Chunked panel build | Yes | Yes | Only option for 260M rows |
| Fused build+features | Yes | Yes | Avoids re-holding chunk in memory |
| float16 quantization | Yes | Yes (partial) | sums/lags overflow float16, kept float32 |
| int16 categorical encoding | Yes | Yes | 60 bytes → 2 bytes per cell, ~50GB saved |
| Parquet column pruning | Yes | Yes | Skip unused target columns per horizon |
| CUDA batched training | Considered | Not needed | LightGBM GPU handles this internally |
| Drop INVENTORY_ITEM_ID from bulk read | Fallback | Pending | Saves ~1.1GB if column pruning isn't enough |

---

## Run Commands

```bash
# Full run (target: fits 32GB with int16 cats + column pruning)
cd ~/Documents/book_predict_lgbm && conda run -n book_predict python -u train_lgbm.py --output-dir artifacts_lgbm

# Full run with row cap (safe fallback if OOM, loses time density)
cd ~/Documents/book_predict_lgbm && conda run -n book_predict python -u train_lgbm.py --max-train-rows 30000000 --output-dir artifacts_lgbm

# Smoke test
cd ~/Documents/book_predict_lgbm && conda run -n book_predict python -u train_lgbm.py \
  --max-items 5000 --panel-days 400 --output-dir artifacts_lgbm_smoke
```

---

## Git Layout

- `~/Documents/book_predict/` → branch `hgbr_residual_blend` (original HGBR trainer)
- `~/Documents/book_predict_lgbm/` → branch `lightGBM` / `main` (this repo)
- Remote: `git@github.com:SNI22/Book_predict.git`

---

## Addendum — 2026-03-30 (Cluster + GPU Stability Updates)

### 1. Subsampled Run Metrics (30M rows)

Latest summary:

```text
horizon=15d  best_iter=590  test_wape=0.6809  baseline_test_wape=0.8156
horizon=30d  best_iter=999  test_wape=0.6384  baseline_test_wape=0.7860
```

Observed improvement vs baseline:
- 15d: ~16.5% better WAPE
- 30d: ~18.8% better WAPE

Notes:
- 30d hitting `best_iter=999` indicates it likely still benefits from a higher boost-round cap.

### 2. Overflow Warning Root Cause + Hard Fix

Observed warning during chunk build:

```text
RuntimeWarning: overflow encountered in cast
```

Root cause:
- `DLNUM` values are around `10,010,001+` in metadata.
- Casting those values to `float16` overflowed (`float16` max is `65504`).

Fixes applied:
- `DLNUM` cast changed to `float32`.
- Added `_to_float16_safe(...)` helper that clips to float16 range before float16 casts.
- Applied safe casting to float16 fields (`week_of_year`, `item_age_days`, and bounded counters).

Result:
- Overflow warnings from pandas `.astype(...)` are prevented, even if ranges drift.

### 3. Threading Visibility for Cluster Runs

Added startup logging:
- prints `n_jobs`
- prints visible logical CPU count (`os.cpu_count()`)
- reports effective threading mode (`all visible logical cores` when `n_jobs=-1`)

This makes scheduler/cluster thread allocation explicit in logs.

### 4. GPU-Safe Mode Added

New CLI flags:
- `--gpu-safe`
- `--max-bin`
- `--max-cat-threshold`
- `--max-cat-codes`

Behavior:
- `--gpu-safe` defaults (GPU only):
  - `max_bin=255`
  - `max_cat_threshold=64`
  - `max_cat_codes=255` (if not explicitly set)
- High-cardinality categorical mappings are truncated by frequency when `max_cat_codes` is set; dropped categories map to unknown (`-1`).

Why:
- Reduces risk of LightGBM GPU failures like:
  - `bin size xxx cannot run on GPU`

### 5. Practical Guidance: Lowering Bins on GPU

Effect of reducing `max_bin`:
- Pros: lower GPU memory pressure, faster histogram build, fewer GPU bin-size failures.
- Cons: coarser numeric quantization, possible accuracy drop if too low.

Recommended sweep:
- `max_bin=255` (baseline safe)
- `max_bin=127`
- `max_bin=63`

Pick the lowest value that keeps WAPE stable for your horizons.

### 6. Recommended Cluster Command

```bash
cd ~/Documents/book_predict_lgbm && conda run -n book_predict python -u train_lgbm.py \
  --device gpu \
  --gpu-safe \
  --n-jobs ${SLURM_CPUS_PER_TASK:-16} \
  --max-train-rows 30000000 \
  --output-dir artifacts_lgbm_30m_gpu_safe
```

---

## Addendum — 2026-04-05 (Metadata Column Pruning)

### Dropped Columns from Metadata Load

`DESCRIPTION`, `ISBN`, and `UN_NUMBER` are now excluded from the metadata CSV read via `usecols` in `load_and_aggregate_sales()`. These columns were previously loaded as strings, joined into `agg`, and carried through the entire build pipeline without ever being used as features.

| Column | What it is | Why dropped |
|--------|-----------|-------------|
| `DESCRIPTION` | Book title string (Chinese text) | Never used as a feature; large string per item |
| `ISBN` | International Standard Book Number | Never used as a feature |
| `UN_NUMBER` | Numeric subject classification code (Chinese library taxonomy, e.g. `112` = National Standards, `5502` = Ethics) | More granular than `ITEM_CATEORY_CODE` but not yet added as a feature |

**Note on `UN_NUMBER`:** This could be a useful categorical feature in future — it appears to encode finer-grained subject classification than the existing `ITEM_CATEORY_CODE` (~50 levels). Worth adding back if subject-level granularity is found to be underfit.

---

## Addendum — 2026-04-05 (Feature Expansion + Hyperparameter Tuning)

### Motivation

3yr run (`artifacts_lgbm_3yr`) completed with improved results vs 2yr baseline:
- 15d: test WAPE 0.6821 → 0.6608 (+3.1%)
- 30d: test WAPE 0.6384 → 0.6196 (+3.0%)

Key observations from feature importance:
- `days_since_sale` remains overwhelmingly #1
- `is_month_start` and `is_month_end` showed 0.0 importance in all runs → dropped
- `day_of_week` near-zero importance → replaced with `is_weekend`
- 30d horizon lacks long-window features aligned to its prediction horizon

### Changes Implemented

#### New Features

**Category-level dynamic features** (computed once from full frame before chunk loop, joined per chunk):
- `category_roll_mean_28` — average daily QTY across all items in the same `ITEM_CATEORY` over last 28 days (shift(1) to avoid leakage). Captures "philosophy books are trending up this month".
- `item_share_of_category` — `roll_mean_28 / (category_roll_mean_28 + 1e-6)`. Item's share of its category's recent sales.
- `category_yoy_ratio` — category-level YoY growth rate. Especially useful for sparse items with weak personal history.

**Longer lags and rolling windows** (aligned to 30d horizon):
- `lag_91`, `lag_182` — quarterly and semi-annual lags
- `roll_sum_91`, `roll_mean_91` — 91-day rolling window

**Derived ratios:**
- `velocity_ratio` = `roll_mean_7 / (roll_mean_28 + 1e-6)` — is item accelerating or decelerating?
- `yoy_ratio` = `roll_mean_28 / (roll_mean_28_yoy + 1e-6)` — normalised YoY growth signal

**UN_NUMBER added as categorical** — finer-grained subject classification code (e.g. `5502` = Ethics). More granular than `ITEM_CATEORY_CODE` (~50 levels).

#### Dropped Features
- `is_month_start`, `is_month_end` — 0.0 feature importance in all runs
- `day_of_week` — near-zero importance; replaced by `is_weekend` (binary, more signal-dense)

#### New Calendar Features
- `is_weekend` = `(dayofweek >= 5).astype("int8")`
- `quarter` = `cal.quarter.astype("int8")` — captures book sales seasonality (e.g. exam cycles)

#### Hyperparameter Changes

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `num_leaves` | 255 | 511 | More capacity for expanded feature set + 3yr data |
| `learning_rate` | 0.05 | 0.03 | Finer convergence with higher round cap |
| `min_child_samples` | 200 | 50 | Less conservative — allows model to learn sparse/long-tail item patterns |
| `num_boost_round` | 1000 | 2000 | 30d previously hit best_iter=905/999 — likely still improving |

### Dataset Path Note

Datasets are now local to the repo: `./dataset_large/TMPNXJ202603271.csv` and `./dataset_large/TMPNXJ202603272.csv`.

### Run Command

```bash
conda run -n book_predict python -u train_lgbm.py \
  --txn-path ./dataset_large/TMPNXJ202603271.csv \
  --meta-path ./dataset_large/TMPNXJ202603272.csv \
  --output-dir artifacts_lgbm_v2 \
  --horizons 15 30 \
  --min-history-days 10 \
  --panel-days 1095 \
  --max-train-rows 30000000 \
  --device gpu \
  --gpu-safe \
  --n-jobs -1 \
  --build-workers 16 \
  --random-state 42
```

---

## Addendum — 2026-04-06 (P1 Bug Fix: yoy_ratio saturation)

### Bug

`yoy_ratio` was computed as:
```python
panel["yoy_ratio"] = panel["roll_mean_28"] / (panel["roll_mean_28_yoy"].fillna(0) + 1e-6)
```

When `roll_mean_28_yoy` is NaN (item has < ~1 year of history), `fillna(0)` made the denominator `1e-6`, producing values ~10,000× inflated. After float16 quantization these saturated to the float16 max (~65504), corrupting training data for all new/young items.

The comment said "zero when yoy is absent" — the code did the opposite.

### Fix

```python
panel["yoy_ratio"] = np.where(
    panel["roll_mean_28_yoy"].notna(),
    panel["roll_mean_28"] / (panel["roll_mean_28_yoy"] + 1e-6),
    0.0,
)
```

Items with < 1 year of history now get `yoy_ratio=0` as intended.

### Impact

Comparison of `artifacts_lgbm_3yr` runs before vs after fix (same 3yr panel, same command):

| Horizon | Split | Before fix | After fix | Delta |
|---------|-------|-----------|-----------|-------|
| 15d | valid | MAE=2.14, WAPE=0.6728 | MAE=2.11, WAPE=0.6650 | −0.03, −0.008 |
| 15d | test  | MAE=2.10, WAPE=0.6608 | MAE=2.07, WAPE=0.6524 | −0.03, −0.008 |
| 30d | valid | MAE=4.06, WAPE=0.6461 | MAE=3.92, WAPE=0.6241 | −0.14, −0.022 |
| 30d | test  | MAE=3.88, WAPE=0.6196 | MAE=3.78, WAPE=0.6040 | −0.10, −0.016 |

30d benefited more — it relies more heavily on the YoY signal over longer windows, and new items were polluting training with saturated values.

### Remaining Known Issue (P2)

Category rolling features (`category_roll_mean_28`, `category_yoy_ratio`) are computed on sale-event dates rather than calendar days. For sparse categories, zero-sale days drop out of the rolling window, making the "28-day" window span more than 28 calendar days. `item_share_of_category` also blows up on zero-sale merge dates. Fix requires reindexing `cat_daily` to a full dense date range per category — deferred to next iteration.

---

## Addendum — 2026-04-06 (Temporal Subsampling + Multi-GPU + Output Structure)

### Temporal Subsampling

**Problem:** `--max-train-rows 30000000` applied uniform random sampling across all 3yr training rows (~200M). With a 3yr panel, the most recent 30 days got the same sparse sampling (~7.7%) as data from 2 years ago, causing the model to undertrain on the distribution closest to test time.

**Fix:** Time-aware sampling with a `recent_days` window (default 365):
- Rows within the last `recent_days` of `train_end` are kept at full rate
- Older rows fill the remaining budget at a lower rate
- If recent rows alone exceed the cap, sample uniformly from the recent window only — still better than uniform across all 3 years

**New CLI flag:** `--recent-days N` (default 365)

**Implementation:** Modified `_write_train_valid_files` in `trainer_lgbm.py`:
- One count pass: tallies `recent_count` and `old_count` separately
- Computes `recent_keep_prob` and `old_keep_prob` with the budget math
- Write pass: applies per-row probability via `np.where(is_recent, recent_keep_prob, old_keep_prob)`

### Multi-GPU Parallel Horizon Training

**Problem:** With 4× RTX A4000 GPUs (16GB VRAM each, CUDA 12.2), horizons were trained sequentially on a single GPU, leaving 3 idle.

**Fix:** When `--device cuda` or `--device gpu` and multiple horizons are requested, horizons are automatically distributed across GPUs:
- horizon[0] (15d) → GPU 0
- horizon[1] (30d) → GPU 1
- Uses `ThreadPoolExecutor` — LightGBM's C++ releases the GIL so threads are safe
- Parquet staging goes to separate tmpdirs per horizon (no conflicts)
- Auto-assign disabled when `--gpu-device-id` is set manually

**New config field / CLI flag:** `gpu_device_id: int | None` / `--gpu-device-id N`

**Smoke test result (2K items):** CPU: 24s/23s per horizon sequentially → CUDA multi-GPU: 11.9s/12.4s in parallel.

### CUDA Installation

The default PyPI `lightgbm` package is CPU-only. CUDA-enabled build installed via conda-forge:

```bash
conda install -n book_predict -c conda-forge lightgbm=4.6.0=cuda_py_4 -y
```

### Timestamped Output Structure

**Problem:** Each run overwrote the previous `horizon_15d/` and `horizon_30d/` directories.

**Fix:** Each run now creates a timestamped subfolder: `<output-dir>/YYYYMMDD_HHMMSS/`

```
artifacts_lgbm_3yr_cuda/
  20260406_033625/
    train_20260406_033625.log
    training_summary.csv
    horizon_15d/  model.lgb, metrics.csv, feature_importance.csv, test_predictions.csv, model_meta.joblib
    horizon_30d/  (same)
```

**Implementation:** `train_lgbm.py` generates timestamp at startup, creates `base_dir / timestamp` as `run_dir`, passes that to both `setup_logging` and `LGBMTrainerConfig.output_dir`.

### Current Best Run Command (CUDA, 3yr panel)

```bash
conda run -n book_predict python -u train_lgbm.py --txn-path ./dataset_large/TMPNXJ202603271.csv --meta-path ./dataset_large/TMPNXJ202603272.csv --output-dir artifacts_lgbm_3yr_cuda --horizons 15 30 --min-history-days 10 --panel-days 1095 --max-train-rows 30000000 --device cuda --gpu-safe --n-jobs -1 --build-workers 16 --random-state 42
```

### Known Remaining Issues (Next Iteration)

| # | Issue | Impact |
|---|-------|--------|
| P2 | Category rolling features computed on sale-event dates, not calendar days | Medium — affects sparse categories |
| - | `is_weekend`, `lag_14`, `lag_28`, `primary_channel`, `DLNUM` near-zero feature importance | Low — cleanup |
| - | `store_count_7`, `store_count_14` not yet added (#1 feature is `store_count_28`) | Medium |
| - | `min_history_days=10` too low — items with <30 days have mostly NaN features | Medium |

---

## Session — 2026-04-25  Item Segmentation (Plan B diagnostic)

### Context
Reviewed `xsyc/` (a small per-item LightGBM script) versus the main `train_lgbm.py`
pipeline. xsyc reports MAE ≈ 14.8 / WAPE ≈ 43.8% but with **R² ≈ −1.79** — i.e. it
loses to the naive mean. Confirmed by writing `xsyc/baseline_compare.py`, which
shows a 7-day rolling-mean baseline matches xsyc's LightGBM (MAE 14.85, WAPE 42.4%).
Conclusion: xsyc's "good" numbers are an artefact of small per-SKU magnitudes plus
a leak-prone in-series 80/20 split with early stopping on the test set.

In contrast, the main pipeline reports test WAPE 64–68% on ~25M rows with a
**−13.5 to −14.8 pp lift over baseline** — real learned signal.

### User idea
Group items by sales pattern (常销 / 中销 / 零散 / 季节) and predict differently per
group. Recommended **Plan B**: keep the single LightGBM model and add segment
labels as features, so the booster can specialize internally without losing
cross-item learning. Data-driven seasonality detection (no manual labels).

### Implemented
- `src/book_predict/segmentation.py`
  - `SegmentationConfig` (window 90d, regular ≥ 60 nonzero days, medium ≥ 30,
    sparse ≥ 5, cold otherwise; seasonal via `corr(QTY_t, QTY_{t-365}) ≥ 0.30`
    or `max(monthly)/mean(monthly) ≥ 2.0` with ≥ 365 days history).
  - `compute_item_segments(panel)` — one row per item with segment label,
    `nonzero_days_90`, `seasonality_score`, `yoy_strength`, `history_days`.
  - `enrich_panel_with_segments(panel)` — int16-encoded segment + numeric stats
    merged onto a featured panel; ready to add into `ALL_FEATURES`.
  - `per_segment_metrics(y, ŷ, codes)` — MAE / WAPE per segment.
- `diagnose_segments.py` — CLI; loads parquet chunks, scores the existing
  trained model, reports per-segment MAE / WAPE. Saves `item_segments.csv`
  and `segment_report_<h>d.csv`. Does NOT retrain.
- `src/book_predict/trainer_lgbm.py`
  - Added `KEEP_CHUNKS=1` env-var guard so training preserves the parquet
    chunks under `<output-dir>/chunks/` for the diagnostic. Default behaviour
    (delete chunks) unchanged.
- `xsyc/baseline_compare.py` — dependency-free baselines (mean / last-value /
  rolling-7-mean) on the xsyc test slice; saves `xsyc/baseline_对比结果.csv`.

### CLI to run
```bash
# 1. Re-train with chunks preserved (one-time)
KEEP_CHUNKS=1 conda run -n book_predict python -u train_lgbm.py \
    --device cuda --gpu-safe --max-train-rows 30000000 \
    --output-dir artifacts_lgbm

# 2. Per-segment diagnostic
conda run -n book_predict python -u diagnose_segments.py \
    --chunks-dir artifacts_lgbm/chunks \
    --model artifacts_lgbm/horizon_30d/model.lgb \
    --horizon 30 \
    --output artifacts_lgbm/segment_report_30d.csv
```

### Decision criterion for next step
If `sparse + cold` segments dominate row count but produce most of the WAPE,
or if `seasonal` items show a YoY signal the current model isn't using, then
integrate `enrich_panel_with_segments` into `_build_chunk_features` and add
`SEGMENT_FEATURES` to `ALL_FEATURES`. Otherwise, keep the diagnostic as-is and
treat segments as a reporting lens.

### Validation
- `python3 -m py_compile src/book_predict/trainer_lgbm.py` — passes.
- `python3 -m py_compile src/book_predict/segmentation.py diagnose_segments.py` — passes.
- `xsyc/baseline_compare.py` ran end-to-end on the local 140K-row dataset.

---

## 2026-04-25 — Segmented training, hyperparameter tuning, panel enrichment

### Per-segment hyperparameter tuning (`train_segmented.py`)
Initial segmented run (15d): OVERALL WAPE **0.6121** (vs 0.6733 unsegmented — a ~9% relative improvement). 30d run gave per-segment WAPE: regular 0.530, medium 0.429, sparse 0.510, cold 1.597, seasonal 0.637 — `cold` is the dominant drag.

Tuned per-segment params based on observed `best_iter` from the prior run:
- **Base LR** 0.05 → 0.03; default `--early-stopping` 50 → 100.
- **regular**: kept `lr=0.05`, `num_leaves=127`, `min_child_samples=50` (tiny n).
- **medium**: `num_leaves=511`, `min_child_samples=100` (1.6M rows can support deeper trees).
- **seasonal**: `num_leaves=511` (already), `num_boost_round` cap raised to 6000 (was hitting 545).
- **cold**: kept `tweedie 1.5`, added `lambda_l2=5.0`.
- `num_boost_round` caps raised across the board so early stopping decides.

### `--build-only` flag
Added `LGBMTrainerConfig.build_only` and `train_lgbm.py --build-only`. After `build_featured_panel` runs, chunks are copied to `<output-dir>/chunks/`, `cat_mappings.joblib` is saved, and the trainer exits — skipping the (expensive) per-horizon staging + training. Lets us iterate on feature engineering without re-running the load/aggregate phase.

### `enrich_panel.py` (Phase-1 feature engineering)
Non-destructive post-processor: reads chunks from `--in-dir`, writes augmented chunks to `--out-dir`. Added columns:
- `holiday_type` int16 (mainland China holidays 2022-2027 hardcoded), `days_to_holiday`, `days_since_holiday`
- `discount_ratio_28` = `avg_revenue_per_unit_28 / LIST_PRICE_PER_UNIT` (proxy for active discount)
- `store_demand_lag1_mean` — cross-item demand: mean of `lag_1` across items sharing the same `primary_store` on the same date

The segmented trainer auto-discovers new columns via `_discover_features` (int16 small-range → categorical, others → numeric), so no trainer changes were needed.

Phase 2 (stockout-corrected rolling means) deferred — would require recomputing existing lag/roll features.

### Workflow
```bash
# 1. Build once (server preferred)
conda run -n book_predict python -u train_lgbm.py \
    --device gpu --gpu-safe --output-dir artifacts_lgbm/run_x --build-only

# 2. Enrich (chunk-by-chunk, low RAM)
conda run -n book_predict python -u enrich_panel.py \
    --in-dir  artifacts_lgbm/run_x/<ts>/chunks \
    --out-dir artifacts_lgbm_enriched/horizon_30d/chunks \
    --io-workers 4

# 3. Train segmented
conda run -n book_predict python -u train_segmented.py \
    --chunks-dir artifacts_lgbm_enriched/horizon_30d/chunks \
    --horizon 30 --output-dir artifacts_lgbm_segmented_v2/horizon_30d \
    --device gpu --gpu-safe --io-workers 4
```

### Open questions
- Will lower base LR + bigger caps actually help, or just train longer for the same WAPE? Need fresh run to confirm.
- Does `holiday_type` add real lift on 30d horizon (15d window may straddle the holiday already)?
- Cold WAPE 1.6 — even tweedie + heavier reg may not help. Likely needs a two-stage P(sale>0) × E[sale|sale>0] model; that's the next big lever.
