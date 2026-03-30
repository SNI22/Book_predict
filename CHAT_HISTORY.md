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
