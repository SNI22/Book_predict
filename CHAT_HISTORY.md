# Chat History — LightGBM Direct Trainer

**Date:** 2026-03-29 → 2026-03-30

## Context

Original trainer (branch `hgbr_residual_blend`) used HGBR on residuals of rolling 28d mean, blend capped at 10%. Poor results (WAPE >1.0 on 15d valid). This branch replaces it with LightGBM predicting targets directly on `dataset_large` (~32.3M txns, 7 years, split into transactions + metadata files). GPU-accelerated (RTX 5070).

## Smoke Test Results (5K items, 400 panel days)

| Horizon | WAPE (test) | Baseline WAPE (test) |
|---|---|---|
| 15d | 0.6552 | 0.7360 |
| 30d | 0.5909 | 0.6637 |

Full run not yet completed (OOM during panel build → fixed with chunked approach → GPU driver crashed during retry).

## Bugs Fixed

1. **Slow aggregation:** per-group `lambda mode()` on 32M rows → vectorized item-level mode via `groupby + size + drop_duplicates`
2. **Slow dense panel:** Python for-loop over 357K items → cross-join + left-join + groupby ffill/bfill
3. **OOM on panel build:** full cross-join (357K × 730 = 260M rows) exceeded 32GB RAM → chunked into 20K-item batches + float32 downcasting

## Run Commands

```bash
# Prefix: cd ~/Documents/book_predict_lgbm && conda run -n book_predict python -u train_lgbm.py
# Smoke:  --max-items 5000 --panel-days 400 --output-dir artifacts_lgbm_smoke
# Medium: --max-items 50000 --panel-days 730 --output-dir artifacts_lgbm_medium
# Full:   --panel-days 730 --output-dir artifacts_lgbm_full
```

## Git Layout

- `~/Documents/book_predict/` → `hgbr_residual_blend` (original trainer)
- `~/Documents/book_predict_lgbm/` → `lgbm_direct` / `lightGBM` (this, also main)
- Remote: `git@github.com:SNI22/Book_predict.git`
