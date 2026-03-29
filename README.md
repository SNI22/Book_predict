# Book Sales Trainer

This repository contains a first-pass trainer for book sales forecasting based on `INVENTORY_ITEM_ID`.

## What it does

- Reads the transaction CSV in `dataset/`
- Aggregates sales to item-day net sales
- Builds a recent dense daily panel for items with enough sales history
- Creates lag, rolling, and calendar features
- Trains global regression models for 15-day and 30-day net sales forecasting
- Compares model performance with a rolling 28-day baseline

## Default assumptions

- Forecast entity: `INVENTORY_ITEM_ID`
- Negative `QTY`: kept as returns, therefore targets are net sales
- Default horizons: 15 and 30 days
- Default history filter: items with at least 10 unique sales dates

## Datasets

### `dataset/` (original)
- Single flat CSV: `TMPNXJ20260322.csv`
- ~1.64M rows, 203K unique items
- Date range: 2024-01-01 → 2026-03-18 (~2.2 years)
- Schema: `INVENTORY_ITEM_ID, ISBN, DESCRIPTION, LIST_PRICE_PER_UNIT, ITEM_CATEORY_CODE, ITEM_CATEORY, UN_NUMBER, BPDNAME, DLNUM, DLNAME, XSRQ, QTY`
- Includes negative `QTY` (returns)

### `dataset_large/` (extended)
- Split into two files:
  - `TMPNXJ202603271.csv` — transactions (~32.3M rows)
    - Schema: `MDHM` (store ID), `XSRQ` (date), `INVENTORY_ITEM_ID`, `XSPC` (channel), `QTY`, `XSJE` (revenue)
    - QTY is positive only (no returns)
  - `TMPNXJ202603272.csv` — item metadata (~1.16M unique items)
    - Schema: `INVENTORY_ITEM_ID, ISBN, DESCRIPTION, LIST_PRICE_PER_UNIT, ITEM_CATEORY_CODE, ITEM_CATEORY, UN_NUMBER, BPDNAME, DLNUM, DLNAME`
- Date range: ~2019 → 2026-03-27 (~7 years)
- **Note:** the trainer currently expects a single flat file; using `dataset_large` requires joining the two files before passing to `load_and_aggregate_sales()`.

## Run

Use the dedicated conda environment:

```bash
conda run -n book_predict python train_sales_model.py
```

Example with tighter scope for a quicker dry run:

```bash
conda run -n book_predict python train_sales_model.py --max-items 5000 --panel-days 240
```

To point at the large dataset (after adapting the loader):

```bash
conda run -n book_predict python train_sales_model.py \
  --data-path dataset_large/TMPNXJ202603271.csv \
  --output-dir artifacts_large \
  --panel-days 730
```

## Outputs

Files are written under `artifacts/`:

- `training_summary.csv`
- `horizon_15d/model.joblib`
- `horizon_15d/metrics.csv`
- `horizon_15d/test_predictions.csv`
- `horizon_30d/model.joblib`
- `horizon_30d/metrics.csv`
- `horizon_30d/test_predictions.csv`
