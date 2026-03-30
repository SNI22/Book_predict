from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

STATIC_COLUMNS = [
    "LIST_PRICE_PER_UNIT",
    "ITEM_CATEORY_CODE",
    "ITEM_CATEORY",
    "BPDNAME",
    "DLNUM",
    "DLNAME",
]

CATEGORICAL_FEATURES = [
    "ITEM_CATEORY_CODE",
    "ITEM_CATEORY",
    "BPDNAME",
    "DLNAME",
    "primary_store",
    "primary_channel",
]

NUMERIC_FEATURES = [
    "LIST_PRICE_PER_UNIT",
    "DLNUM",
    # short-term lags
    "lag_1", "lag_7", "lag_14", "lag_28",
    # rolling sums / means
    "roll_sum_7", "roll_sum_14", "roll_sum_28",
    "roll_mean_7", "roll_mean_14", "roll_mean_28",
    "nonzero_days_28",
    "days_since_sale",
    # year-over-year
    "lag_365",
    "roll_mean_28_yoy",
    # revenue-derived
    "avg_revenue_per_unit_28",
    "store_count_28",
    # calendar
    "day_of_week", "day_of_month", "month", "week_of_year",
    "is_month_start", "is_month_end",
    "item_age_days",
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LGBMTrainerConfig:
    txn_path: Path                          # large dataset transactions
    meta_path: Path                         # large dataset item metadata
    output_dir: Path
    horizons: list[int] = field(default_factory=lambda: [15, 30])
    min_history_days: int = 10
    panel_days: int = 730
    max_items: int | None = None
    random_state: int = 42
    n_jobs: int = -1


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_aggregate_sales(config: LGBMTrainerConfig) -> pd.DataFrame:
    print("Loading transactions...")
    txn = pd.read_csv(
        config.txn_path,
        encoding="utf-8-sig",
        parse_dates=["XSRQ"],
        dtype={"INVENTORY_ITEM_ID": "int64", "QTY": "float32", "XSJE": "float32"},
    )

    print("Loading item metadata...")
    meta = pd.read_csv(
        config.meta_path,
        encoding="utf-8-sig",
        dtype={"INVENTORY_ITEM_ID": "int64"},
    )

    # Fill metadata nulls
    for col, fill in [
        ("BPDNAME", "UNKNOWN_PUBLISHER"),
        ("ITEM_CATEORY_CODE", "UNKNOWN_CATEGORY_CODE"),
        ("ITEM_CATEORY", "UNKNOWN_CATEGORY"),
        ("DLNAME", "UNKNOWN_DLNAME"),
    ]:
        meta[col] = meta[col].fillna(fill)

    txn["MDHM"] = txn["MDHM"].fillna("UNKNOWN_STORE").astype(str)
    txn["XSPC"] = txn["XSPC"].fillna(-1).astype(str)

    print("Aggregating to item-day...")
    # Fast vectorised aggregation — avoid per-group lambdas on 32M rows
    agg = (
        txn.groupby(["INVENTORY_ITEM_ID", "XSRQ"], as_index=False)
        .agg(
            QTY=("QTY", "sum"),
            XSJE=("XSJE", "sum"),
            store_count=("MDHM", "nunique"),
        )
        .sort_values(["INVENTORY_ITEM_ID", "XSRQ"])
        .reset_index(drop=True)
    )

    # Derive primary_store / primary_channel at item level (not item-day)
    # — mode per item, computed once, then joined back
    print("Computing per-item store/channel modes...")
    item_store = (
        txn.groupby(["INVENTORY_ITEM_ID", "MDHM"], sort=False)
        .size()
        .reset_index(name="cnt")
        .sort_values("cnt", ascending=False)
        .drop_duplicates(subset="INVENTORY_ITEM_ID", keep="first")
        .rename(columns={"MDHM": "primary_store"})
        [["INVENTORY_ITEM_ID", "primary_store"]]
    )
    item_channel = (
        txn.groupby(["INVENTORY_ITEM_ID", "XSPC"], sort=False)
        .size()
        .reset_index(name="cnt")
        .sort_values("cnt", ascending=False)
        .drop_duplicates(subset="INVENTORY_ITEM_ID", keep="first")
        .rename(columns={"XSPC": "primary_channel"})
        [["INVENTORY_ITEM_ID", "primary_channel"]]
    )
    agg = agg.merge(item_store, on="INVENTORY_ITEM_ID", how="left")
    agg = agg.merge(item_channel, on="INVENTORY_ITEM_ID", how="left")
    agg["primary_store"] = agg["primary_store"].fillna("UNKNOWN_STORE")
    agg["primary_channel"] = agg["primary_channel"].fillna("-1")

    print("Joining item metadata...")
    agg = agg.merge(meta, on="INVENTORY_ITEM_ID", how="left")

    for col, fill in [
        ("BPDNAME", "UNKNOWN_PUBLISHER"),
        ("ITEM_CATEORY_CODE", "UNKNOWN_CATEGORY_CODE"),
        ("ITEM_CATEORY", "UNKNOWN_CATEGORY"),
        ("DLNAME", "UNKNOWN_DLNAME"),
        ("LIST_PRICE_PER_UNIT", 0.0),
        ("DLNUM", 0),
    ]:
        agg[col] = agg[col].fillna(fill)

    return agg


# ---------------------------------------------------------------------------
# Item selection
# ---------------------------------------------------------------------------

def select_eligible_items(
    frame: pd.DataFrame, min_history_days: int, max_items: int | None
) -> np.ndarray:
    history = frame.groupby("INVENTORY_ITEM_ID")["XSRQ"].nunique()
    eligible = history[history >= min_history_days]
    if max_items is not None:
        eligible = eligible.sort_values(ascending=False).head(max_items)
    return eligible.index.to_numpy()


# ---------------------------------------------------------------------------
# Dense panel
# ---------------------------------------------------------------------------

def _build_chunk_panel(
    chunk_items: np.ndarray,
    frame: pd.DataFrame,
    all_dates: pd.DatetimeIndex,
    item_bounds: pd.DataFrame,
) -> pd.DataFrame:
    """Build dense panel for a subset of items."""
    bounds = item_bounds[item_bounds["INVENTORY_ITEM_ID"].isin(chunk_items)]
    date_df = pd.DataFrame({"XSRQ": all_dates})

    spine = bounds.merge(date_df, how="cross")
    spine = spine[spine["XSRQ"] >= spine["panel_start"]].drop(columns=["panel_start"])

    chunk_frame = frame[frame["INVENTORY_ITEM_ID"].isin(chunk_items)]
    panel = spine.merge(chunk_frame, on=["INVENTORY_ITEM_ID", "XSRQ"], how="left")

    panel["QTY"] = panel["QTY"].fillna(0.0).astype("float32")
    panel["XSJE"] = panel["XSJE"].fillna(0.0).astype("float32")
    panel["store_count"] = panel["store_count"].fillna(0.0).astype("float32")

    static_cols = STATIC_COLUMNS + ["primary_store", "primary_channel"]
    panel = panel.sort_values(["INVENTORY_ITEM_ID", "XSRQ"]).reset_index(drop=True)
    grouped = panel.groupby("INVENTORY_ITEM_ID", sort=False)
    for col in static_cols:
        panel[col] = grouped[col].ffill()
        panel[col] = grouped[col].bfill()

    return panel


def build_dense_panel(
    frame: pd.DataFrame,
    eligible_items: np.ndarray,
    panel_days: int,
    max_horizon: int,
    chunk_size: int = 20_000,
) -> pd.DataFrame:
    print("Building dense panel (chunked)...")
    frame = frame[frame["INVENTORY_ITEM_ID"].isin(eligible_items)].copy()

    # Downcast numerics to save memory
    frame["QTY"] = frame["QTY"].astype("float32")
    frame["XSJE"] = frame["XSJE"].astype("float32")

    latest_date = frame["XSRQ"].max()
    panel_start_floor = latest_date - pd.Timedelta(days=panel_days + max_horizon + 35)

    item_first = frame.groupby("INVENTORY_ITEM_ID")["XSRQ"].min().rename("first_sale_date")
    item_start = item_first.clip(lower=panel_start_floor).rename("panel_start")
    all_dates = pd.date_range(start=panel_start_floor, end=latest_date, freq="D")

    item_bounds = pd.DataFrame({
        "INVENTORY_ITEM_ID": item_start.index,
        "panel_start": item_start.values,
        "first_sale_date": item_first.values,
    })

    # Process in chunks to avoid OOM
    n_chunks = max(1, len(eligible_items) // chunk_size)
    item_chunks = np.array_split(eligible_items, n_chunks)

    panels: list[pd.DataFrame] = []
    total_rows = 0
    for i, chunk_items in enumerate(item_chunks):
        chunk_panel = _build_chunk_panel(chunk_items, frame, all_dates, item_bounds)
        total_rows += len(chunk_panel)
        panels.append(chunk_panel)
        print(f"  Chunk {i + 1}/{len(item_chunks)}: {len(chunk_items):,} items, {len(chunk_panel):,} rows (cumulative: {total_rows:,})")

    panel = pd.concat(panels, ignore_index=True)
    print(f"  Panel shape: {panel.shape}")
    return panel


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _forward_sum(values: pd.Series, horizon: int) -> pd.Series:
    shifted = values.shift(-1)
    return shifted[::-1].rolling(horizon, min_periods=horizon).sum()[::-1]


def add_features(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    frame = panel.copy()
    g = frame.groupby("INVENTORY_ITEM_ID", sort=False)

    # Target
    frame[f"target_{horizon}d"] = g["QTY"].transform(lambda s: _forward_sum(s, horizon))

    # Lags
    frame["lag_1"] = g["QTY"].shift(1)
    frame["lag_7"] = g["QTY"].shift(7)
    frame["lag_14"] = g["QTY"].shift(14)
    frame["lag_28"] = g["QTY"].shift(28)
    frame["lag_365"] = g["QTY"].shift(365)

    # Rolling (on lag-1 shifted series to avoid leakage)
    shifted = g["QTY"].shift(1)
    rg = shifted.groupby(frame["INVENTORY_ITEM_ID"], sort=False)
    frame["roll_sum_7"] = rg.transform(lambda s: s.rolling(7, min_periods=7).sum())
    frame["roll_sum_14"] = rg.transform(lambda s: s.rolling(14, min_periods=14).sum())
    frame["roll_sum_28"] = rg.transform(lambda s: s.rolling(28, min_periods=28).sum())
    frame["roll_mean_7"] = frame["roll_sum_7"] / 7.0
    frame["roll_mean_14"] = frame["roll_sum_14"] / 14.0
    frame["roll_mean_28"] = frame["roll_sum_28"] / 28.0
    frame["nonzero_days_28"] = rg.transform(lambda s: s.gt(0).rolling(28, min_periods=28).sum())

    # Year-over-year rolling mean (28-day window ending 365 days ago)
    shifted_yoy = g["QTY"].shift(365)
    rg_yoy = shifted_yoy.groupby(frame["INVENTORY_ITEM_ID"], sort=False)
    frame["roll_mean_28_yoy"] = rg_yoy.transform(lambda s: s.rolling(28, min_periods=14).mean())

    # Days since last sale
    prior = frame["XSRQ"].where(frame["QTY"] > 0)
    prior = prior.groupby(frame["INVENTORY_ITEM_ID"], sort=False).ffill()
    frame["days_since_sale"] = (frame["XSRQ"] - prior).dt.days

    # Revenue-derived: avg revenue per unit over last 28 days
    shifted_rev = g["XSJE"].shift(1)
    rg_rev = shifted_rev.groupby(frame["INVENTORY_ITEM_ID"], sort=False)
    roll_rev_28 = rg_rev.transform(lambda s: s.rolling(28, min_periods=1).sum())
    roll_qty_28 = frame["roll_sum_28"].replace(0, np.nan)
    frame["avg_revenue_per_unit_28"] = roll_rev_28 / roll_qty_28
    frame["avg_revenue_per_unit_28"] = frame["avg_revenue_per_unit_28"].fillna(
        frame["LIST_PRICE_PER_UNIT"]
    )

    # Store count rolling 28d
    shifted_sc = g["store_count"].shift(1)
    rg_sc = shifted_sc.groupby(frame["INVENTORY_ITEM_ID"], sort=False)
    frame["store_count_28"] = rg_sc.transform(lambda s: s.rolling(28, min_periods=1).mean())

    # Calendar
    cal = frame["XSRQ"].dt
    frame["day_of_week"] = cal.dayofweek
    frame["day_of_month"] = cal.day
    frame["month"] = cal.month
    frame["week_of_year"] = cal.isocalendar().week.astype("int16")
    frame["is_month_start"] = cal.is_month_start.astype("int8")
    frame["is_month_end"] = cal.is_month_end.astype("int8")
    frame["item_age_days"] = (frame["XSRQ"] - frame["first_sale_date"]).dt.days

    frame = frame.drop(columns=["first_sale_date"])
    return frame


# ---------------------------------------------------------------------------
# Train/valid/test split
# ---------------------------------------------------------------------------

def split_by_time(
    frame: pd.DataFrame, horizon: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    usable = frame.dropna(subset=[f"target_{horizon}d"]).copy()
    unique_dates = np.sort(usable["XSRQ"].unique())
    if len(unique_dates) < 120:
        raise ValueError("Not enough history to create reliable time-based splits.")

    train_end = unique_dates[int(len(unique_dates) * 0.70)]
    valid_end = unique_dates[int(len(unique_dates) * 0.85)]

    train = usable[usable["XSRQ"] <= train_end].copy()
    valid = usable[(usable["XSRQ"] > train_end) & (usable["XSRQ"] <= valid_end)].copy()
    test = usable[usable["XSRQ"] > valid_end].copy()
    return train, valid, test


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(y_true - y_pred).sum() / max(np.abs(y_true).sum(), 1e-9))


def _metric_frame(
    y_true: np.ndarray, y_pred: np.ndarray, baseline: np.ndarray
) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "wape": _wape(y_true, y_pred),
        "baseline_mae": float(mean_absolute_error(y_true, baseline)),
        "baseline_wape": _wape(y_true, baseline),
    }


# ---------------------------------------------------------------------------
# LightGBM model
# ---------------------------------------------------------------------------

def build_lgbm_params(random_state: int, n_jobs: int) -> dict:
    return {
        "objective": "regression_l1",      # MAE — robust to QTY outliers
        "metric": "mae",
        "device": "gpu",
        "learning_rate": 0.05,
        "num_leaves": 255,
        "min_child_samples": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "seed": random_state,
        "n_jobs": n_jobs,
    }


# ---------------------------------------------------------------------------
# Per-horizon training
# ---------------------------------------------------------------------------

def train_for_horizon(
    panel: pd.DataFrame, config: LGBMTrainerConfig, horizon: int
) -> dict[str, object]:
    print(f"\n--- Horizon {horizon}d ---")
    featured = add_features(panel, horizon=horizon)
    featured = featured.dropna(
        subset=[
            f"target_{horizon}d",
            "lag_1", "lag_7", "lag_14", "lag_28",
            "roll_sum_7", "roll_sum_14", "roll_sum_28",
            "days_since_sale",
        ]
    ).copy()

    # Encode categoricals as pandas category (LightGBM reads these natively)
    for col in CATEGORICAL_FEATURES:
        featured[col] = featured[col].astype("category")

    target_col = f"target_{horizon}d"
    train, valid, test = split_by_time(featured, horizon)

    train_x, train_y = train[ALL_FEATURES], train[target_col].to_numpy()
    valid_x, valid_y = valid[ALL_FEATURES], valid[target_col].to_numpy()
    test_x, test_y = test[ALL_FEATURES], test[target_col].to_numpy()

    baseline_valid = valid["roll_mean_28"].to_numpy() * horizon
    baseline_test = test["roll_mean_28"].to_numpy() * horizon

    dtrain = lgb.Dataset(train_x, label=train_y, categorical_feature=CATEGORICAL_FEATURES, free_raw_data=False)
    dvalid = lgb.Dataset(valid_x, label=valid_y, categorical_feature=CATEGORICAL_FEATURES, reference=dtrain, free_raw_data=False)

    params = build_lgbm_params(config.random_state, config.n_jobs)
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=50),
    ]

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dvalid],
        callbacks=callbacks,
    )

    valid_pred = booster.predict(valid_x)
    test_pred = booster.predict(test_x)

    valid_metrics = _metric_frame(valid_y, valid_pred, baseline_valid)
    test_metrics = _metric_frame(test_y, test_pred, baseline_test)

    # Save
    output_dir = config.output_dir / f"horizon_{horizon}d"
    output_dir.mkdir(parents=True, exist_ok=True)

    booster.save_model(str(output_dir / "model.lgb"))
    joblib.dump({"booster_path": str(output_dir / "model.lgb"), "features": ALL_FEATURES}, output_dir / "model_meta.joblib")

    predictions = test[["INVENTORY_ITEM_ID", "XSRQ", target_col, "QTY"]].copy()
    predictions["prediction"] = test_pred
    predictions["baseline_prediction"] = baseline_test
    predictions.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame([
        {"split": "valid", **valid_metrics},
        {"split": "test", **test_metrics},
    ]).to_csv(output_dir / "metrics.csv", index=False, encoding="utf-8-sig")

    # Feature importance
    fi = pd.DataFrame({
        "feature": booster.feature_name(),
        "importance": booster.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    fi.to_csv(output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")

    print(f"  valid  MAE={valid_metrics['mae']:.2f}  WAPE={valid_metrics['wape']:.4f}  (baseline WAPE={valid_metrics['baseline_wape']:.4f})")
    print(f"  test   MAE={test_metrics['mae']:.2f}  WAPE={test_metrics['wape']:.4f}  (baseline WAPE={test_metrics['baseline_wape']:.4f})")

    return {
        "horizon": horizon,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
        "test_rows": int(len(test)),
        "best_iteration": int(booster.best_iteration),
        "output_dir": output_dir,
    }


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def run_training(config: LGBMTrainerConfig) -> list[dict[str, object]]:
    sales = load_and_aggregate_sales(config)
    eligible_items = select_eligible_items(
        sales,
        min_history_days=config.min_history_days,
        max_items=config.max_items,
    )
    print(f"Eligible items: {len(eligible_items):,}")
    if len(eligible_items) == 0:
        raise ValueError("No items satisfied the minimum history threshold.")

    panel = build_dense_panel(
        sales,
        eligible_items=eligible_items,
        panel_days=config.panel_days,
        max_horizon=max(config.horizons),
    )
    print(f"Panel shape: {panel.shape}")

    results = []
    for horizon in config.horizons:
        results.append(train_for_horizon(panel, config, horizon))

    summary = []
    for r in results:
        summary.append({
            "horizon": r["horizon"],
            "train_rows": r["train_rows"],
            "valid_rows": r["valid_rows"],
            "test_rows": r["test_rows"],
            "best_iteration": r["best_iteration"],
            "valid_mae": r["valid_metrics"]["mae"],
            "valid_wape": r["valid_metrics"]["wape"],
            "valid_baseline_wape": r["valid_metrics"]["baseline_wape"],
            "test_mae": r["test_metrics"]["mae"],
            "test_wape": r["test_metrics"]["wape"],
            "test_baseline_wape": r["test_metrics"]["baseline_wape"],
        })

    config.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(
        config.output_dir / "training_summary.csv", index=False, encoding="utf-8-sig"
    )
    return results
