from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Iterable

import os
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def _mem_gb() -> str:
    """Current process RSS in GB (Linux)."""
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return f"{kb / 1_048_576:.1f}GB"
    except Exception:
        pass
    return "?GB"


def _iter_with_progress(iterable, total: int, desc: str):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit="chunk")


_FLOAT16_MAX = np.finfo(np.float16).max
_BUILD_WORKER_CTX: dict[str, object] | None = None


def _to_float16_safe(values: pd.Series) -> pd.Series:
    """Clip to float16 range before casting to avoid RuntimeWarning overflow."""
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.clip(lower=-_FLOAT16_MAX, upper=_FLOAT16_MAX).astype("float16")


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
    "UN_NUMBER",
]

CATEGORICAL_FEATURES = [
    "ITEM_CATEORY_CODE",
    "ITEM_CATEORY",
    "BPDNAME",
    "DLNAME",
    "UN_NUMBER",
    "primary_store",
    "primary_channel",
]

NUMERIC_FEATURES = [
    "LIST_PRICE_PER_UNIT",
    "DLNUM",
    # short-term lags
    "lag_1", "lag_7", "lag_14", "lag_28",
    # medium/long lags
    "lag_91", "lag_182",
    # rolling sums / means
    "roll_sum_7", "roll_sum_14", "roll_sum_28", "roll_sum_91",
    "roll_mean_7", "roll_mean_14", "roll_mean_28", "roll_mean_91",
    "nonzero_days_28",
    "days_since_sale",
    # year-over-year
    "lag_365",
    "roll_mean_28_yoy",
    # derived ratios
    "velocity_ratio",
    "yoy_ratio",
    # revenue-derived
    "avg_revenue_per_unit_28",
    "store_count_28",
    # category-level
    "category_roll_mean_28",
    "item_share_of_category",
    "category_yoy_ratio",
    # calendar
    "day_of_month", "month", "week_of_year",
    "is_weekend", "quarter",
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
    max_train_rows: int | None = None          # cap training rows (subsample if exceeded)
    recent_days: int = 365                     # rows within this many days of train_end are kept at full rate
    device: str = "gpu"
    random_state: int = 42
    n_jobs: int = -1
    max_bin: int | None = None
    max_cat_threshold: int | None = None
    max_cat_codes: int | None = None
    gpu_safe: bool = False
    build_workers: int = 1
    gpu_device_id: int | None = None    # None = auto-assign (horizon 0→GPU0, horizon 1→GPU1, …)


def _resolve_n_jobs(requested_n_jobs: int) -> int:
    """Resolve user n_jobs into an explicit positive thread count."""
    visible_cpus = os.cpu_count() or 1
    if requested_n_jobs in (-1, 0):
        return visible_cpus
    if requested_n_jobs < -1:
        return min(abs(requested_n_jobs), visible_cpus)
    return min(requested_n_jobs, visible_cpus)


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
        usecols=["INVENTORY_ITEM_ID", "LIST_PRICE_PER_UNIT",
                 "ITEM_CATEORY_CODE", "ITEM_CATEORY",
                 "BPDNAME", "DLNUM", "DLNAME", "UN_NUMBER"],
    )

    # Fill metadata nulls
    for col, fill in [
        ("BPDNAME", "UNKNOWN_PUBLISHER"),
        ("ITEM_CATEORY_CODE", "UNKNOWN_CATEGORY_CODE"),
        ("ITEM_CATEORY", "UNKNOWN_CATEGORY"),
        ("DLNAME", "UNKNOWN_DLNAME"),
        ("UN_NUMBER", "UNKNOWN_UN"),
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
        ("UN_NUMBER", "UNKNOWN_UN"),
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
# Dense panel + feature engineering (fused per-chunk)
# ---------------------------------------------------------------------------

def _forward_sum(values: pd.Series, horizon: int) -> pd.Series:
    shifted = values.shift(-1)
    return shifted[::-1].rolling(horizon, min_periods=horizon).sum()[::-1]


def _build_chunk_features(
    chunk_items: np.ndarray,
    frame: pd.DataFrame,
    all_dates: pd.DatetimeIndex,
    item_bounds: pd.DataFrame,
    horizons: list[int],
    cat_mappings: dict[str, dict[str, int]] | None = None,
    cat_features_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build dense panel for a subset of items AND compute all features in-place.

    Returns only the columns needed for training (ALL_FEATURES + targets + keys),
    dropping heavy intermediates (XSJE, store_count, raw QTY) before returning.
    If cat_mappings is provided, encodes categoricals as int16 codes.
    """
    # --- build dense panel ---
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

    # --- feature engineering (in-place on panel) ---
    g = panel.groupby("INVENTORY_ITEM_ID", sort=False)

    # Targets for each horizon
    for h in horizons:
        panel[f"target_{h}d"] = g["QTY"].transform(lambda s, hz=h: _forward_sum(s, hz))

    # Lags
    panel["lag_1"] = g["QTY"].shift(1)
    panel["lag_7"] = g["QTY"].shift(7)
    panel["lag_14"] = g["QTY"].shift(14)
    panel["lag_28"] = g["QTY"].shift(28)
    panel["lag_91"] = g["QTY"].shift(91)
    panel["lag_182"] = g["QTY"].shift(182)
    panel["lag_365"] = g["QTY"].shift(365)

    # Rolling (on lag-1 shifted series to avoid leakage)
    shifted = g["QTY"].shift(1)
    rg = shifted.groupby(panel["INVENTORY_ITEM_ID"], sort=False)
    panel["roll_sum_7"] = rg.transform(lambda s: s.rolling(7, min_periods=7).sum())
    panel["roll_sum_14"] = rg.transform(lambda s: s.rolling(14, min_periods=14).sum())
    panel["roll_sum_28"] = rg.transform(lambda s: s.rolling(28, min_periods=28).sum())
    panel["roll_sum_91"] = rg.transform(lambda s: s.rolling(91, min_periods=45).sum())
    panel["roll_mean_7"] = panel["roll_sum_7"] / 7.0
    panel["roll_mean_14"] = panel["roll_sum_14"] / 14.0
    panel["roll_mean_28"] = panel["roll_sum_28"] / 28.0
    panel["roll_mean_91"] = panel["roll_sum_91"] / 91.0
    panel["nonzero_days_28"] = rg.transform(lambda s: s.gt(0).rolling(28, min_periods=28).sum())
    # Velocity ratio: short-term vs medium-term trend (is item accelerating?)
    panel["velocity_ratio"] = panel["roll_mean_7"] / (panel["roll_mean_28"] + 1e-6)

    # Year-over-year rolling mean
    shifted_yoy = g["QTY"].shift(365)
    rg_yoy = shifted_yoy.groupby(panel["INVENTORY_ITEM_ID"], sort=False)
    panel["roll_mean_28_yoy"] = rg_yoy.transform(lambda s: s.rolling(28, min_periods=14).mean())
    # YoY growth ratio: normalised signal (zero when yoy is absent)
    panel["yoy_ratio"] = np.where(
        panel["roll_mean_28_yoy"].notna(),
        panel["roll_mean_28"] / (panel["roll_mean_28_yoy"] + 1e-6),
        0.0,
    )

    # Days since last sale
    prior = panel["XSRQ"].where(panel["QTY"] > 0)
    prior = prior.groupby(panel["INVENTORY_ITEM_ID"], sort=False).ffill()
    panel["days_since_sale"] = (panel["XSRQ"] - prior).dt.days

    # Revenue-derived: avg revenue per unit over last 28 days
    shifted_rev = g["XSJE"].shift(1)
    rg_rev = shifted_rev.groupby(panel["INVENTORY_ITEM_ID"], sort=False)
    roll_rev_28 = rg_rev.transform(lambda s: s.rolling(28, min_periods=1).sum())
    roll_qty_28 = panel["roll_sum_28"].replace(0, np.nan)
    panel["avg_revenue_per_unit_28"] = roll_rev_28 / roll_qty_28
    panel["avg_revenue_per_unit_28"] = panel["avg_revenue_per_unit_28"].fillna(
        panel["LIST_PRICE_PER_UNIT"]
    )

    # Store count rolling 28d
    shifted_sc = g["store_count"].shift(1)
    rg_sc = shifted_sc.groupby(panel["INVENTORY_ITEM_ID"], sort=False)
    panel["store_count_28"] = rg_sc.transform(lambda s: s.rolling(28, min_periods=1).mean())

    # Calendar features
    cal = panel["XSRQ"].dt
    panel["day_of_month"] = cal.day.astype("int8")
    panel["month"] = cal.month.astype("int8")
    panel["quarter"] = cal.quarter.astype("int8")
    panel["week_of_year"] = _to_float16_safe(cal.isocalendar().week.astype("Int16"))
    panel["is_weekend"] = (cal.dayofweek >= 5).astype("int8")
    panel["item_age_days"] = _to_float16_safe((panel["XSRQ"] - panel["first_sale_date"]).dt.days)

    # --- category-level features (joined from pre-computed category aggregates) ---
    if cat_features_df is not None:
        panel = panel.merge(cat_features_df, on=["ITEM_CATEORY", "XSRQ"], how="left")
        panel["item_share_of_category"] = (
            panel["roll_mean_28"] / (panel["category_roll_mean_28"].fillna(0) + 1e-6)
        )
    else:
        panel["category_roll_mean_28"] = np.nan
        panel["item_share_of_category"] = np.nan
        panel["category_yoy_ratio"] = np.nan

    # --- drop intermediates to free memory ---
    panel.drop(columns=["XSJE", "store_count", "first_sale_date", "QTY"], inplace=True)

    # --- encode categoricals as int16 codes (massive memory saving) ---
    if cat_mappings is not None:
        for col in CATEGORICAL_FEATURES:
            mapping = cat_mappings[col]
            panel[col] = panel[col].map(mapping).fillna(-1).astype("int16")

    # --- quantize: keep higher-range values in float32, small bounded ratios in float16 ---
    float16_cols = [
        "nonzero_days_28",
        "store_count_28",
        # ratios are bounded (typically 0–10), safe for float16
        "velocity_ratio",
        "yoy_ratio",
        "item_share_of_category",
        "category_yoy_ratio",
    ]
    for col in float16_cols:
        panel[col] = _to_float16_safe(panel[col])

    # float32 for cols that can exceed 65504 (sums, lags, revenue, category aggregates)
    float32_cols = [
        "lag_1", "lag_7", "lag_14", "lag_28", "lag_91", "lag_182", "lag_365",
        "roll_sum_7", "roll_sum_14", "roll_sum_28", "roll_sum_91",
        "roll_mean_7", "roll_mean_14", "roll_mean_28", "roll_mean_91",
        "roll_mean_28_yoy",
        "avg_revenue_per_unit_28",
        "category_roll_mean_28",
    ]
    for col in float32_cols:
        panel[col] = panel[col].astype("float32")

    # days_since_sale and LIST_PRICE_PER_UNIT stay float32 (wider range)
    panel["days_since_sale"] = panel["days_since_sale"].astype("float32")
    # DLNUM contains large identifiers (e.g. ~10,010,001), which overflow float16.
    panel["DLNUM"] = pd.to_numeric(panel["DLNUM"], errors="coerce").astype("float32")

    # Keep only what we need
    target_cols = [f"target_{h}d" for h in horizons]
    keep_cols = ["INVENTORY_ITEM_ID", "XSRQ"] + ALL_FEATURES + target_cols
    panel = panel[keep_cols]

    return panel


def _build_cat_mapping(series: pd.Series, max_codes: int | None) -> tuple[dict, int]:
    """Build category→code map, optionally keeping only top-frequency levels."""
    non_null = series.dropna()
    raw_cardinality = int(non_null.nunique())
    if max_codes is not None:
        if max_codes < 1:
            raise ValueError("max_cat_codes must be >= 1 when provided.")
        if raw_cardinality > max_codes:
            kept_values = non_null.value_counts().head(max_codes).index.to_list()
        else:
            kept_values = pd.unique(non_null).tolist()
    else:
        kept_values = pd.unique(non_null).tolist()
    return {v: i for i, v in enumerate(kept_values)}, raw_cardinality


def _build_and_write_chunk_task(task: tuple[int, np.ndarray]) -> tuple[int, int, int]:
    """Worker task for parallel chunk build."""
    import gc

    idx, chunk_items = task
    ctx = _BUILD_WORKER_CTX
    if ctx is None:
        raise RuntimeError("Build worker context not initialized.")

    chunk_panel = _build_chunk_features(
        chunk_items=chunk_items,
        frame=ctx["frame"],  # type: ignore[index]
        all_dates=ctx["all_dates"],  # type: ignore[index]
        item_bounds=ctx["item_bounds"],  # type: ignore[index]
        horizons=ctx["horizons"],  # type: ignore[index]
        cat_mappings=ctx["cat_mappings"],  # type: ignore[index]
        cat_features_df=ctx.get("cat_features_df"),  # type: ignore[index]
    )
    n_rows = len(chunk_panel)
    out_path = Path(ctx["tmp_dir"]) / f"chunk_{idx:04d}.parquet"  # type: ignore[index]
    chunk_panel.to_parquet(out_path, index=False)
    del chunk_panel
    gc.collect()
    return idx, int(len(chunk_items)), int(n_rows)


def build_featured_panel(
    frame: pd.DataFrame,
    eligible_items: np.ndarray,
    panel_days: int,
    horizons: list[int],
    max_cat_codes: int | None = None,
    build_workers: int = 1,
    chunk_size: int = 20_000,
) -> tuple[Path, dict[str, dict[str, int]]]:
    """Build dense panel with features, writing chunks to parquet on disk.

    Returns (chunk_dir, cat_mappings). Categoricals are int16-encoded.
    The full panel is never held in memory at once.
    """
    import gc
    import tempfile

    print("Building featured panel (chunked, fused → parquet)...")
    max_horizon = max(horizons)
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

    # Build categorical mappings: string → int16 code (built once, applied per chunk)
    cap_note = f" [max_cat_codes={max_cat_codes}]" if max_cat_codes is not None else ""
    print(f"Building categorical mappings...{cap_note}")
    cat_mappings: dict[str, dict[str, int]] = {}
    truncated_cardinality: dict[str, tuple[int, int]] = {}
    for col in CATEGORICAL_FEATURES:
        if col in frame.columns:
            mapping, raw_cardinality = _build_cat_mapping(frame[col], max_codes=max_cat_codes)
        else:
            mapping, raw_cardinality = {}, 0
        cat_mappings[col] = mapping
        if raw_cardinality > len(mapping):
            truncated_cardinality[col] = (raw_cardinality, len(mapping))

    print(f"  Categorical cardinalities: { {k: len(v) for k, v in cat_mappings.items()} }")
    if truncated_cardinality:
        print(f"  Truncated categories (raw -> kept): {truncated_cardinality}")

    # --- category-level rolling features (computed once, joined per chunk) ---
    print("Computing category-level rolling features...")
    cat_daily = (
        frame.groupby(["ITEM_CATEORY", "XSRQ"])["QTY"]
        .sum()
        .reset_index()
        .sort_values(["ITEM_CATEORY", "XSRQ"])
    )
    cat_grp = cat_daily.groupby("ITEM_CATEORY", sort=False)
    cat_daily["category_roll_mean_28"] = (
        cat_grp["QTY"]
        .transform(lambda s: s.shift(1).rolling(28, min_periods=14).mean())
        .astype("float32")
    )
    cat_daily["cat_roll_mean_28_yoy"] = (
        cat_grp["QTY"]
        .transform(lambda s: s.shift(365).rolling(28, min_periods=14).mean())
        .astype("float32")
    )
    cat_daily["category_yoy_ratio"] = (
        cat_daily["category_roll_mean_28"] / (cat_daily["cat_roll_mean_28_yoy"] + 1e-6)
    ).astype("float32")
    cat_features_df = cat_daily[
        ["ITEM_CATEORY", "XSRQ", "category_roll_mean_28", "category_yoy_ratio"]
    ].copy()
    del cat_daily
    print(f"  Category features shape: {cat_features_df.shape} [RSS: {_mem_gb()}]")

    if build_workers < 1:
        raise ValueError("build_workers must be >= 1.")

    # Process in chunks — write each to parquet, free memory immediately
    tmp_dir = Path(tempfile.mkdtemp(prefix="book_predict_chunks_"))
    n_chunks = max(1, int(np.ceil(len(eligible_items) / chunk_size)))
    item_chunks = np.array_split(eligible_items, n_chunks)

    total_rows = 0
    if build_workers == 1:
        for i, chunk_items in enumerate(
            _iter_with_progress(item_chunks, total=len(item_chunks), desc="Build chunks")
        ):
            chunk_panel = _build_chunk_features(
                chunk_items, frame, all_dates, item_bounds, horizons,
                cat_mappings=cat_mappings,
                cat_features_df=cat_features_df,
            )
            n_rows = len(chunk_panel)
            total_rows += n_rows
            chunk_panel.to_parquet(tmp_dir / f"chunk_{i:04d}.parquet", index=False)
            del chunk_panel
            gc.collect()
            print(
                f"  Chunk {i + 1}/{len(item_chunks)}: "
                f"{len(chunk_items):,} items, {n_rows:,} rows "
                f"(cumulative: {total_rows:,}) [RSS: {_mem_gb()}]"
            )
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        if os.name != "posix":
            print("  build_workers>1 requires POSIX fork; falling back to build_workers=1.")
            return build_featured_panel(
                frame=frame,
                eligible_items=eligible_items,
                panel_days=panel_days,
                horizons=horizons,
                max_cat_codes=max_cat_codes,
                build_workers=1,
                chunk_size=chunk_size,
            )

        try:
            mp_ctx = mp.get_context("fork")
        except ValueError:
            print("  multiprocessing fork context unavailable; falling back to build_workers=1.")
            return build_featured_panel(
                frame=frame,
                eligible_items=eligible_items,
                panel_days=panel_days,
                horizons=horizons,
                max_cat_codes=max_cat_codes,
                build_workers=1,
                chunk_size=chunk_size,
            )

        print(
            f"Building chunks in parallel with {build_workers} workers "
            "(process-based; higher workers need more RAM)..."
        )
        global _BUILD_WORKER_CTX
        _BUILD_WORKER_CTX = {
            "frame": frame,
            "all_dates": all_dates,
            "item_bounds": item_bounds,
            "horizons": horizons,
            "cat_mappings": cat_mappings,
            "cat_features_df": cat_features_df,
            "tmp_dir": str(tmp_dir),
        }

        try:
            with ProcessPoolExecutor(max_workers=build_workers, mp_context=mp_ctx) as executor:
                futures = [
                    executor.submit(_build_and_write_chunk_task, (i, chunk_items))
                    for i, chunk_items in enumerate(item_chunks)
                ]

                for future in _iter_with_progress(
                    as_completed(futures), total=len(futures), desc="Build chunks (parallel)"
                ):
                    i, item_count, n_rows = future.result()
                    total_rows += n_rows
                    print(
                        f"  Chunk {i + 1}/{len(item_chunks)}: "
                        f"{item_count:,} items, {n_rows:,} rows "
                        f"(cumulative: {total_rows:,}) [RSS: {_mem_gb()}]"
                    )
        finally:
            _BUILD_WORKER_CTX = None

    print(f"  Total rows written: {total_rows:,} across {len(item_chunks)} parquet files")

    # Save categorical mappings for inference
    joblib.dump(cat_mappings, tmp_dir / "cat_mappings.joblib")

    return tmp_dir, cat_mappings


def _required_cols_for_horizon(horizon: int) -> tuple[str, list[str]]:
    target_col = f"target_{horizon}d"
    required_cols = [
        target_col,
        "lag_1", "lag_7", "lag_14", "lag_28",
        "roll_sum_7", "roll_sum_14", "roll_sum_28",
        "days_since_sale",
    ]
    return target_col, required_cols


def _load_cols_for_horizon(horizon: int, include_ids: bool = True) -> list[str]:
    target_col, _ = _required_cols_for_horizon(horizon)
    cols = ["XSRQ"] + (["INVENTORY_ITEM_ID"] if include_ids else []) + ALL_FEATURES + [target_col]
    return cols


def _iter_filtered_chunks(
    chunk_dir: Path,
    horizon: int,
    *,
    include_ids: bool,
    desc: str,
):
    target_col, required_cols = _required_cols_for_horizon(horizon)
    load_cols = _load_cols_for_horizon(horizon, include_ids=include_ids)
    chunk_files = sorted(chunk_dir.glob("chunk_*.parquet"))

    total_kept = 0
    for i, f in enumerate(
        _iter_with_progress(chunk_files, total=len(chunk_files), desc=desc)
    ):
        chunk = pd.read_parquet(f, columns=load_cols)
        chunk = chunk.dropna(subset=required_cols)
        total_kept += len(chunk)
        print(
            f"  Read chunk {i + 1}/{len(chunk_files)}: "
            f"kept {len(chunk):,} rows "
            f"(cumulative: {total_kept:,}) [RSS: {_mem_gb()}]"
        )
        yield chunk


def _determine_split_dates(chunk_dir: Path, horizon: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    unique_dates: set[pd.Timestamp] = set()
    for chunk in _iter_filtered_chunks(
        chunk_dir, horizon, include_ids=False, desc=f"Dates {horizon}d"
    ):
        unique_dates.update(pd.to_datetime(chunk["XSRQ"].unique()).tolist())

    ordered_dates = np.sort(np.array(list(unique_dates), dtype="datetime64[ns]"))
    if len(ordered_dates) < 120:
        raise ValueError("Not enough history to create reliable time-based splits.")

    train_end = pd.Timestamp(ordered_dates[int(len(ordered_dates) * 0.70)])
    valid_end = pd.Timestamp(ordered_dates[int(len(ordered_dates) * 0.85)])
    print(
        f"  Split dates for {horizon}d: "
        f"train<= {train_end.date()}  valid<= {valid_end.date()}  "
        f"({len(ordered_dates)} usable dates)"
    )
    return train_end, valid_end


def _count_split_rows(
    chunk_dir: Path,
    horizon: int,
    train_end: pd.Timestamp,
    valid_end: pd.Timestamp,
) -> tuple[int, int, int]:
    train_rows = 0
    valid_rows = 0
    test_rows = 0
    for chunk in _iter_filtered_chunks(
        chunk_dir, horizon, include_ids=False, desc=f"Count {horizon}d"
    ):
        train_rows += int((chunk["XSRQ"] <= train_end).sum())
        valid_rows += int(((chunk["XSRQ"] > train_end) & (chunk["XSRQ"] <= valid_end)).sum())
        test_rows += int((chunk["XSRQ"] > valid_end).sum())

    print(
        f"  Split row counts for {horizon}d: "
        f"train={train_rows:,} valid={valid_rows:,} test={test_rows:,}"
    )
    return train_rows, valid_rows, test_rows


def _write_train_valid_files(
    chunk_dir: Path,
    horizon: int,
    train_end: pd.Timestamp,
    valid_end: pd.Timestamp,
    staging_dir: Path,
    *,
    max_train_rows: int | None,
    recent_days: int,
    random_state: int,
) -> tuple[Path, Path, int, int, int]:
    import gc

    rng = np.random.default_rng(random_state)
    target_col, _ = _required_cols_for_horizon(horizon)
    train_path = staging_dir / f"train_{horizon}d.csv"
    valid_path = staging_dir / f"valid_{horizon}d.csv"

    recent_cutoff = train_end - pd.Timedelta(days=recent_days)
    recent_keep_prob = 1.0
    old_keep_prob = 1.0

    if max_train_rows is not None:
        # Count recent vs old train rows in one pass
        recent_count = 0
        old_count = 0
        for chunk in _iter_filtered_chunks(
            chunk_dir, horizon, include_ids=False, desc=f"Count {horizon}d"
        ):
            train_mask = chunk["XSRQ"] <= train_end
            recent_count += int((train_mask & (chunk["XSRQ"] > recent_cutoff)).sum())
            old_count += int((train_mask & (chunk["XSRQ"] <= recent_cutoff)).sum())

        total_count = recent_count + old_count
        if total_count <= max_train_rows:
            print(f"  Training row cap for {horizon}d: {total_count:,} <= {max_train_rows:,}, keeping all")
        elif recent_count >= max_train_rows:
            # Recent data alone exceeds cap — sample uniformly from recent, drop all old
            recent_keep_prob = max_train_rows / recent_count
            old_keep_prob = 0.0
            print(
                f"  [temporal sampling {horizon}d] recent({recent_days}d)={recent_count:,} > cap={max_train_rows:,}; "
                f"sample recent at {recent_keep_prob:.4f}, drop old"
            )
        else:
            # Keep all recent, fill remaining budget from old
            old_keep_prob = (max_train_rows - recent_count) / max(old_count, 1)
            print(
                f"  [temporal sampling {horizon}d] keep all recent({recent_days}d)={recent_count:,}, "
                f"sample old={old_count:,} at {old_keep_prob:.4f} "
                f"(total target ~{max_train_rows:,})"
            )

    written_train = 0
    written_valid = 0
    seen_test = 0
    wrote_train_header = False
    wrote_valid_header = False
    export_cols = [target_col] + ALL_FEATURES

    for chunk in _iter_filtered_chunks(
        chunk_dir, horizon, include_ids=False, desc=f"Stage {horizon}d"
    ):
        train_chunk = chunk[chunk["XSRQ"] <= train_end]
        if len(train_chunk) > 0 and (recent_keep_prob < 1.0 or old_keep_prob < 1.0):
            is_recent = train_chunk["XSRQ"] > recent_cutoff
            probs = np.where(is_recent, recent_keep_prob, old_keep_prob)
            mask = rng.random(len(train_chunk)) < probs
            train_chunk = train_chunk.loc[mask]
        valid_chunk = chunk[(chunk["XSRQ"] > train_end) & (chunk["XSRQ"] <= valid_end)]
        seen_test += int((chunk["XSRQ"] > valid_end).sum())

        if len(train_chunk) > 0:
            train_chunk[export_cols].to_csv(
                train_path, mode="a", header=not wrote_train_header, index=False
            )
            wrote_train_header = True
            written_train += len(train_chunk)
        if len(valid_chunk) > 0:
            valid_chunk[export_cols].to_csv(
                valid_path, mode="a", header=not wrote_valid_header, index=False
            )
            wrote_valid_header = True
            written_valid += len(valid_chunk)

        del chunk, train_chunk, valid_chunk
        gc.collect()

    print(
        f"  Staged {horizon}d files: "
        f"train={written_train:,} valid={written_valid:,} test={seen_test:,}"
    )
    return train_path, valid_path, written_train, written_valid, seen_test


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


def _empty_metric_accumulator() -> dict[str, float]:
    return {
        "count": 0.0,
        "abs_err_sum": 0.0,
        "sq_err_sum": 0.0,
        "abs_true_sum": 0.0,
        "baseline_abs_err_sum": 0.0,
    }


def _update_metric_accumulator(
    acc: dict[str, float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline: np.ndarray,
) -> None:
    acc["count"] += float(len(y_true))
    acc["abs_err_sum"] += float(np.abs(y_true - y_pred).sum())
    acc["sq_err_sum"] += float(np.square(y_true - y_pred).sum())
    acc["abs_true_sum"] += float(np.abs(y_true).sum())
    acc["baseline_abs_err_sum"] += float(np.abs(y_true - baseline).sum())


def _finalize_metric_accumulator(acc: dict[str, float]) -> dict[str, float]:
    count = max(acc["count"], 1.0)
    abs_true_sum = max(acc["abs_true_sum"], 1e-9)
    return {
        "mae": acc["abs_err_sum"] / count,
        "rmse": float(np.sqrt(acc["sq_err_sum"] / count)),
        "wape": acc["abs_err_sum"] / abs_true_sum,
        "baseline_mae": acc["baseline_abs_err_sum"] / count,
        "baseline_wape": acc["baseline_abs_err_sum"] / abs_true_sum,
    }


# ---------------------------------------------------------------------------
# LightGBM model
# ---------------------------------------------------------------------------

def build_lgbm_params(config: LGBMTrainerConfig, device: str) -> dict:
    effective_n_jobs = _resolve_n_jobs(config.n_jobs)
    params = {
        "objective": "regression_l1",      # MAE — robust to QTY outliers
        "metric": "mae",
        "device": device,
        "learning_rate": 0.03,
        "num_leaves": 511,
        "min_child_samples": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "seed": config.random_state,
        "n_jobs": effective_n_jobs,
    }
    if config.max_bin is not None:
        params["max_bin"] = int(config.max_bin)
    if config.max_cat_threshold is not None:
        params["max_cat_threshold"] = int(config.max_cat_threshold)
    if config.gpu_safe and device in {"gpu", "cuda"}:
        params.setdefault("max_bin", 255)
        params.setdefault("max_cat_threshold", 64)
    if config.gpu_device_id is not None and device in {"gpu", "cuda"}:
        params["gpu_device_id"] = config.gpu_device_id
    return params


def _train_booster_with_fallback(
    dtrain: lgb.Dataset,
    dvalid: lgb.Dataset,
    config: LGBMTrainerConfig,
    callbacks: list,
) -> tuple[lgb.Booster, float, str]:
    """Train with backend fallback: cuda -> gpu -> cpu, gpu -> cpu, cpu only."""
    requested_device = config.device
    if requested_device == "cuda":
        attempt_order = ["cuda", "gpu", "cpu"]
    elif requested_device == "gpu":
        attempt_order = ["gpu", "cpu"]
    else:
        attempt_order = ["cpu"]

    last_error: Exception | None = None
    for i, device in enumerate(attempt_order):
        params = build_lgbm_params(config, device)
        try:
            train_started = time.perf_counter()
            booster = lgb.train(
                params,
                dtrain,
                num_boost_round=2000,
                valid_sets=[dvalid],
                callbacks=callbacks,
            )
            train_elapsed = time.perf_counter() - train_started
            return booster, train_elapsed, device
        except lgb.basic.LightGBMError as exc:
            last_error = exc
            is_last_attempt = i == len(attempt_order) - 1
            if is_last_attempt:
                raise
            next_device = attempt_order[i + 1]
            print(f"  {device.upper()} training failed: {exc}")
            print(f"  Falling back to {next_device.upper()} for this horizon.")

    assert last_error is not None
    raise last_error


def _evaluate_streaming_splits(
    chunk_dir: Path,
    horizon: int,
    booster: lgb.Booster,
    output_dir: Path,
    train_end: pd.Timestamp,
    valid_end: pd.Timestamp,
) -> tuple[dict[str, float], dict[str, float], int, int]:
    target_col, _ = _required_cols_for_horizon(horizon)
    valid_acc = _empty_metric_accumulator()
    test_acc = _empty_metric_accumulator()
    valid_rows = 0
    test_rows = 0
    predictions_path = output_dir / "test_predictions.csv"
    if predictions_path.exists():
        predictions_path.unlink()
    wrote_predictions_header = False

    for chunk in _iter_filtered_chunks(
        chunk_dir, horizon, include_ids=True, desc=f"Eval {horizon}d"
    ):
        valid_chunk = chunk[(chunk["XSRQ"] > train_end) & (chunk["XSRQ"] <= valid_end)]
        test_chunk = chunk[chunk["XSRQ"] > valid_end]

        if len(valid_chunk) > 0:
            valid_pred = booster.predict(valid_chunk[ALL_FEATURES])
            valid_y = valid_chunk[target_col].to_numpy()
            valid_baseline = valid_chunk["roll_mean_28"].to_numpy().astype("float32") * horizon
            _update_metric_accumulator(valid_acc, valid_y, valid_pred, valid_baseline)
            valid_rows += len(valid_chunk)

        if len(test_chunk) > 0:
            test_pred = booster.predict(test_chunk[ALL_FEATURES])
            test_y = test_chunk[target_col].to_numpy()
            test_baseline = test_chunk["roll_mean_28"].to_numpy().astype("float32") * horizon
            _update_metric_accumulator(test_acc, test_y, test_pred, test_baseline)
            test_rows += len(test_chunk)

            predictions = test_chunk[["INVENTORY_ITEM_ID", "XSRQ", target_col]].copy()
            predictions["prediction"] = test_pred
            predictions["baseline_prediction"] = test_baseline
            predictions.to_csv(
                predictions_path,
                mode="a",
                header=not wrote_predictions_header,
                index=False,
                encoding="utf-8-sig",
            )
            wrote_predictions_header = True

    return (
        _finalize_metric_accumulator(valid_acc),
        _finalize_metric_accumulator(test_acc),
        valid_rows,
        test_rows,
    )


# ---------------------------------------------------------------------------
# Per-horizon training
# ---------------------------------------------------------------------------

def train_for_horizon(
    chunk_dir: Path, config: LGBMTrainerConfig, horizon: int,
    cat_mappings: dict[str, dict[str, int]] | None = None,
) -> dict[str, object]:
    """Train a LightGBM model for a single horizon.

    Trains from staged on-disk files to avoid concatenating all rows in RAM.
    """
    import gc
    import shutil
    import tempfile

    print(f"\n--- Horizon {horizon}d ---")
    target_col = f"target_{horizon}d"
    train_end, valid_end = _determine_split_dates(chunk_dir, horizon)

    staging_dir = Path(tempfile.mkdtemp(prefix=f"book_predict_stage_{horizon}d_"))
    train_path, valid_path, train_rows, _, _ = _write_train_valid_files(
        chunk_dir,
        horizon,
        train_end,
        valid_end,
        staging_dir,
        max_train_rows=config.max_train_rows,
        recent_days=config.recent_days,
        random_state=config.random_state,
    )

    # Tell LightGBM which columns are categorical (they're int16-encoded)
    cat_feature_indices = [ALL_FEATURES.index(c) for c in CATEGORICAL_FEATURES]
    dtrain = lgb.Dataset(
        str(train_path),
        params={"header": True, "label_column": 0},
        feature_name=ALL_FEATURES,
        categorical_feature=cat_feature_indices,
        free_raw_data=True,
    )
    dvalid = lgb.Dataset(
        str(valid_path),
        params={"header": True, "label_column": 0},
        feature_name=ALL_FEATURES,
        categorical_feature=cat_feature_indices,
        reference=dtrain,
        free_raw_data=True,
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=50),
    ]

    booster, train_elapsed, actual_device = _train_booster_with_fallback(
        dtrain, dvalid, config, callbacks
    )

    # Save
    output_dir = config.output_dir / f"horizon_{horizon}d"
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_metrics, test_metrics, valid_rows, test_rows = _evaluate_streaming_splits(
        chunk_dir, horizon, booster, output_dir, train_end, valid_end
    )

    booster.save_model(str(output_dir / "model.lgb"))
    joblib.dump({
        "booster_path": str(output_dir / "model.lgb"),
        "features": ALL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "cat_mappings": cat_mappings,
    }, output_dir / "model_meta.joblib")

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
    print(f"  train  elapsed={train_elapsed:.1f}s  device={actual_device}")
    shutil.rmtree(staging_dir, ignore_errors=True)
    gc.collect()

    return {
        "horizon": horizon,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "train_rows": int(train_rows),
        "valid_rows": int(valid_rows),
        "test_rows": int(test_rows),
        "best_iteration": int(booster.best_iteration),
        "device_used": actual_device,
        "train_elapsed_sec": float(train_elapsed),
        "output_dir": output_dir,
    }


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def run_training(config: LGBMTrainerConfig) -> list[dict[str, object]]:
    import gc

    effective_max_cat_codes = config.max_cat_codes
    if config.gpu_safe and config.device in {"gpu", "cuda"} and effective_max_cat_codes is None:
        effective_max_cat_codes = 255

    visible_cpus = os.cpu_count() or 1
    effective_n_jobs = _resolve_n_jobs(config.n_jobs)
    if config.n_jobs < -1:
        thread_note = f"interpreted as {effective_n_jobs} (abs of n_jobs)"
    elif config.n_jobs in (-1, 0):
        thread_note = "all visible logical cores"
    else:
        thread_note = str(effective_n_jobs)
    print(
        f"Threading: n_jobs={config.n_jobs} "
        f"(visible_cpus={visible_cpus}, effective={thread_note}) "
        f"build_workers={config.build_workers}"
    )
    if config.device in {"gpu", "cuda"}:
        effective_max_bin = config.max_bin if config.max_bin is not None else (255 if config.gpu_safe else "default")
        effective_max_cat_threshold = (
            config.max_cat_threshold
            if config.max_cat_threshold is not None
            else (64 if config.gpu_safe else "default")
        )
        print(
            f"{config.device.upper()} binning: "
            f"max_bin={effective_max_bin}, "
            f"max_cat_threshold={effective_max_cat_threshold}, "
            f"max_cat_codes={effective_max_cat_codes if effective_max_cat_codes is not None else 'unlimited'}"
        )

    sales = load_and_aggregate_sales(config)
    eligible_items = select_eligible_items(
        sales,
        min_history_days=config.min_history_days,
        max_items=config.max_items,
    )
    print(f"Eligible items: {len(eligible_items):,} [RSS: {_mem_gb()}]")
    if len(eligible_items) == 0:
        raise ValueError("No items satisfied the minimum history threshold.")

    chunk_dir, cat_mappings = build_featured_panel(
        sales,
        eligible_items=eligible_items,
        panel_days=config.panel_days,
        horizons=config.horizons,
        max_cat_codes=effective_max_cat_codes,
        build_workers=config.build_workers,
    )
    print(f"Chunk parquet directory: {chunk_dir} [RSS: {_mem_gb()}]")
    del sales
    gc.collect()
    print(f"Freed sales data [RSS: {_mem_gb()}]")

    use_multi_gpu = (
        config.device in {"gpu", "cuda"}
        and len(config.horizons) > 1
        and config.gpu_device_id is None   # manual override disables auto-assign
    )

    if use_multi_gpu:
        import dataclasses
        from concurrent.futures import ThreadPoolExecutor, as_completed

        print(f"Multi-GPU mode: distributing {len(config.horizons)} horizons across GPUs 0–{len(config.horizons)-1}")

        def _train_on_gpu(horizon: int, gpu_id: int) -> dict[str, object]:
            h_config = dataclasses.replace(config, gpu_device_id=gpu_id)
            result = train_for_horizon(chunk_dir, h_config, horizon, cat_mappings=cat_mappings)
            print(f"Finished horizon {horizon}d on GPU {gpu_id} [RSS: {_mem_gb()}]")
            return result

        with ThreadPoolExecutor(max_workers=len(config.horizons)) as executor:
            futures = {
                executor.submit(_train_on_gpu, horizon, gpu_id): horizon
                for gpu_id, horizon in enumerate(config.horizons)
            }
            horizon_results = {}
            for future in as_completed(futures):
                result = future.result()
                horizon_results[result["horizon"]] = result
        results = [horizon_results[h] for h in config.horizons]
    else:
        results = []
        for horizon in config.horizons:
            results.append(train_for_horizon(chunk_dir, config, horizon, cat_mappings=cat_mappings))
            gc.collect()
            print(f"Finished horizon {horizon}d [RSS: {_mem_gb()}]")

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

    # Clean up temp parquet files
    import shutil
    shutil.rmtree(chunk_dir, ignore_errors=True)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(
        config.output_dir / "training_summary.csv", index=False, encoding="utf-8-sig"
    )
    return results
