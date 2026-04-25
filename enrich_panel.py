"""Phase-1 panel enrichment: add holiday + price-discount + store-demand
features to existing parquet chunks, write augmented chunks to a new dir.

Designed to be NON-DESTRUCTIVE: reads from --in-dir, writes to --out-dir.
Run train_segmented.py against --out-dir to A/B against the original chunks.

Usage:
    conda run -n book_predict python -u enrich_panel.py \\
        --in-dir  artifacts_lgbm/.../chunks \\
        --out-dir artifacts_lgbm_enriched/horizon_30d/chunks \\
        --io-workers 4

Added columns:
    holiday_type            int16 categorical: 0=none, 1=spring_festival,
                            2=qingming, 3=labor, 4=dragon_boat, 5=mid_autumn,
                            6=national, 7=new_year
    days_to_holiday         float16, clipped to 30
    days_since_holiday      float16, clipped to 30
    discount_ratio_28       float16, avg_revenue_per_unit_28 / LIST_PRICE_PER_UNIT
                            (clipped to [0, 2]); 1.0 means no discount
    store_demand_lag1_mean  float32, mean lag_1 across items sharing primary_store
                            on the same XSRQ (cross-item demand signal)

Phase 2 (not in this file): stockout-corrected rolling means.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Hardcoded mainland China public holidays 2022-2027 (observed/adjusted dates).
# Verify against your panel date range; extend if needed.
# ---------------------------------------------------------------------------
HOLIDAY_RANGES: dict[int, list[tuple[str, str]]] = {
    # type 1: Spring Festival
    1: [("2022-01-31", "2022-02-06"), ("2023-01-21", "2023-01-27"),
        ("2024-02-10", "2024-02-17"), ("2025-01-28", "2025-02-04"),
        ("2026-02-16", "2026-02-24"), ("2027-02-06", "2027-02-12")],
    # type 2: Qingming
    2: [("2022-04-03", "2022-04-05"), ("2023-04-05", "2023-04-05"),
        ("2024-04-04", "2024-04-06"), ("2025-04-04", "2025-04-06"),
        ("2026-04-05", "2026-04-06"), ("2027-04-05", "2027-04-05")],
    # type 3: Labor Day
    3: [("2022-04-30", "2022-05-04"), ("2023-04-29", "2023-05-03"),
        ("2024-05-01", "2024-05-05"), ("2025-05-01", "2025-05-05"),
        ("2026-05-01", "2026-05-05"), ("2027-05-01", "2027-05-05")],
    # type 4: Dragon Boat
    4: [("2022-06-03", "2022-06-05"), ("2023-06-22", "2023-06-24"),
        ("2024-06-08", "2024-06-10"), ("2025-05-31", "2025-06-02"),
        ("2026-06-19", "2026-06-21"), ("2027-06-09", "2027-06-11")],
    # type 5: Mid-Autumn
    5: [("2022-09-10", "2022-09-12"), ("2023-09-29", "2023-09-29"),
        ("2024-09-15", "2024-09-17"), ("2026-09-25", "2026-09-27"),
        ("2027-09-15", "2027-09-17")],
    # type 6: National Day (incl. combined Mid-Autumn for 2023, 2025)
    6: [("2022-10-01", "2022-10-07"), ("2023-09-29", "2023-10-06"),
        ("2024-10-01", "2024-10-07"), ("2025-10-01", "2025-10-08"),
        ("2026-10-01", "2026-10-07"), ("2027-10-01", "2027-10-07")],
    # type 7: New Year's Day
    7: [(f"{y}-01-01", f"{y}-01-03") for y in range(2022, 2028)],
}


def _build_holiday_lookup(min_date: pd.Timestamp,
                          max_date: pd.Timestamp) -> pd.DataFrame:
    """Return a DataFrame with one row per date in [min_date, max_date]:
    columns XSRQ, holiday_type (int16), days_to_holiday, days_since_holiday."""
    dates = pd.date_range(min_date, max_date, freq="D")
    htype = np.zeros(len(dates), dtype=np.int16)
    for type_id, ranges in HOLIDAY_RANGES.items():
        for start, end in ranges:
            mask = (dates >= start) & (dates <= end)
            htype[mask] = type_id

    is_h = htype > 0
    n = len(dates)
    days_since = np.full(n, 99, dtype=np.int32)
    days_to = np.full(n, 99, dtype=np.int32)

    last = -10_000
    for i in range(n):
        if is_h[i]:
            last = i
        days_since[i] = i - last
    nxt = 10_000
    for i in range(n - 1, -1, -1):
        if is_h[i]:
            nxt = i
        days_to[i] = nxt - i

    return pd.DataFrame({
        "XSRQ": dates,
        "holiday_type": htype,
        "days_to_holiday": np.clip(days_to, 0, 30).astype(np.float16),
        "days_since_holiday": np.clip(days_since, 0, 30).astype(np.float16),
    })


# ---------------------------------------------------------------------------
# Store-level demand pre-pass: aggregate lag_1 by (primary_store, XSRQ).
# Streams chunks once with minimal columns to keep RSS low.
# ---------------------------------------------------------------------------

def _compute_store_demand(in_dir: Path, io_workers: int) -> pd.DataFrame:
    files = sorted(in_dir.glob("*.parquet"))
    cols = ["primary_store", "XSRQ", "lag_1"]

    def _read(f: Path) -> pd.DataFrame:
        df = pd.read_parquet(f, columns=cols)
        return (df.groupby(["primary_store", "XSRQ"], observed=True)["lag_1"]
                  .agg(["sum", "count"]).reset_index())

    if io_workers > 1 and len(files) > 1:
        with ThreadPoolExecutor(max_workers=io_workers) as ex:
            parts = list(ex.map(_read, files))
    else:
        parts = [_read(f) for f in files]

    agg = (pd.concat(parts, ignore_index=True)
             .groupby(["primary_store", "XSRQ"], observed=True)
             [["sum", "count"]].sum().reset_index())
    agg["store_demand_lag1_mean"] = (
        agg["sum"] / agg["count"].clip(lower=1)
    ).astype(np.float32)
    return agg[["primary_store", "XSRQ", "store_demand_lag1_mean"]]


# ---------------------------------------------------------------------------
# Per-chunk enrichment
# ---------------------------------------------------------------------------

def _enrich_chunk(df: pd.DataFrame,
                  holiday_df: pd.DataFrame,
                  store_df: pd.DataFrame | None) -> pd.DataFrame:
    df = df.merge(holiday_df, on="XSRQ", how="left")

    if "avg_revenue_per_unit_28" in df.columns and "LIST_PRICE_PER_UNIT" in df.columns:
        list_price = df["LIST_PRICE_PER_UNIT"].astype("float32")
        eff_price = df["avg_revenue_per_unit_28"].astype("float32")
        ratio = eff_price / list_price.where(list_price > 0)
        ratio = ratio.fillna(1.0).clip(0.0, 2.0)
        df["discount_ratio_28"] = ratio.astype(np.float16)

    if store_df is not None and "primary_store" in df.columns:
        df = df.merge(store_df, on=["primary_store", "XSRQ"], how="left")
        df["store_demand_lag1_mean"] = (
            df["store_demand_lag1_mean"].fillna(0.0).astype(np.float32)
        )

    df["holiday_type"] = df["holiday_type"].fillna(0).astype(np.int16)
    df["days_to_holiday"] = df["days_to_holiday"].fillna(np.float16(30))
    df["days_since_holiday"] = df["days_since_holiday"].fillna(np.float16(30))
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--io-workers", type=int, default=1)
    p.add_argument("--skip-store-demand", action="store_true",
                   help="Skip the store-level cross-item aggregation pass.")
    args = p.parse_args()

    files = sorted(args.in_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No parquet chunks found under {args.in_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Scanning date range across {len(files)} chunks ...")
    def _read_dates(f: Path) -> pd.DataFrame:
        return pd.read_parquet(f, columns=["XSRQ"])
    if args.io_workers > 1:
        with ThreadPoolExecutor(max_workers=args.io_workers) as ex:
            ds = list(ex.map(_read_dates, files))
    else:
        ds = [_read_dates(f) for f in files]
    lo = min(d["XSRQ"].min() for d in ds if not d.empty)
    hi = max(d["XSRQ"].max() for d in ds if not d.empty)
    print(f"      panel dates: {lo.date()} → {hi.date()}")
    holiday_df = _build_holiday_lookup(lo, hi)

    store_df = None
    if not args.skip_store_demand:
        print("[2/3] Computing per-(store, day) demand aggregate ...")
        store_df = _compute_store_demand(args.in_dir, args.io_workers)
        print(f"      {len(store_df):,} (store, day) rows")
    else:
        print("[2/3] Skipping store-demand pass (--skip-store-demand).")

    print(f"[3/3] Writing enriched chunks to {args.out_dir} ...")
    for i, f in enumerate(files, 1):
        df = pd.read_parquet(f)
        before = df.shape
        df = _enrich_chunk(df, holiday_df, store_df)
        df.to_parquet(args.out_dir / f.name, index=False)
        print(f"      chunk {i:>3}/{len(files)}  {before} -> {df.shape}")

    print(f"\nDone. Run training with --chunks-dir {args.out_dir}")


if __name__ == "__main__":
    main()
