"""Diagnostic CLI — Plan B step 1.

Computes item segments from the on-disk parquet panel chunks and reports:
  1. Population: how many items / rows fall into each segment.
  2. Per-segment WAPE/MAE for the existing 15d and 30d models.
  3. Saves segment labels to artifacts_lgbm/item_segments.csv for later reuse.

This script does NOT retrain. Run it after a normal `train_lgbm.py` run.
Goal: tell us whether sparse/seasonal segments are the bottleneck — i.e.
whether Plan B is worth doing as a feature addition next.

Usage:
    python3 diagnose_segments.py \
        --chunks-dir artifacts_lgbm/horizon_30d/chunks \
        --model artifacts_lgbm/horizon_30d/model.lgb \
        --horizon 30 \
        --output artifacts_lgbm/segment_report_30d.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.book_predict.segmentation import (
    SegmentationConfig,
    compute_item_segments,
    enrich_panel_with_segments,
    per_segment_metrics,
    SEGMENT_LABELS,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--chunks-dir", type=Path, required=True,
                   help="Directory of parquet chunks produced during training.")
    p.add_argument("--model", type=Path, required=True,
                   help="Path to trained LightGBM model (.lgb).")
    p.add_argument("--horizon", type=int, required=True, choices=[15, 30])
    p.add_argument("--test-start", type=str, default=None,
                   help="ISO date — only rows with XSRQ >= this are used as test. "
                        "If omitted, uses the last 20% of dates.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--segment-labels-out", type=Path,
                   default=Path("artifacts_lgbm/item_segments.csv"))
    p.add_argument("--max-rows-per-chunk", type=int, default=None,
                   help="Optional cap (debug only).")
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="LightGBM prediction threads (-1 = all cores).")
    return p.parse_args()


def _gather_panel_for_segmentation(chunks_dir: Path) -> pd.DataFrame:
    """Read minimal cols from every chunk: ITEM_ID, XSRQ, lag_1 (proxy for QTY)."""
    files = sorted(chunks_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet chunks under {chunks_dir}")
    cols = ["INVENTORY_ITEM_ID", "XSRQ", "lag_1"]
    parts = []
    for f in files:
        parts.append(pd.read_parquet(f, columns=cols))
    return pd.concat(parts, ignore_index=True)


def _determine_test_cutoff(chunks_dir: Path, override: str | None) -> pd.Timestamp:
    if override is not None:
        return pd.Timestamp(override)
    # Fall back: take the last 20% of the panel's date range
    files = sorted(chunks_dir.glob("*.parquet"))
    dates = []
    for f in files:
        d = pd.read_parquet(f, columns=["XSRQ"])
        dates.append((d["XSRQ"].min(), d["XSRQ"].max()))
    lo = min(x[0] for x in dates)
    hi = max(x[1] for x in dates)
    span = (hi - lo).days
    return lo + pd.Timedelta(days=int(span * 0.8))


def main() -> None:
    args = parse_args()
    target_col = f"target_{args.horizon}d"

    print(f"[1/4] Loading panel for segmentation from {args.chunks_dir} ...")
    panel = _gather_panel_for_segmentation(args.chunks_dir)
    print(f"      panel rows: {len(panel):,}, items: {panel['INVENTORY_ITEM_ID'].nunique():,}")

    print("[2/4] Computing item segments ...")
    segments = compute_item_segments(panel, config=SegmentationConfig())
    args.segment_labels_out.parent.mkdir(parents=True, exist_ok=True)
    segments.to_csv(args.segment_labels_out, index=False, encoding="utf-8-sig")
    pop = segments["item_segment"].value_counts().reindex(SEGMENT_LABELS, fill_value=0)
    print("      segment populations (items):")
    for label, count in pop.items():
        print(f"        {label:>8}: {count:>10,}")
    del panel

    print("[3/4] Loading model and scoring test chunks ...")
    booster = lgb.Booster(model_file=str(args.model))
    booster.params["num_threads"] = args.n_jobs
    feature_names = booster.feature_name()

    test_cutoff = _determine_test_cutoff(args.chunks_dir, args.test_start)
    print(f"      test cutoff date: {test_cutoff.date()}")

    seg_lookup = segments.set_index("INVENTORY_ITEM_ID")["item_segment"]
    seg_codes = pd.Categorical(
        segments["item_segment"], categories=SEGMENT_LABELS
    ).codes
    item_to_code = dict(zip(segments["INVENTORY_ITEM_ID"], seg_codes))

    files = sorted(args.chunks_dir.glob("*.parquet"))
    needed = list(dict.fromkeys(["INVENTORY_ITEM_ID", "XSRQ", target_col, *feature_names]))

    rows_seen = 0
    metric_acc = {label: {"n": 0, "abs_err": 0.0, "abs_y": 0.0} for label in SEGMENT_LABELS}

    for i, f in enumerate(files, 1):
        df = pd.read_parquet(f, columns=needed)
        df = df[df["XSRQ"] >= test_cutoff]
        df = df.dropna(subset=[target_col])
        if args.max_rows_per_chunk is not None:
            df = df.head(args.max_rows_per_chunk)
        if df.empty:
            continue

        X = df[feature_names]
        y = df[target_col].to_numpy()
        p = booster.predict(X, num_iteration=booster.best_iteration)
        p = np.clip(p, 0, None)

        codes = df["INVENTORY_ITEM_ID"].map(item_to_code).fillna(
            SEGMENT_LABELS.index("cold")
        ).to_numpy()

        for code, label in enumerate(SEGMENT_LABELS):
            mask = codes == code
            if not mask.any():
                continue
            metric_acc[label]["n"] += int(mask.sum())
            metric_acc[label]["abs_err"] += float(np.abs(y[mask] - p[mask]).sum())
            metric_acc[label]["abs_y"] += float(np.abs(y[mask]).sum())

        rows_seen += len(df)
        print(f"      chunk {i}/{len(files)}  test rows so far: {rows_seen:,}")

    print("[4/4] Aggregating per-segment metrics ...")
    out_rows = []
    for label in SEGMENT_LABELS:
        acc = metric_acc[label]
        if acc["n"] == 0:
            continue
        wape = acc["abs_err"] / acc["abs_y"] if acc["abs_y"] > 0 else float("nan")
        mae = acc["abs_err"] / acc["n"]
        out_rows.append({
            "segment": label,
            "n_rows": acc["n"],
            "abs_y_sum": acc["abs_y"],
            "mae": mae,
            "wape": wape,
        })
    report = pd.DataFrame(out_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False, encoding="utf-8-sig")
    print()
    print(report.to_string(index=False))
    print()
    print(f"Saved per-segment metrics to {args.output}")
    print(f"Saved item segment labels to {args.segment_labels_out}")


if __name__ == "__main__":
    main()
