"""Plan B — segment-conditional LightGBM training.

Trains a separate LightGBM booster per item segment (regular / medium /
seasonal / sparse / cold) with per-segment objectives:

    regular, medium, seasonal -> regression_l1  (current main pipeline)
    sparse, cold              -> tweedie         (zero-inflated low-volume)

At inference, every test row is routed to its segment's booster and
metrics are aggregated globally + per segment.

PRE-REQ: a previous run of train_lgbm.py with KEEP_CHUNKS=1 must have
produced parquet panel chunks under <main-output>/horizon_{H}d/chunks/
(or any directory you pass as --chunks-dir).

Usage:
    conda run -n book_predict python -u train_segmented.py \\
        --chunks-dir artifacts_lgbm/horizon_30d/chunks \\
        --horizon 30 \\
        --output-dir artifacts_lgbm_segmented/horizon_30d \\
        --device gpu --gpu-safe
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.book_predict.segmentation import (
    SEGMENT_LABELS,
    SegmentationConfig,
    compute_item_segments,
)


# Per-segment LightGBM hyperparameters. Anything missing falls back to BASE.
BASE_PARAMS = {
    "objective": "regression_l1",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 255,
    "min_child_samples": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "verbose": -1,
}

SEGMENT_PARAMS: dict[str, dict] = {
    "regular":  {},                                       # base
    "medium":   {},                                       # base
    "seasonal": {"num_leaves": 511, "min_child_samples": 100},
    "sparse":   {"objective": "tweedie", "tweedie_variance_power": 1.3, "metric": "rmse"},
    "cold":     {"objective": "tweedie", "tweedie_variance_power": 1.5,
                 "metric": "rmse", "num_leaves": 63, "min_child_samples": 50},
}

SEGMENT_NUM_BOOST = {
    "regular": 1000, "medium": 1000, "seasonal": 1500,
    "sparse": 800, "cold": 400,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--chunks-dir", type=Path, required=True)
    p.add_argument("--horizon", type=int, required=True, choices=[15, 30])
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    p.add_argument("--gpu-safe", action="store_true",
                   help="Cap max_bin/cat thresholds for GPU stability.")
    p.add_argument("--early-stopping", type=int, default=50)
    p.add_argument("--valid-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--max-rows-per-segment", type=int, default=None,
                   help="Optional row cap per segment for memory.")
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="LightGBM threads per booster (-1 = all cores).")
    p.add_argument("--parallel-segments", type=int, default=1,
                   help="Train this many segments concurrently (server use). "
                        "Each concurrent booster uses --n-jobs threads, so set "
                        "n_jobs * parallel_segments <= total cores.")
    p.add_argument("--io-workers", type=int, default=1,
                   help="Thread workers for parallel parquet reads when "
                        "scanning chunk files (1 = sequential).")
    p.add_argument("--feature-list", type=Path, default=None,
                   help="Optional JSON with a pre-saved feature list. "
                        "Defaults to all numeric+categorical cols in chunks "
                        "minus targets/ids/dates.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Panel I/O
# ---------------------------------------------------------------------------

EXCLUDE_FROM_FEATURES = {
    "INVENTORY_ITEM_ID", "XSRQ", "QTY",
    "target_15d", "target_30d",
    "baseline_15d", "baseline_30d",
}


def _discover_features(sample_path: Path) -> tuple[list[str], list[str]]:
    df = pd.read_parquet(sample_path)
    cols = [c for c in df.columns if c not in EXCLUDE_FROM_FEATURES]
    cat_cols = [c for c in cols if df[c].dtype == "int16" and df[c].max() < 32767]
    # Heuristic: small int16 columns are categorical codes; LightGBM accepts
    # the rest as numeric. The trainer was already saving them this way.
    numeric_cols = [c for c in cols if c not in cat_cols]
    return cat_cols, numeric_cols


def _load_segment_panel(
    chunks_dir: Path,
    horizon: int,
    item_to_seg: dict[int, int],
    target_seg: int,
    feature_cols: list[str],
    test_cutoff: pd.Timestamp,
    valid_cutoff: pd.Timestamp,
    max_rows: int | None,
    io_workers: int = 1,
) -> dict[str, pd.DataFrame]:
    """Stream every parquet chunk, pulling rows whose item belongs to the target
    segment, and split into train/valid/test by date."""
    target_col = f"target_{horizon}d"
    needed = list(dict.fromkeys(
        ["INVENTORY_ITEM_ID", "XSRQ", target_col, *feature_cols]
    ))
    train_parts, valid_parts, test_parts = [], [], []

    files = sorted(chunks_dir.glob("*.parquet"))

    def _read(f: Path) -> pd.DataFrame:
        return pd.read_parquet(f, columns=needed)

    if io_workers > 1 and len(files) > 1:
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor
        ex = ThreadPoolExecutor(max_workers=io_workers)
        pending: deque = deque()
        idx = 0
        while idx < len(files) and len(pending) < io_workers:
            pending.append(ex.submit(_read, files[idx])); idx += 1
        chunks_iter = iter(())
        def gen():
            nonlocal idx
            while pending:
                yield pending.popleft().result()
                if idx < len(files):
                    pending.append(ex.submit(_read, files[idx])); idx += 1
            ex.shutdown(wait=True)
        chunks_iter = gen()
    else:
        chunks_iter = (_read(f) for f in files)

    for i, df in enumerate(chunks_iter, 1):
        seg_codes = df["INVENTORY_ITEM_ID"].map(item_to_seg).fillna(
            SEGMENT_LABELS.index("cold")
        ).astype("int16").to_numpy()
        df = df[seg_codes == target_seg]
        df = df.dropna(subset=[target_col])
        if df.empty:
            continue

        df_train = df[df["XSRQ"] < valid_cutoff]
        df_valid = df[(df["XSRQ"] >= valid_cutoff) & (df["XSRQ"] < test_cutoff)]
        df_test = df[df["XSRQ"] >= test_cutoff]

        if not df_train.empty: train_parts.append(df_train)
        if not df_valid.empty: valid_parts.append(df_valid)
        if not df_test.empty:  test_parts.append(df_test)
        print(f"      [{SEGMENT_LABELS[target_seg]}] chunk {i}/{len(files)}  "
              f"train+={len(df_train):,} valid+={len(df_valid):,} test+={len(df_test):,}")

    train = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()
    valid = pd.concat(valid_parts, ignore_index=True) if valid_parts else pd.DataFrame()
    test  = pd.concat(test_parts,  ignore_index=True) if test_parts  else pd.DataFrame()

    if max_rows is not None and len(train) > max_rows:
        train = train.sample(n=max_rows, random_state=42).reset_index(drop=True)

    return {"train": train, "valid": valid, "test": test}


# ---------------------------------------------------------------------------
# Date splits
# ---------------------------------------------------------------------------

def _compute_date_cutoffs(chunks_dir: Path, valid_frac: float, test_frac: float,
                          io_workers: int = 1
                          ) -> tuple[pd.Timestamp, pd.Timestamp]:
    files = sorted(chunks_dir.glob("*.parquet"))
    def _read_dates(f: Path) -> pd.DataFrame:
        return pd.read_parquet(f, columns=["XSRQ"])
    if io_workers > 1 and len(files) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=io_workers) as ex:
            dfs = list(ex.map(_read_dates, files))
    else:
        dfs = [_read_dates(f) for f in files]
    lo, hi = None, None
    for d in dfs:
        if d.empty:
            continue
        a, b = d["XSRQ"].min(), d["XSRQ"].max()
        lo = a if lo is None else min(lo, a)
        hi = b if hi is None else max(hi, b)
    if lo is None:
        raise RuntimeError(f"No chunks with rows under {chunks_dir}")
    span = (hi - lo).days
    test_cutoff = lo + pd.Timedelta(days=int(span * (1 - test_frac)))
    valid_cutoff = lo + pd.Timedelta(days=int(span * (1 - test_frac - valid_frac)))
    return valid_cutoff, test_cutoff


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def _build_segments(chunks_dir: Path, io_workers: int = 1) -> pd.DataFrame:
    """Read minimal cols (ITEM_ID, XSRQ, lag_1) from every chunk and label items."""
    files = sorted(chunks_dir.glob("*.parquet"))
    cols = ["INVENTORY_ITEM_ID", "XSRQ", "lag_1"]
    if io_workers > 1 and len(files) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=io_workers) as ex:
            parts = list(ex.map(lambda f: pd.read_parquet(f, columns=cols), files))
    else:
        parts = [pd.read_parquet(f, columns=cols) for f in files]
    panel = pd.concat(parts, ignore_index=True)
    return compute_item_segments(panel, config=SegmentationConfig())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _build_params(args: argparse.Namespace, segment: str) -> dict:
    params = dict(BASE_PARAMS)
    params.update(SEGMENT_PARAMS.get(segment, {}))
    params["device_type"] = args.device
    params["num_threads"] = args.n_jobs
    if args.gpu_safe and args.device == "gpu":
        params["max_bin"] = 255
        params["max_cat_threshold"] = 64
    return params


def _train_one_segment(
    segment: str,
    splits: dict[str, pd.DataFrame],
    feature_cols: list[str],
    cat_cols: list[str],
    args: argparse.Namespace,
    out_dir: Path,
) -> dict:
    target_col = f"target_{args.horizon}d"
    train, valid, test = splits["train"], splits["valid"], splits["test"]

    if train.empty or valid.empty:
        print(f"  [{segment}] skip — train={len(train):,} valid={len(valid):,}")
        return {"segment": segment, "skipped": True,
                "n_train": len(train), "n_valid": len(valid), "n_test": len(test)}

    params = _build_params(args, segment)
    print(f"  [{segment}] train rows: {len(train):,}  valid: {len(valid):,}  "
          f"test: {len(test):,}  objective={params['objective']}")

    cat_idx = [feature_cols.index(c) for c in cat_cols if c in feature_cols]
    dtrain = lgb.Dataset(
        train[feature_cols], label=train[target_col].astype("float32"),
        categorical_feature=cat_idx, free_raw_data=False,
    )
    dvalid = lgb.Dataset(
        valid[feature_cols], label=valid[target_col].astype("float32"),
        categorical_feature=cat_idx, reference=dtrain, free_raw_data=False,
    )

    t0 = time.time()
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=SEGMENT_NUM_BOOST.get(segment, 1000),
        valid_sets=[dvalid],
        callbacks=[
            lgb.early_stopping(args.early_stopping, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )
    elapsed = time.time() - t0
    booster.save_model(str(out_dir / f"model_{segment}.lgb"))

    # Test metrics for this segment
    if test.empty:
        return {"segment": segment, "skipped": False,
                "n_train": len(train), "n_valid": len(valid), "n_test": 0,
                "best_iter": booster.best_iteration, "train_seconds": elapsed}

    pred = np.clip(booster.predict(test[feature_cols],
                                    num_iteration=booster.best_iteration), 0, None)
    y = test[target_col].to_numpy(dtype="float64")
    mae = float(np.abs(y - pred).mean())
    wape = float(np.abs(y - pred).sum() / max(np.abs(y).sum(), 1e-9))

    # Save predictions
    out_pred = pd.DataFrame({
        "INVENTORY_ITEM_ID": test["INVENTORY_ITEM_ID"].to_numpy(),
        "XSRQ": test["XSRQ"].to_numpy(),
        "y_true": y, "y_pred": pred, "segment": segment,
    })
    out_pred.to_csv(out_dir / f"test_predictions_{segment}.csv",
                    index=False, encoding="utf-8-sig")

    return {"segment": segment, "skipped": False,
            "n_train": len(train), "n_valid": len(valid), "n_test": len(test),
            "best_iter": booster.best_iteration, "train_seconds": elapsed,
            "mae": mae, "wape": wape}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Discovering panel features under {args.chunks_dir} ...")
    sample_chunk = next(iter(sorted(args.chunks_dir.glob("*.parquet"))))
    cat_cols, num_cols = _discover_features(sample_chunk)
    if args.feature_list and args.feature_list.exists():
        feats = json.loads(args.feature_list.read_text())
        cat_cols = feats["categorical"]
        num_cols = feats["numeric"]
    feature_cols = cat_cols + num_cols
    print(f"      {len(feature_cols)} features ({len(cat_cols)} categorical)")

    print("[2/5] Computing item segments ...")
    segs = _build_segments(args.chunks_dir, io_workers=args.io_workers)
    segs.to_csv(args.output_dir / "item_segments.csv",
                index=False, encoding="utf-8-sig")
    pop = segs["item_segment"].value_counts().reindex(SEGMENT_LABELS, fill_value=0)
    for label, count in pop.items():
        print(f"      {label:>8}: {count:>10,}")
    item_to_seg = dict(zip(
        segs["INVENTORY_ITEM_ID"],
        pd.Categorical(segs["item_segment"], categories=SEGMENT_LABELS).codes,
    ))

    print("[3/5] Computing date cutoffs ...")
    valid_cutoff, test_cutoff = _compute_date_cutoffs(
        args.chunks_dir, args.valid_frac, args.test_frac,
        io_workers=args.io_workers,
    )
    print(f"      valid >= {valid_cutoff.date()}, test >= {test_cutoff.date()}")

    print(f"[4/5] Training one model per segment "
          f"(parallel_segments={args.parallel_segments}, n_jobs={args.n_jobs}) ...")
    summaries = []
    active_segments = [(c, l) for c, l in enumerate(SEGMENT_LABELS) if pop[l] > 0]

    def _run(code: int, label: str) -> dict:
        print(f"\n--- segment: {label} ---")
        splits = _load_segment_panel(
            args.chunks_dir, args.horizon, item_to_seg, code,
            feature_cols, test_cutoff, valid_cutoff,
            args.max_rows_per_segment,
            io_workers=args.io_workers,
        )
        return _train_one_segment(
            label, splits, feature_cols, cat_cols, args, args.output_dir
        )

    if args.parallel_segments <= 1:
        for code, label in active_segments:
            summaries.append(_run(code, label))
    else:
        with ThreadPoolExecutor(max_workers=args.parallel_segments) as ex:
            futures = {ex.submit(_run, code, label): label
                       for code, label in active_segments}
            for fut in as_completed(futures):
                summaries.append(fut.result())

    print("[5/5] Aggregating overall metrics ...")
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(args.output_dir / "segment_metrics.csv",
                      index=False, encoding="utf-8-sig")

    # Combine all per-segment test predictions for an overall metric
    pred_files = sorted(args.output_dir.glob("test_predictions_*.csv"))
    if pred_files:
        merged = pd.concat([pd.read_csv(p) for p in pred_files], ignore_index=True)
        y, p = merged["y_true"].to_numpy(), merged["y_pred"].to_numpy()
        overall_mae = float(np.abs(y - p).mean())
        overall_wape = float(np.abs(y - p).sum() / max(np.abs(y).sum(), 1e-9))
        merged.to_csv(args.output_dir / "test_predictions_all.csv",
                      index=False, encoding="utf-8-sig")
        print(f"\n  OVERALL (segment-routed)  MAE={overall_mae:.4f}  WAPE={overall_wape:.4f}")
        with open(args.output_dir / "overall_metrics.json", "w") as f:
            json.dump({"mae": overall_mae, "wape": overall_wape,
                       "n_test": int(len(merged))}, f, indent=2)

    # Persist the routing meta so a downstream predictor can use it
    joblib.dump({
        "horizon": args.horizon,
        "feature_cols": feature_cols,
        "categorical_cols": cat_cols,
        "segment_labels": SEGMENT_LABELS,
        "item_to_segment_code": item_to_seg,
    }, args.output_dir / "routing_meta.joblib")

    print(f"\nSaved outputs under {args.output_dir}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
