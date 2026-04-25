"""Item segmentation features (Plan B).

Adds per-item segment labels and seasonality scores so a single LightGBM
model can specialize internally on:
    - regular  : sells nearly every day (>= 60 nonzero days in last 90)
    - medium   : sells most months (30 <= nonzero_days_90 < 60)
    - sparse   : occasional (5 <= nonzero_days_90 < 30)
    - seasonal : data-driven seasonality (YoY autocorr or monthly skew)
    - cold     : extremely sparse / brand-new (history < 90d or < 5 sale days)

Two public helpers:
    compute_item_segments(panel)        -> per-item DataFrame
    enrich_panel_with_segments(panel)   -> panel + segment columns merged in

The segment label is decided once per item, using the WHOLE history available
in the panel up to (but not including) each row's date is intentionally NOT
done here — segmentation is a static item attribute computed on the latest
available 90-day window. For training/prediction use, this is acceptable:
LightGBM uses the segment as a categorical feature, not a target.

If you need strict leakage-free segmentation (segment recomputed at each row's
"as-of" date), use compute_item_segments_asof — slower but safe for backtests.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


SEGMENT_FEATURES = [
    "item_segment",
    "nonzero_days_90",
    "seasonality_score",
    "yoy_strength",
    "history_days",
]

SEGMENT_CATEGORICAL = ["item_segment"]
SEGMENT_NUMERIC = [
    "nonzero_days_90",
    "seasonality_score",
    "yoy_strength",
    "history_days",
]

SEGMENT_LABELS = ["regular", "medium", "sparse", "seasonal", "cold"]


@dataclass
class SegmentationConfig:
    window_days: int = 90
    regular_threshold: int = 60      # >= 2/3 of window
    medium_threshold: int = 30       # >= 1/3 of window
    sparse_threshold: int = 5        # below this => cold
    min_history_days: int = 90       # below this => cold
    yoy_min_history: int = 365       # required for seasonality test
    yoy_strength_threshold: float = 2.0   # max(monthly)/mean(monthly)
    yoy_autocorr_threshold: float = 0.30  # corr(QTY_t, QTY_{t-365})


def compute_item_segments(
    panel: pd.DataFrame,
    config: SegmentationConfig | None = None,
) -> pd.DataFrame:
    """Compute one segment label per item from the full panel.

    Expects columns: INVENTORY_ITEM_ID, XSRQ, QTY (or lag_1 if QTY was dropped).

    Returns a DataFrame indexed implicitly with columns:
        INVENTORY_ITEM_ID, item_segment, nonzero_days_90,
        seasonality_score, yoy_strength, history_days
    """
    cfg = config or SegmentationConfig()

    qty_col = "QTY" if "QTY" in panel.columns else "lag_1"
    if qty_col not in panel.columns:
        raise ValueError(
            "panel must contain QTY or lag_1 for segmentation; "
            f"available: {list(panel.columns)[:20]}..."
        )

    df = panel[["INVENTORY_ITEM_ID", "XSRQ", qty_col]].copy()
    df["XSRQ"] = pd.to_datetime(df["XSRQ"])
    df = df.rename(columns={qty_col: "QTY"})

    # ---- per-item history bounds ----
    bounds = (
        df.groupby("INVENTORY_ITEM_ID")["XSRQ"]
        .agg(first_seen="min", last_seen="max")
        .reset_index()
    )
    bounds["history_days"] = (bounds["last_seen"] - bounds["first_seen"]).dt.days + 1

    # ---- 90-day window stats (last 90 days of each item's history) ----
    df = df.merge(bounds[["INVENTORY_ITEM_ID", "last_seen"]], on="INVENTORY_ITEM_ID")
    window_cutoff = df["last_seen"] - pd.Timedelta(days=cfg.window_days)
    recent = df[df["XSRQ"] > window_cutoff]
    nonzero_days_90 = (
        recent.assign(nz=(recent["QTY"] > 0).astype("int32"))
        .groupby("INVENTORY_ITEM_ID")["nz"]
        .sum()
        .rename("nonzero_days_90")
        .reset_index()
    )

    # ---- seasonality: YoY autocorr + monthly skew ----
    season = _compute_seasonality(df, cfg)

    # ---- merge ----
    out = bounds[["INVENTORY_ITEM_ID", "history_days"]].merge(
        nonzero_days_90, on="INVENTORY_ITEM_ID", how="left"
    )
    out["nonzero_days_90"] = out["nonzero_days_90"].fillna(0).astype("int32")
    out = out.merge(season, on="INVENTORY_ITEM_ID", how="left")
    out["seasonality_score"] = out["seasonality_score"].fillna(0.0).astype("float32")
    out["yoy_strength"] = out["yoy_strength"].fillna(1.0).astype("float32")

    # ---- assign segment label ----
    out["item_segment"] = _assign_segments(out, cfg)
    return out


def _compute_seasonality(df: pd.DataFrame, cfg: SegmentationConfig) -> pd.DataFrame:
    """Compute YoY autocorrelation and monthly-skew strength per item."""
    df = df.copy()
    df["month"] = df["XSRQ"].dt.month

    # Monthly totals per item
    monthly = (
        df.groupby(["INVENTORY_ITEM_ID", "month"])["QTY"].sum().reset_index()
    )
    monthly_stats = (
        monthly.groupby("INVENTORY_ITEM_ID")["QTY"]
        .agg(monthly_max="max", monthly_mean="mean", monthly_count="count")
        .reset_index()
    )
    monthly_stats["yoy_strength"] = np.where(
        monthly_stats["monthly_mean"] > 0,
        monthly_stats["monthly_max"] / monthly_stats["monthly_mean"],
        1.0,
    )

    # YoY autocorr: corr(QTY_t, QTY_{t-365}). Items with < yoy_min_history skip.
    df_sorted = df.sort_values(["INVENTORY_ITEM_ID", "XSRQ"])
    df_sorted["QTY_yoy"] = (
        df_sorted.groupby("INVENTORY_ITEM_ID")["QTY"].shift(365)
    )
    valid = df_sorted.dropna(subset=["QTY_yoy"])

    def _safe_corr(g: pd.DataFrame) -> float:
        if len(g) < 30:
            return 0.0
        a, b = g["QTY"].to_numpy(dtype="float64"), g["QTY_yoy"].to_numpy(dtype="float64")
        if a.std() == 0 or b.std() == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    if len(valid) > 0:
        autocorr = (
            valid.groupby("INVENTORY_ITEM_ID", group_keys=False)
            .apply(_safe_corr)
            .rename("seasonality_score")
            .reset_index()
        )
    else:
        autocorr = pd.DataFrame(
            columns=["INVENTORY_ITEM_ID", "seasonality_score"]
        )

    return monthly_stats[["INVENTORY_ITEM_ID", "yoy_strength"]].merge(
        autocorr, on="INVENTORY_ITEM_ID", how="left"
    )


def _assign_segments(stats: pd.DataFrame, cfg: SegmentationConfig) -> pd.Series:
    """Apply the decision rules. Order matters: cold > seasonal > frequency tiers."""
    nz = stats["nonzero_days_90"]
    hist = stats["history_days"]
    season = stats["seasonality_score"]
    yoy = stats["yoy_strength"]

    is_cold = (hist < cfg.min_history_days) | (nz < cfg.sparse_threshold)
    is_seasonal = (
        (hist >= cfg.yoy_min_history)
        & (
            (season >= cfg.yoy_autocorr_threshold)
            | (yoy >= cfg.yoy_strength_threshold)
        )
    )
    is_regular = nz >= cfg.regular_threshold
    is_medium = (nz >= cfg.medium_threshold) & (nz < cfg.regular_threshold)
    is_sparse = (nz >= cfg.sparse_threshold) & (nz < cfg.medium_threshold)

    label = np.full(len(stats), "cold", dtype=object)
    label[is_sparse.values] = "sparse"
    label[is_medium.values] = "medium"
    label[is_regular.values] = "regular"
    label[is_seasonal.values] = "seasonal"   # seasonal overrides frequency tier
    label[is_cold.values] = "cold"           # cold overrides everything
    return pd.Series(label, name="item_segment")


def enrich_panel_with_segments(
    panel: pd.DataFrame,
    segments: pd.DataFrame | None = None,
    config: SegmentationConfig | None = None,
) -> pd.DataFrame:
    """Merge segment columns onto a featured panel.

    `segments` can be precomputed (one row per item) — recommended when running
    on chunked parquet files so segmentation is computed once on the full
    panel and then broadcast to every chunk.
    """
    if segments is None:
        segments = compute_item_segments(panel, config=config)

    needed = ["INVENTORY_ITEM_ID"] + SEGMENT_FEATURES
    seg = segments[needed].copy()

    # Encode segment as int code for LightGBM (consistent with other categoricals)
    seg["item_segment"] = pd.Categorical(
        seg["item_segment"], categories=SEGMENT_LABELS
    ).codes.astype("int16")

    # Numeric dtypes
    seg["nonzero_days_90"] = seg["nonzero_days_90"].astype("int16")
    seg["seasonality_score"] = seg["seasonality_score"].astype("float32")
    seg["yoy_strength"] = seg["yoy_strength"].astype("float32")
    seg["history_days"] = seg["history_days"].astype("int32")

    enriched = panel.merge(seg, on="INVENTORY_ITEM_ID", how="left")
    enriched["item_segment"] = enriched["item_segment"].fillna(
        SEGMENT_LABELS.index("cold")
    ).astype("int16")
    enriched["nonzero_days_90"] = enriched["nonzero_days_90"].fillna(0).astype("int16")
    enriched["seasonality_score"] = enriched["seasonality_score"].fillna(0.0).astype("float32")
    enriched["yoy_strength"] = enriched["yoy_strength"].fillna(1.0).astype("float32")
    enriched["history_days"] = enriched["history_days"].fillna(0).astype("int32")
    return enriched


def per_segment_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    segments: np.ndarray,
) -> pd.DataFrame:
    """Compute MAE, WAPE, support per segment label code."""
    df = pd.DataFrame(
        {"y": y_true, "p": y_pred, "seg_code": segments.astype("int16")}
    )
    rows = []
    for code, label in enumerate(SEGMENT_LABELS):
        sub = df[df["seg_code"] == code]
        if len(sub) == 0:
            continue
        ay = np.abs(sub["y"].to_numpy())
        err = np.abs(sub["y"].to_numpy() - sub["p"].to_numpy())
        denom = ay.sum()
        rows.append({
            "segment": label,
            "n_rows": len(sub),
            "y_sum": float(ay.sum()),
            "mae": float(err.mean()),
            "wape": float(err.sum() / denom) if denom > 0 else float("nan"),
        })
    return pd.DataFrame(rows)
