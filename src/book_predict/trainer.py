from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


STATIC_COLUMNS = [
    "LIST_PRICE_PER_UNIT",
    "ITEM_CATEORY_CODE",
    "ITEM_CATEORY",
    "BPDNAME",
    "DLNUM",
    "DLNAME",
]

CATEGORICAL_COLUMNS = [
    "ITEM_CATEORY_CODE",
    "ITEM_CATEORY",
    "BPDNAME",
    "DLNAME",
]

NUMERIC_FEATURE_COLUMNS = [
    "LIST_PRICE_PER_UNIT",
    "DLNUM",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "roll_sum_7",
    "roll_sum_14",
    "roll_sum_28",
    "roll_mean_7",
    "roll_mean_14",
    "roll_mean_28",
    "nonzero_days_28",
    "days_since_sale",
    "day_of_week",
    "day_of_month",
    "month",
    "week_of_year",
    "is_month_start",
    "is_month_end",
    "item_age_days",
]


@dataclass
class TrainerConfig:
    data_path: Path
    output_dir: Path
    horizons: list[int]
    min_history_days: int = 10
    panel_days: int = 400
    max_items: int | None = None
    min_nonzero_target_share: float = 0.05
    random_state: int = 42


def _forward_sum(values: pd.Series, horizon: int) -> pd.Series:
    # Sum qty over the next `horizon` days, excluding today.
    shifted = values.shift(-1)
    return shifted[::-1].rolling(horizon, min_periods=horizon).sum()[::-1]


def _ensure_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_and_aggregate_sales(data_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(data_path, encoding="utf-8-sig", parse_dates=["XSRQ"])
    _ensure_columns(frame, ["INVENTORY_ITEM_ID", "XSRQ", "QTY", *STATIC_COLUMNS])

    frame["BPDNAME"] = frame["BPDNAME"].fillna("UNKNOWN_PUBLISHER")
    frame["UN_NUMBER"] = frame["UN_NUMBER"].fillna("UNKNOWN_UN_NUMBER")
    frame["ITEM_CATEORY_CODE"] = frame["ITEM_CATEORY_CODE"].fillna("UNKNOWN_CATEGORY_CODE")
    frame["ITEM_CATEORY"] = frame["ITEM_CATEORY"].fillna("UNKNOWN_CATEGORY")
    frame["DLNAME"] = frame["DLNAME"].fillna("UNKNOWN_DLNAME")

    aggregated = (
        frame.groupby(["INVENTORY_ITEM_ID", "XSRQ"], as_index=False)
        .agg(
            QTY=("QTY", "sum"),
            LIST_PRICE_PER_UNIT=("LIST_PRICE_PER_UNIT", "first"),
            ITEM_CATEORY_CODE=("ITEM_CATEORY_CODE", "first"),
            ITEM_CATEORY=("ITEM_CATEORY", "first"),
            BPDNAME=("BPDNAME", "first"),
            DLNUM=("DLNUM", "first"),
            DLNAME=("DLNAME", "first"),
        )
        .sort_values(["INVENTORY_ITEM_ID", "XSRQ"])
        .reset_index(drop=True)
    )
    aggregated["INVENTORY_ITEM_ID"] = aggregated["INVENTORY_ITEM_ID"].astype("int64")
    return aggregated


def select_eligible_items(frame: pd.DataFrame, min_history_days: int, max_items: int | None) -> np.ndarray:
    history = frame.groupby("INVENTORY_ITEM_ID")["XSRQ"].nunique()
    eligible = history[history >= min_history_days]
    if max_items is not None:
        eligible = eligible.sort_values(ascending=False).head(max_items)
    return eligible.index.to_numpy()


def build_dense_panel(frame: pd.DataFrame, eligible_items: np.ndarray, panel_days: int, max_horizon: int) -> pd.DataFrame:
    frame = frame[frame["INVENTORY_ITEM_ID"].isin(eligible_items)].copy()
    latest_date = frame["XSRQ"].max()
    panel_start_floor = latest_date - pd.Timedelta(days=panel_days + max_horizon + 35)

    panels: list[pd.DataFrame] = []
    for item_id, item_frame in frame.groupby("INVENTORY_ITEM_ID", sort=False):
        item_frame = item_frame.sort_values("XSRQ").copy()
        start_date = max(item_frame["XSRQ"].min(), panel_start_floor)
        date_index = pd.date_range(start=start_date, end=latest_date, freq="D")
        dense = item_frame.set_index("XSRQ").reindex(date_index)
        dense.index.name = "XSRQ"
        dense = dense.reset_index()
        dense["INVENTORY_ITEM_ID"] = item_id
        dense["QTY"] = dense["QTY"].fillna(0.0)

        for column in STATIC_COLUMNS:
            dense[column] = dense[column].ffill().bfill()

        dense["first_sale_date"] = item_frame["XSRQ"].min()
        panels.append(dense)

    panel = pd.concat(panels, ignore_index=True)
    return panel.sort_values(["INVENTORY_ITEM_ID", "XSRQ"]).reset_index(drop=True)


def add_time_series_features(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    frame = panel.copy()
    grouped = frame.groupby("INVENTORY_ITEM_ID", sort=False)

    frame[f"target_{horizon}d"] = grouped["QTY"].transform(lambda s: _forward_sum(s, horizon))
    frame["lag_1"] = grouped["QTY"].shift(1)
    frame["lag_7"] = grouped["QTY"].shift(7)
    frame["lag_14"] = grouped["QTY"].shift(14)
    frame["lag_28"] = grouped["QTY"].shift(28)

    shifted = grouped["QTY"].shift(1)
    roll_group = shifted.groupby(frame["INVENTORY_ITEM_ID"], sort=False)
    frame["roll_sum_7"] = roll_group.transform(lambda s: s.rolling(7, min_periods=7).sum())
    frame["roll_sum_14"] = roll_group.transform(lambda s: s.rolling(14, min_periods=14).sum())
    frame["roll_sum_28"] = roll_group.transform(lambda s: s.rolling(28, min_periods=28).sum())
    frame["roll_mean_7"] = frame["roll_sum_7"] / 7.0
    frame["roll_mean_14"] = frame["roll_sum_14"] / 14.0
    frame["roll_mean_28"] = frame["roll_sum_28"] / 28.0
    frame["nonzero_days_28"] = roll_group.transform(lambda s: s.gt(0).rolling(28, min_periods=28).sum())

    prior_sales_date = frame["XSRQ"].where(frame["QTY"] > 0)
    prior_sales_date = prior_sales_date.groupby(frame["INVENTORY_ITEM_ID"], sort=False).ffill()
    frame["days_since_sale"] = (frame["XSRQ"] - prior_sales_date).dt.days

    calendar = frame["XSRQ"].dt
    frame["day_of_week"] = calendar.dayofweek
    frame["day_of_month"] = calendar.day
    frame["month"] = calendar.month
    frame["week_of_year"] = calendar.isocalendar().week.astype("int16")
    frame["is_month_start"] = calendar.is_month_start.astype("int8")
    frame["is_month_end"] = calendar.is_month_end.astype("int8")
    frame["item_age_days"] = (frame["XSRQ"] - frame["first_sale_date"]).dt.days

    frame = frame.drop(columns=["first_sale_date"])
    return frame


def split_by_time(frame: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    usable = frame.dropna(subset=[f"target_{horizon}d"]).copy()
    unique_dates = np.sort(usable["XSRQ"].unique())
    if len(unique_dates) < 120:
        raise ValueError("Not enough history to create reliable time-based splits.")

    train_end = unique_dates[int(len(unique_dates) * 0.7)]
    valid_end = unique_dates[int(len(unique_dates) * 0.85)]

    train = usable[usable["XSRQ"] <= train_end].copy()
    valid = usable[(usable["XSRQ"] > train_end) & (usable["XSRQ"] <= valid_end)].copy()
    test = usable[usable["XSRQ"] > valid_end].copy()
    return train, valid, test


def build_model(random_state: int) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
                CATEGORICAL_COLUMNS,
            ),
            ("numeric", "passthrough", NUMERIC_FEATURE_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.03,
        max_iter=400,
        max_depth=4,
        min_samples_leaf=300,
        l2_regularization=1.0,
        random_state=random_state,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def _metric_frame(y_true: pd.Series, y_pred: np.ndarray, baseline: np.ndarray) -> dict[str, float]:
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    wape = float(np.abs(y_true - y_pred).sum() / max(np.abs(y_true).sum(), 1e-9))
    baseline_mae = mean_absolute_error(y_true, baseline)
    baseline_wape = float(np.abs(y_true - baseline).sum() / max(np.abs(y_true).sum(), 1e-9))
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "wape": wape,
        "baseline_mae": float(baseline_mae),
        "baseline_wape": baseline_wape,
    }


def _wape(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
    y_true_array = np.asarray(y_true)
    return float(np.abs(y_true_array - y_pred).sum() / max(np.abs(y_true_array).sum(), 1e-9))


def train_for_horizon(frame: pd.DataFrame, config: TrainerConfig, horizon: int) -> dict[str, object]:
    featured = add_time_series_features(frame, horizon=horizon)
    featured = featured.dropna(
        subset=[
            f"target_{horizon}d",
            "lag_1",
            "lag_7",
            "lag_14",
            "lag_28",
            "roll_sum_7",
            "roll_sum_14",
            "roll_sum_28",
            "days_since_sale",
        ]
    ).copy()

    target_column = f"target_{horizon}d"
    featured["baseline_28_scaled"] = featured["roll_mean_28"] * horizon
    train, valid, test = split_by_time(featured, horizon)

    feature_columns = CATEGORICAL_COLUMNS + NUMERIC_FEATURE_COLUMNS + ["baseline_28_scaled"]
    train_x = train[feature_columns]
    valid_x = valid[feature_columns]
    test_x = test[feature_columns]

    train_y = train[target_column]
    valid_y = valid[target_column]
    test_y = test[target_column]

    baseline_valid = valid["roll_mean_28"].to_numpy() * horizon
    baseline_test = test["roll_mean_28"].to_numpy() * horizon
    baseline_train = train["roll_mean_28"].to_numpy() * horizon

    residual_train_y = train_y.to_numpy() - baseline_train

    model = build_model(random_state=config.random_state)
    model.fit(train_x, residual_train_y)

    valid_residual_pred = model.predict(valid_x)
    test_residual_pred = model.predict(test_x)

    # Keep the residual model as a small correction to the stronger rolling baseline.
    blend_candidates = np.array([0.0, 0.02, 0.05, 0.1])
    best_blend = min(
        blend_candidates,
        key=lambda alpha: _wape(valid_y, baseline_valid + alpha * valid_residual_pred),
    )

    valid_pred = baseline_valid + best_blend * valid_residual_pred
    test_pred = baseline_test + best_blend * test_residual_pred

    valid_metrics = _metric_frame(valid_y, valid_pred, baseline_valid)
    test_metrics = _metric_frame(test_y, test_pred, baseline_test)

    output_dir = config.output_dir / f"horizon_{horizon}d"
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_dir / "model.joblib")

    predictions = test[["INVENTORY_ITEM_ID", "XSRQ", target_column, "QTY"]].copy()
    predictions["prediction"] = test_pred
    predictions["baseline_prediction"] = baseline_test
    predictions.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        [
            {"split": "valid", **valid_metrics},
            {"split": "test", **test_metrics},
        ]
    ).to_csv(output_dir / "metrics.csv", index=False, encoding="utf-8-sig")

    return {
        "horizon": horizon,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
        "test_rows": int(len(test)),
        "blend_weight": float(best_blend),
        "output_dir": output_dir,
    }


def run_training(config: TrainerConfig) -> list[dict[str, object]]:
    sales = load_and_aggregate_sales(config.data_path)
    eligible_items = select_eligible_items(
        sales,
        min_history_days=config.min_history_days,
        max_items=config.max_items,
    )
    if len(eligible_items) == 0:
        raise ValueError("No items satisfied the minimum history threshold.")

    panel = build_dense_panel(
        sales,
        eligible_items=eligible_items,
        panel_days=config.panel_days,
        max_horizon=max(config.horizons),
    )

    results = []
    for horizon in config.horizons:
        results.append(train_for_horizon(panel, config, horizon))

    summary_rows = []
    for result in results:
        summary_rows.append(
            {
                "horizon": result["horizon"],
                "train_rows": result["train_rows"],
                "valid_rows": result["valid_rows"],
                "test_rows": result["test_rows"],
                "valid_mae": result["valid_metrics"]["mae"],
                "valid_wape": result["valid_metrics"]["wape"],
                "valid_baseline_wape": result["valid_metrics"]["baseline_wape"],
                "test_mae": result["test_metrics"]["mae"],
                "test_wape": result["test_metrics"]["wape"],
                "test_baseline_wape": result["test_metrics"]["baseline_wape"],
                "blend_weight": result["blend_weight"],
            }
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(
        config.output_dir / "training_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return results
