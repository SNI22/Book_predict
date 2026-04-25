"""Compute naive baselines on the xsyc per-item test split for fair comparison.

Mirrors xsyc.py: per INVENTORY_ITEM_ID, sort by XSRQ, 80/20 chronological split,
items with <20 rows skipped, train_size>=5. Baselines:
  - mean: predict mean(y_train)
  - last: predict last value of y_train
  - roll7: predict rolling-7-day mean of y_train tail
Reports MAE, RMSE, MAPE (zero-masked, like xsyc), WAPE, R2 per item, then aggregates.
Also reports the same aggregates from xsyc's saved LightGBM eval for side-by-side.
"""
import numpy as np
import pandas as pd
def mean_absolute_error(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def mean_squared_error(y, p):
    return float(np.mean((np.asarray(y) - np.asarray(p)) ** 2))


def r2_score(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

SRC = "/home/sni22/Documents/book_predict_lgbm/xsyc/训练原数据.csv"
LGBM_EVAL = "/home/sni22/Documents/book_predict_lgbm/xsyc/商品模型评估结果.csv"
OUT = "/home/sni22/Documents/book_predict_lgbm/xsyc/baseline_对比结果.csv"


def mape(y, p):
    y, p = np.asarray(y), np.asarray(p)
    m = y != 0
    return np.mean(np.abs((y[m] - p[m]) / y[m])) * 100 if m.any() else np.nan


def wape(y, p):
    y, p = np.asarray(y), np.asarray(p)
    s = np.sum(np.abs(y))
    return np.sum(np.abs(y - p)) / s * 100 if s else np.nan


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return mae, rmse, mape(y_true, y_pred), wape(y_true, y_pred), r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan


def main():
    df = pd.read_csv(SRC)
    df["XSRQ"] = pd.to_datetime(df["XSRQ"])
    df = df.sort_values(["INVENTORY_ITEM_ID", "XSRQ"])

    rows = []
    for item_id, g in df.groupby("INVENTORY_ITEM_ID"):
        if len(g) < 20:
            continue
        split = int(len(g) * 0.8)
        if split < 5:
            continue
        y_train = g["QTY"].iloc[:split].to_numpy()
        y_test = g["QTY"].iloc[split:].to_numpy()
        if len(y_test) == 0:
            continue

        preds = {
            "mean": np.full_like(y_test, y_train.mean(), dtype=float),
            "last": np.full_like(y_test, y_train[-1], dtype=float),
            "roll7": np.full_like(y_test, y_train[-7:].mean(), dtype=float),
        }
        rec = {"INVENTORY_ITEM_ID": item_id, "n": len(g), "n_test": len(y_test)}
        for name, p in preds.items():
            mae, rmse, mp, wp, r2 = evaluate(y_test, p)
            rec[f"{name}_MAE"] = mae
            rec[f"{name}_RMSE"] = rmse
            rec[f"{name}_MAPE"] = mp
            rec[f"{name}_WAPE"] = wp
            rec[f"{name}_R2"] = r2
        rows.append(rec)

    base = pd.DataFrame(rows)
    base.to_csv(OUT, index=False, encoding="utf-8-sig")

    print(f"Per-item baseline rows: {len(base)}")
    print()
    print("=== BASELINES (per-item averages over xsyc test slices) ===")
    for name in ["mean", "last", "roll7"]:
        print(
            f"{name:>6}  MAE={base[f'{name}_MAE'].mean():.3f}  "
            f"RMSE={base[f'{name}_RMSE'].mean():.3f}  "
            f"MAPE={base[f'{name}_MAPE'].dropna().mean():.2f}%  "
            f"WAPE={base[f'{name}_WAPE'].dropna().mean():.2f}%  "
            f"R2={base[f'{name}_R2'].dropna().mean():.4f}"
        )

    lgbm = pd.read_csv(LGBM_EVAL)
    print()
    print("=== xsyc LightGBM (from 商品模型评估结果.csv) ===")
    print(
        f" lgbm  MAE={lgbm['MAE'].mean():.3f}  "
        f"RMSE={lgbm['RMSE'].mean():.3f}  "
        f"MAPE={lgbm['MAPE(%)'].dropna().mean():.2f}%  "
        f"WAPE={lgbm['WAPE(%)'].dropna().mean():.2f}%  "
        f"R2={lgbm['R2'].dropna().mean():.4f}"
    )
    print(f"\nSaved per-item baseline detail -> {OUT}")


if __name__ == "__main__":
    main()
