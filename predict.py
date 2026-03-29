"""
Book Sales Prediction Model

Trains a model on historical book sales data and predicts next month's sales.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

DATA_PATH = "data/book_sales.csv"
OUTPUT_DIR = "output"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load and parse the book sales CSV."""
    df = pd.read_csv(path, parse_dates=["date"])
    df.sort_values(["book_id", "date"], inplace=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-series and categorical features."""
    df = df.copy()

    # Time features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter

    # Lag features (previous months' sales per book)
    for lag in [1, 2, 3]:
        df[f"sales_lag_{lag}"] = df.groupby("book_id")["units_sold"].shift(lag)

    # Rolling averages
    df["sales_rolling_3m"] = df.groupby("book_id")["units_sold"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df["sales_rolling_6m"] = df.groupby("book_id")["units_sold"].transform(
        lambda x: x.shift(1).rolling(6, min_periods=1).mean()
    )

    # Month-over-month change
    df["sales_mom_change"] = df.groupby("book_id")["units_sold"].pct_change()

    # Encode categoricals
    for col in ["genre", "format"]:
        le = LabelEncoder()
        df[f"{col}_encoded"] = le.fit_transform(df[col])

    # Drop rows with NaN from lag features
    df.dropna(subset=["sales_lag_1", "sales_lag_2", "sales_lag_3"], inplace=True)

    return df


FEATURE_COLS = [
    "month",
    "quarter",
    "year",
    "genre_encoded",
    "format_encoded",
    "price",
    "author_popularity",
    "marketing_spend",
    "avg_rating",
    "sales_lag_1",
    "sales_lag_2",
    "sales_lag_3",
    "sales_rolling_3m",
    "sales_rolling_6m",
    "sales_mom_change",
]


def train_model(df: pd.DataFrame):
    """Train a Gradient Boosting model with time-series cross-validation."""
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS)

    X = df[FEATURE_COLS]
    y = df["units_sold"]

    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )

    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_absolute_error")
    print(f"Cross-Validation MAE: {-cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")

    # Train on all data
    model.fit(X, y)

    return model, X, y


def evaluate_model(model, X, y):
    """Print evaluation metrics on training data."""
    preds = model.predict(X)
    print(f"\n--- Training Set Metrics ---")
    print(f"  MAE:  {mean_absolute_error(y, preds):.2f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y, preds)):.2f}")
    print(f"  R²:   {r2_score(y, preds):.4f}")


def predict_next_month(model, df: pd.DataFrame) -> pd.DataFrame:
    """Generate predictions for the next month for each book."""
    latest = df.sort_values("date").groupby("book_id").tail(1).copy()
    last_date = latest["date"].max()
    next_month = last_date + pd.offsets.MonthBegin(1)

    # Build feature rows for next month
    pred_rows = []
    for _, row in latest.iterrows():
        history = df[df["book_id"] == row["book_id"]].sort_values("date")

        pred_row = {
            "book_id": row["book_id"],
            "genre": row["genre"],
            "format": row["format"],
            "month": next_month.month,
            "quarter": (next_month.month - 1) // 3 + 1,
            "year": next_month.year,
            "genre_encoded": row["genre_encoded"],
            "format_encoded": row["format_encoded"],
            "price": row["price"],
            "author_popularity": row["author_popularity"],
            "marketing_spend": row["marketing_spend"],
            "avg_rating": row["avg_rating"],
            "sales_lag_1": history["units_sold"].iloc[-1],
            "sales_lag_2": history["units_sold"].iloc[-2] if len(history) >= 2 else history["units_sold"].iloc[-1],
            "sales_lag_3": history["units_sold"].iloc[-3] if len(history) >= 3 else history["units_sold"].iloc[-1],
            "sales_rolling_3m": history["units_sold"].iloc[-3:].mean(),
            "sales_rolling_6m": history["units_sold"].iloc[-6:].mean(),
            "sales_mom_change": (
                (history["units_sold"].iloc[-1] - history["units_sold"].iloc[-2]) / history["units_sold"].iloc[-2]
                if len(history) >= 2 and history["units_sold"].iloc[-2] != 0
                else 0
            ),
        }
        pred_rows.append(pred_row)

    pred_df = pd.DataFrame(pred_rows)
    pred_df["predicted_sales"] = model.predict(pred_df[FEATURE_COLS]).round().astype(int)
    pred_df["predicted_sales"] = pred_df["predicted_sales"].clip(lower=0)
    pred_df["prediction_month"] = next_month

    return pred_df[["book_id", "genre", "format", "prediction_month", "predicted_sales"]]


def plot_feature_importance(model, output_dir: str = OUTPUT_DIR):
    """Save a feature importance bar chart."""
    os.makedirs(output_dir, exist_ok=True)
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    importances.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=150)
    plt.close()
    print(f"\nFeature importance plot saved to {output_dir}/feature_importance.png")


def plot_predictions_by_genre(predictions: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    """Save a bar chart of predicted sales by genre."""
    os.makedirs(output_dir, exist_ok=True)
    genre_sales = predictions.groupby("genre")["predicted_sales"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    genre_sales.plot(kind="bar", ax=ax, color="coral")
    ax.set_title("Predicted Next-Month Sales by Genre")
    ax.set_ylabel("Total Predicted Units Sold")
    ax.set_xlabel("Genre")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "predictions_by_genre.png"), dpi=150)
    plt.close()
    print(f"Predictions by genre plot saved to {output_dir}/predictions_by_genre.png")


def main():
    print("=" * 50)
    print("  Book Sales Prediction Model")
    print("=" * 50)

    # Load
    print(f"\nLoading data from {DATA_PATH}...")
    df = load_data()
    print(f"  {len(df)} rows, {df['book_id'].nunique()} books, {df['date'].nunique()} months")

    # Feature engineering
    print("\nEngineering features...")
    df_feat = engineer_features(df)
    print(f"  {len(df_feat)} rows after feature engineering")

    # Train
    print("\nTraining model...")
    model, X, y = train_model(df_feat)
    evaluate_model(model, X, y)

    # Predict next month
    print("\nPredicting next month's sales...")
    predictions = predict_next_month(model, df_feat)
    print(f"\n  Prediction month: {predictions['prediction_month'].iloc[0].strftime('%B %Y')}")
    print(f"  Total predicted units: {predictions['predicted_sales'].sum():,}")
    print(f"\n  Top 10 books by predicted sales:")
    top10 = predictions.nlargest(10, "predicted_sales")
    print(top10.to_string(index=False))

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "next_month_predictions.csv")
    predictions.to_csv(out_path, index=False)
    print(f"\n  Full predictions saved to {out_path}")

    # Plots
    plot_feature_importance(model)
    plot_predictions_by_genre(predictions)

    print("\nDone!")


if __name__ == "__main__":
    main()
