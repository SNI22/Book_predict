"""
Generate synthetic book sales data for model training.

Replace this with your own dataset by providing a CSV file with the same columns.
See README.md for the expected schema.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_book_sales_data(
    n_books: int = 200,
    start_date: str = "2023-01-01",
    end_date: str = "2026-03-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic monthly book sales data."""
    rng = np.random.default_rng(seed)

    genres = ["Fiction", "Non-Fiction", "Sci-Fi", "Romance", "Thriller", "Self-Help", "Children", "Biography"]
    formats = ["Hardcover", "Paperback", "E-book", "Audiobook"]

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    months = pd.date_range(start, end, freq="MS")

    rows = []
    for book_id in range(1, n_books + 1):
        genre = rng.choice(genres)
        book_format = rng.choice(formats)
        price = round(rng.uniform(5.99, 34.99), 2)
        author_popularity = rng.uniform(0.1, 1.0)  # 0-1 scale
        base_sales = rng.integers(50, 2000)

        for month in months:
            # Seasonality: higher in Nov-Dec (holiday), dip in summer
            month_num = month.month
            if month_num in (11, 12):
                seasonal_factor = rng.uniform(1.3, 1.8)
            elif month_num in (6, 7, 8):
                seasonal_factor = rng.uniform(0.7, 0.9)
            else:
                seasonal_factor = rng.uniform(0.9, 1.1)

            # Trend: slight growth over time
            months_elapsed = (month - start).days / 30
            trend_factor = 1 + 0.002 * months_elapsed

            # Marketing spend (random, correlated with sales)
            marketing_spend = round(rng.uniform(100, 5000) * author_popularity, 2)
            marketing_boost = 1 + 0.0001 * marketing_spend

            # Rating effect
            avg_rating = round(rng.uniform(2.5, 5.0), 1)
            rating_factor = 0.6 + 0.1 * avg_rating

            # Compute sales
            sales = int(
                base_sales
                * seasonal_factor
                * trend_factor
                * marketing_boost
                * rating_factor
                * rng.uniform(0.8, 1.2)
            )
            sales = max(0, sales)

            rows.append(
                {
                    "date": month,
                    "book_id": book_id,
                    "genre": genre,
                    "format": book_format,
                    "price": price,
                    "author_popularity": round(author_popularity, 3),
                    "marketing_spend": marketing_spend,
                    "avg_rating": avg_rating,
                    "units_sold": sales,
                }
            )

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = generate_book_sales_data()
    df.to_csv("data/book_sales.csv", index=False)
    print(f"Generated {len(df)} rows for {df['book_id'].nunique()} books over {df['date'].nunique()} months.")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nSample:\n{df.head(10)}")
