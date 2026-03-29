# Book Sales Prediction Model

Predicts next month's book sales using a Gradient Boosting model trained on historical sales data.

## Quick Start

```bash
pip install -r requirements.txt

# Generate synthetic data (or provide your own — see below)
mkdir -p data
python generate_data.py

# Train model and predict next month
python predict.py
```

## Using Your Own Data

Replace `data/book_sales.csv` with your own CSV. Required columns:

| Column | Type | Description |
|---|---|---|
| `date` | date (YYYY-MM-DD) | First day of the month |
| `book_id` | int | Unique book identifier |
| `genre` | string | Book genre |
| `format` | string | Hardcover, Paperback, E-book, or Audiobook |
| `price` | float | Retail price |
| `author_popularity` | float | 0.0–1.0 popularity score |
| `marketing_spend` | float | Monthly marketing budget |
| `avg_rating` | float | Average customer rating (1–5) |
| `units_sold` | int | Units sold that month |

Then run `python predict.py` — no other changes needed.

## Output

- `output/next_month_predictions.csv` — per-book predicted sales
- `output/feature_importance.png` — which features drive predictions
- `output/predictions_by_genre.png` — predicted sales broken down by genre
