from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train book sales forecasting models.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("dataset/TMPNXJ20260322.csv"),
        help="Path to the sales CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory where models and reports will be written.",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[15, 30],
        help="Forecast horizons in days.",
    )
    parser.add_argument(
        "--min-history-days",
        type=int,
        default=10,
        help="Minimum number of unique sales dates required for an item to enter training.",
    )
    parser.add_argument(
        "--panel-days",
        type=int,
        default=400,
        help="How many recent days to include when building the dense daily panel.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional cap on eligible items, sorted by historical activity.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for deterministic training.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="OpenMP threads for HistGradientBoosting (-1 = all cores).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_jobs == -1:
        os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 1)
    else:
        os.environ["OMP_NUM_THREADS"] = str(args.n_jobs)
    # Defer import until after env var is set so OpenMP picks it up.
    from src.book_predict.trainer import TrainerConfig, run_training
    config = TrainerConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        horizons=args.horizons,
        min_history_days=args.min_history_days,
        panel_days=args.panel_days,
        max_items=args.max_items,
        random_state=args.random_state,
    )
    results = run_training(config)
    for result in results:
        print(
            f"horizon={result['horizon']} "
            f"test_wape={result['test_metrics']['wape']:.4f} "
            f"baseline_test_wape={result['test_metrics']['baseline_wape']:.4f} "
            f"output={result['output_dir']}"
        )


if __name__ == "__main__":
    main()

