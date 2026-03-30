from __future__ import annotations

import argparse
from pathlib import Path

from src.book_predict.trainer_lgbm import LGBMTrainerConfig, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGBM book sales forecasting models.")
    parser.add_argument(
        "--txn-path",
        type=Path,
        default=Path("../book_predict/dataset_large/TMPNXJ202603271.csv"),
        help="Path to the transactions CSV (dataset_large file 1).",
    )
    parser.add_argument(
        "--meta-path",
        type=Path,
        default=Path("../book_predict/dataset_large/TMPNXJ202603272.csv"),
        help="Path to the item metadata CSV (dataset_large file 2).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts_lgbm"),
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
        help="Minimum unique sales dates required per item.",
    )
    parser.add_argument(
        "--panel-days",
        type=int,
        default=730,
        help="How many recent days to include in the dense daily panel.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional cap on eligible items (sorted by activity). Use for quick runs.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel threads for LightGBM (-1 = all cores).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = LGBMTrainerConfig(
        txn_path=args.txn_path,
        meta_path=args.meta_path,
        output_dir=args.output_dir,
        horizons=args.horizons,
        min_history_days=args.min_history_days,
        panel_days=args.panel_days,
        max_items=args.max_items,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )
    results = run_training(config)
    print("\n=== Summary ===")
    for r in results:
        print(
            f"horizon={r['horizon']}d  "
            f"best_iter={r['best_iteration']}  "
            f"test_wape={r['test_metrics']['wape']:.4f}  "
            f"baseline_test_wape={r['test_metrics']['baseline_wape']:.4f}  "
            f"output={r['output_dir']}"
        )


if __name__ == "__main__":
    main()
