from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from src.book_predict.trainer_lgbm import LGBMTrainerConfig, run_training


def setup_logging(output_dir: Path) -> None:
    """Configure logging to both console and a timestamped log file.

    Uses line-buffering so the log file stays current even if the process
    is killed (e.g. OOM).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"train_{timestamp}.log"

    # Unbuffered file handler
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # Redirect print() → logging so trainer_lgbm.py prints are captured
    class _LogStream:
        def write(self, msg: str) -> None:
            if msg.strip():
                logging.info(msg.rstrip())
        def flush(self) -> None:
            for h in root.handlers:
                h.flush()

    sys.stdout = _LogStream()  # type: ignore[assignment]

    logging.info(f"Log file: {log_path}")


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
        "--max-train-rows",
        type=int,
        default=None,
        help="Cap on training rows per horizon. Subsamples if usable rows exceed this. "
             "With 250M+ rows, 20-30M is usually sufficient for LightGBM.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu", "cuda"],
        default="gpu",
        help="LightGBM training device (`gpu`=OpenCL, `cuda`=CUDA).",
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
    parser.add_argument(
        "--max-bin",
        type=int,
        default=None,
        help="LightGBM max_bin. Lower values are more GPU-friendly but can reduce split fidelity.",
    )
    parser.add_argument(
        "--max-cat-threshold",
        type=int,
        default=None,
        help="LightGBM max_cat_threshold. Limits categorical split search complexity.",
    )
    parser.add_argument(
        "--max-cat-codes",
        type=int,
        default=None,
        help="Cap categorical codes per feature (keep top-frequency levels, map others to unknown).",
    )
    parser.add_argument(
        "--gpu-safe",
        action="store_true",
        help="Enable GPU-safe defaults: max_bin=255, max_cat_threshold=64, max_cat_codes=255 (unless overridden).",
    )
    parser.add_argument(
        "--build-workers",
        type=int,
        default=1,
        help="Process workers for chunked pandas feature build (1 = sequential). Increase carefully due RAM usage.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or Path("artifacts_lgbm")
    setup_logging(output_dir)
    config = LGBMTrainerConfig(
        txn_path=args.txn_path,
        meta_path=args.meta_path,
        output_dir=args.output_dir,
        horizons=args.horizons,
        min_history_days=args.min_history_days,
        panel_days=args.panel_days,
        max_items=args.max_items,
        max_train_rows=args.max_train_rows,
        device=args.device,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        max_bin=args.max_bin,
        max_cat_threshold=args.max_cat_threshold,
        max_cat_codes=args.max_cat_codes,
        gpu_safe=args.gpu_safe,
        build_workers=args.build_workers,
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
