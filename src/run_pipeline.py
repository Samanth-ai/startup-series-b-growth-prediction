"""Command-line entry point. Use ``python -m src.run_pipeline --help``."""

import argparse
import shutil
from pathlib import Path

from .pipeline import run_pipeline


def parse_args():
    p = argparse.ArgumentParser(
        description="Train the Series B forecasting models end-to-end.",
    )
    p.add_argument("--input", required=True,
                   help="Path to the investments_VC.csv Crunchbase dump.")
    p.add_argument("--output-dir", default=".",
                   help="Project root for the artifacts/ and reports/ folders.")
    p.add_argument("--clear-outputs", action="store_true",
                   help="Wipe artifacts/ and reports/ before running.")
    p.add_argument("--encoding", default="latin1",
                   help="CSV encoding (the Kaggle dump is latin1, not utf-8).")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--skip-shap", action="store_true",
                   help="Skip the SHAP beeswarm step (it's optional).")
    p.add_argument("--max-shap-samples", type=int, default=300,
                   help="Cap on rows used for SHAP, for speed.")
    return p.parse_args()


def main():
    args = parse_args()

    root = Path(args.output_dir).resolve()
    if args.clear_outputs:
        for sub in ("artifacts", "reports"):
            target = root / sub
            if target.exists():
                shutil.rmtree(target)

    run_pipeline(
        csv_path=Path(args.input).resolve(),
        root=root,
        encoding=args.encoding,
        random_state=args.random_state,
        skip_shap=args.skip_shap,
        max_shap_samples=args.max_shap_samples,
    )


if __name__ == "__main__":
    main()
