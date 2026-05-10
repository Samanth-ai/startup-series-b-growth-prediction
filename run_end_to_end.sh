#!/usr/bin/env bash
set -euo pipefail

# End-to-end training run.
# Update the path below if your CSV is stored somewhere else.
python -m src.run_pipeline   --input data/investments_VC.csv   --output-dir .   --clear-outputs
