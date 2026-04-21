#!/usr/bin/env bash
set -euo pipefail

# Example usage:
# ./run_experiment.sh
# ./run_experiment.sh --compare_models --epochs 2 --train_size 1000 --val_size 200

python -m src.main "$@"
