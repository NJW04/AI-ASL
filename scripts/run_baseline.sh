#!/usr/bin/env bash
set -euo pipefail

echo "================================"
echo " START: Baseline (LogReg + HOG) "
echo "================================"

# Pass-through args, default paths are repo-relative
python baselines/logreg_hog.py "$@"

latest_run=$(ls -dt artifacts/asl_runs/*-baseline-logreg | head -n1 || true)
echo "Baseline artifacts -> ${latest_run}"
