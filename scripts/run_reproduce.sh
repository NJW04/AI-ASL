#!/usr/bin/env bash
set -euo pipefail

echo "====================================="
echo " START: Reproduce (download->run all) "
echo "====================================="

USE_KAGGLE=0
for arg in "$@"; do
  if [[ "$arg" == "--use-kaggle" ]]; then
    USE_KAGGLE=1
  fi
done

if [[ "$USE_KAGGLE" -eq 1 ]]; then
  echo "Attempting Kaggle download into ./data/ ..."
  python - <<'PY'
from pathlib import Path
from data.asl import ensure_data
ensure_data(Path("data"), use_kaggle=True)
PY
fi

echo "[1/3] Baseline ..."
bash scripts/run_baseline.sh --train-dir data/asl_alphabet_train --test-dir data/asl_alphabet_test

echo "[2/3] Train CNN ..."
bash scripts/run_train.sh --train-dir data/asl_alphabet_train --test-dir data/asl_alphabet_test

echo "[3/3] Evaluate last checkpoint ..."
last_train=$(ls -dt artifacts/asl_runs/*-train-* | head -n1)
python eval.py --checkpoint "${last_train}/best.pt" \
               --train-dir data/asl_alphabet_train --test-dir data/asl_alphabet_test

echo "Done. See artifacts in ./artifacts/asl_runs/"
