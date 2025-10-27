#!/usr/bin/env bash
set -euo pipefail

echo "========================="
echo " START: Train CNN (ASL) "
echo "========================="

# Sensible defaults; pass-through will override any.
python train.py \
  --train-dir data/asl_alphabet_train \
  --test-dir data/asl_alphabet_test \
  --epochs 15 \
  --batch-size 64 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --patience 5 \
  --model cnn_small \
  --aug \
  "$@"

latest_run=$(ls -dt artifacts/asl_runs/*-train-* | head -n1 || true)
echo "Train artifacts -> ${latest_run}"
