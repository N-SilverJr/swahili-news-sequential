#!/bin/bash
# Runs every notebook end-to-end in QUICK mode (small subsets, 2 epochs, tiny random transformer).
# Use it to check the code works after a change. Never report numbers from this mode.
set -e
cd "$(dirname "$0")/.."
export QUICK=1 TINY_TRANSFORMER=1 MPLBACKEND=Agg
OUT=$(mktemp -d)
for nb in notebooks/0*.ipynb; do
  echo "== $nb"
  jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1800 --output-dir "$OUT" "$nb"
done
echo "All notebooks ran. Smoke-test outputs are in _quick/ (git-ignored)."
