#!/usr/bin/env bash
# Sweep the helix experiment across several models and all seven numeral
# scripts (--pool mean, the honest default).
#
# 4 models * 7 scripts = 28 runs. On an M-series Mac with all models
# already in the HF cache the smaller models are ~2 min each and the
# 31B Gemma is ~5-10 min, so plan ~60-90 min end-to-end.
#
# Order: small/fast models first so you see results before the big ones.
#
# Run:    ./run.sh
#
# Failures: any failing run prints "!! FAILED: ..." and the sweep
# continues. Re-run individual combos manually after fixing the issue.

set -euo pipefail

cd "$(dirname "$0")"

MODELS=(
  "EleutherAI/pythia-6.9b"
  "meta-llama/Llama-3.1-8B"
  "google/gemma-4-E4B"
  "google/gemma-4-31B"
)

SCRIPTS=(
  latin
  arabic
  persian
  devanagari
  chinese
  greek
  roman
)

for model in "${MODELS[@]}"; do
  for script in "${SCRIPTS[@]}"; do
    echo "============================================================"
    echo "  model=${model}    script=${script}"
    echo "============================================================"
    # `|| true` so a missing/gated/typo'd model id doesn't kill the
    # rest of the sweep under `set -e`.
    uv run main.py --model "${model}" --script "${script}" \
      || echo "  !! FAILED: model=${model} script=${script}  (continuing)"
  done
done

echo
echo "all runs complete. Figures in:  out/"
find out -name "fig*.png" 2>/dev/null | sort
