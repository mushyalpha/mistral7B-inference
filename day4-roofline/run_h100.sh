#!/usr/bin/env bash
# Full Day 4 H100 pipeline: decode sweep -> torch.profiler traces -> charts.
# Safe to re-run; each stage is idempotent.
set -uo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-Qwen/Qwen2.5-7B}"

if [ -f results_decode.json ] && [ "${SKIP_SWEEP:-0}" = "1" ]; then
    echo "== 1/4: SKIPPED (results_decode.json already present, SKIP_SWEEP=1) =="
else
    echo "== 1/4: decode sweep =="
    python bench_decode.py --model "$MODEL" --output results_decode.json
fi
if [ ! -f results_decode.json ]; then
    echo "FATAL: results_decode.json missing, aborting." >&2
    exit 1
fi

echo "== 2/4, 3/4: torch.profiler busy fraction + breakdown (no nsys needed) =="
python profile_torch.py --model "$MODEL" --output nsys_summary.json \
    || echo "WARNING: torch.profiler run failed, charts 2 & 3 will fall back to illustrative placeholder data"

echo "== 4/4: generate charts =="
if [ -f nsys_summary.json ]; then
    python plot_hero.py --data results_decode.json --nsys nsys_summary.json --output-dir charts
else
    python plot_hero.py --data results_decode.json --output-dir charts
fi

echo
echo "DONE. Pull these back to your local machine:"
echo "  results_decode.json"
echo "  nsys_summary.json   (real busy%/breakdown if profile_torch.py succeeded)"
echo "  charts/*.png"
