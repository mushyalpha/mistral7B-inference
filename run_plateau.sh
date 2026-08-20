#!/bin/bash
set -e

# We use a separate CSV so we don't mix with previous runs
CSV="results/plateau_results.csv"
rm -f "$CSV"

# finer grid
CONCURRENCY="1,2,4,8,16,24,32,48,64,96,128,192,256"

# Running long inputs and 512 outputs
echo "Running HuggingFace benchmark..."
python benchmark.py --engine hf --concurrency $CONCURRENCY --input-lengths long --output-lengths 512 --output-csv $CSV || echo "HF benchmark failed or OOMed. Continuing to vLLM..."

echo "Running vLLM benchmark..."
python benchmark.py --engine vllm --concurrency $CONCURRENCY --input-lengths long --output-lengths 512 --output-csv $CSV || echo "vLLM benchmark failed. Continuing..."

echo "Plotting results..."
python plot_results.py --csv $CSV --out-dir results/plateau_plots

echo "Done! Check results/plateau_plots for the graphs."
