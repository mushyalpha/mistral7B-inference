"""
Builds the report deliverables from results/benchmark_results.csv:
  1. Throughput vs. concurrency plot (the money chart), faceted by
     input-length category and output length so prefill vs. decode effects
     are visible separately.
  2. A latency percentile (p50/p90/p99) table at a few concurrency levels.
  3. A peak-GPU-memory comparison plot.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PALETTE = {"hf": "#ff7f0e", "vllm": "#1f77b4"}


def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise SystemExit(f"{csv_path} is empty -- run benchmark.py for both --engine hf and --engine vllm first.")
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Averages across trials so plots/tables reflect the whole sweep, not one run."""
    group_cols = ["engine", "input_len_category", "output_len_tokens", "concurrency"]
    agg = df.groupby(group_cols).agg(
        throughput_mean=("throughput_tok_s", "mean"),
        throughput_std=("throughput_tok_s", "std"),
        latency_p50_mean=("latency_p50_s", "mean"),
        latency_p90_mean=("latency_p90_s", "mean"),
        latency_p99_mean=("latency_p99_s", "mean"),
        peak_mem_mean=("peak_mem_gb", "mean"),
        n_trials=("trial", "count"),
    ).reset_index()
    return agg


def plot_throughput(agg: pd.DataFrame, out_dir: Path) -> None:
    g = sns.relplot(
        data=agg, x="concurrency", y="throughput_mean", hue="engine",
        row="input_len_category", col="output_len_tokens",
        kind="line", marker="o", palette=PALETTE, height=3.5, aspect=1.3,
        facet_kws={"sharey": False},
    )
    g.set(xscale="log")
    for ax in g.axes.flat:
        ax.set_xlabel("Concurrency (batch size)")
        ax.set_ylabel("Throughput (tok/s)")
    g.figure.suptitle("Throughput vs. Concurrency: vLLM vs HuggingFace", y=1.03, fontsize=14)
    path = out_dir / "throughput_vs_concurrency.png"
    g.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved {path}")


def plot_memory(agg: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(9, 5.5))
    sns.barplot(data=agg, x="concurrency", y="peak_mem_mean", hue="engine", palette=PALETTE)
    plt.title("Peak GPU Memory Usage vs. Concurrency", fontsize=14)
    plt.xlabel("Concurrency (batch size)")
    plt.ylabel("Peak memory allocated (GB)")
    plt.tight_layout()
    path = out_dir / "memory_comparison.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved {path}")


def write_latency_table(agg: pd.DataFrame, out_dir: Path, concurrency_levels) -> None:
    subset = agg[agg["concurrency"].isin(concurrency_levels)].copy()
    subset = subset.sort_values(["input_len_category", "output_len_tokens", "concurrency", "engine"])
    cols = ["engine", "input_len_category", "output_len_tokens", "concurrency",
            "latency_p50_mean", "latency_p90_mean", "latency_p99_mean", "n_trials"]
    table = subset[cols].round(3)

    csv_path = out_dir / "latency_percentile_table.csv"
    table.to_csv(csv_path, index=False)

    md_path = out_dir / "latency_percentile_table.md"
    try:
        md_text = table.to_markdown(index=False)
    except ImportError:
        md_text = table.to_string(index=False)
    md_path.write_text(md_text)

    print(f"Saved {csv_path} and {md_path}\n")
    print("Latency percentile table:\n")
    print(table.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/benchmark_results.csv")
    parser.add_argument("--out-dir", default="results/plots")
    parser.add_argument("--latency-concurrencies", type=str, default=None,
                         help="comma list of concurrency levels for the latency table; "
                              "defaults to the smallest/median/largest values present in the data")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    df = load(args.csv)
    agg = aggregate(df)

    if args.latency_concurrencies:
        levels = [int(c) for c in args.latency_concurrencies.split(",")]
    else:
        uniq = sorted(agg["concurrency"].unique())
        levels = sorted({uniq[0], uniq[len(uniq) // 2], uniq[-1]})

    plot_throughput(agg, out_dir)
    plot_memory(agg, out_dir)
    write_latency_table(agg, out_dir, levels)


if __name__ == "__main__":
    main()
