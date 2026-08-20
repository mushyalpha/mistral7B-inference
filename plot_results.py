"""
Builds the report deliverables from results/benchmark_results.csv:
  1. A bold, single-panel "hero" chart for sharing (the money chart, one
     representative scenario, annotated with the peak speedup).
  2. The full throughput vs. concurrency plot, faceted by input-length
     category and output length so prefill vs. decode effects are visible
     separately.
  3. A latency percentile (p50/p90/p99) table at a few concurrency levels.
  4. A peak-GPU-memory comparison plot.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DISPLAY_NAMES = {"hf": "HuggingFace", "vllm": "vLLM"}
PALETTE = {"HuggingFace": "#ff7f0e", "vLLM": "#1f77b4"}


def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise SystemExit(f"{csv_path} is empty -- run benchmark.py for both --engine hf and --engine vllm first.")
    engines = set(df["engine"].unique())
    if engines != {"hf", "vllm"}:
        print(
            f"[plot_results] WARNING: CSV only contains engine(s) {sorted(engines)}. "
            "Charts need BOTH `python benchmark.py --engine hf` and `--engine vllm` to have "
            "run into this file before a real comparison is possible."
        )
    return df


def _with_display_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Engine"] = df["engine"].map(DISPLAY_NAMES).fillna(df["engine"])
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
    return _with_display_names(agg)


def plot_hero_chart(agg: pd.DataFrame, out_dir: Path) -> None:
    """One bold panel, built for sharing: throughput vs. concurrency at the
    single most demanding scenario tested (longest input + longest output),
    with the peak speedup annotated directly on the chart."""
    if agg.empty or agg["Engine"].nunique() < 2:
        print("[plot_results] Skipping hero_chart.png: need both `hf` and `vllm` rows in the CSV.")
        return

    input_choice = "long" if "long" in agg["input_len_category"].unique() else sorted(agg["input_len_category"].unique())[0]
    output_choice = agg["output_len_tokens"].max()
    subset = agg[(agg["input_len_category"] == input_choice) & (agg["output_len_tokens"] == output_choice)]
    if subset["Engine"].nunique() < 2:
        print("[plot_results] Skipping hero_chart.png: chosen scenario is missing one engine's data.")
        return

    with sns.plotting_context("talk"), sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(9, 6.5))
        sns.lineplot(
            data=subset, x="concurrency", y="throughput_mean", hue="Engine", hue_order=["HuggingFace", "vLLM"],
            palette=PALETTE, marker="o", linewidth=3.5, markersize=11, ax=ax,
        )
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(subset["concurrency"].unique()))
        ax.set_xticklabels([str(c) for c in sorted(subset["concurrency"].unique())])
        ax.set_xlabel("Concurrent requests (batch size)")
        ax.set_ylabel("Throughput (tokens/sec)")
        ax.set_title(
            f"vLLM vs. HuggingFace Transformers - Mistral-7B\n"
            f"({input_choice} prompts, {output_choice}-token generations)",
            fontsize=16, fontweight="bold", pad=14,
        )
        ax.margins(y=0.18)  # headroom so the top data point + its annotation clear the title

        max_conc = subset["concurrency"].max()
        at_max = subset[subset["concurrency"] == max_conc].set_index("Engine")["throughput_mean"]
        if "vLLM" in at_max.index and "HuggingFace" in at_max.index and at_max["HuggingFace"] > 0:
            speedup = at_max["vLLM"] / at_max["HuggingFace"]
            ax.annotate(
                f"{speedup:.1f}x faster\nat batch {max_conc}",
                xy=(max_conc, at_max["vLLM"]),
                xytext=(-150, -55), textcoords="offset points",
                fontsize=15, fontweight="bold", color=PALETTE["vLLM"],
                arrowprops=dict(arrowstyle="-|>", color=PALETTE["vLLM"], lw=2),
            )

        ax.legend(title=None, loc="upper left", frameon=True)
        sns.despine()
        fig.tight_layout()
        path = out_dir / "hero_chart.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
    print(f"Saved {path}")


def plot_throughput(agg: pd.DataFrame, out_dir: Path) -> None:
    g = sns.relplot(
        data=agg, x="concurrency", y="throughput_mean", hue="Engine", hue_order=["HuggingFace", "vLLM"],
        row="input_len_category", col="output_len_tokens",
        kind="line", marker="o", linewidth=2.5, markersize=8,
        palette=PALETTE, height=3.6, aspect=1.25,
        facet_kws={"sharey": False, "margin_titles": True},
    )
    g.set(xscale="log")
    g.set_titles(row_template="{row_name} input", col_template="{col_name}-tok output")
    for ax in g.axes.flat:
        ax.set_xlabel("Concurrency (batch size)")
        ax.set_ylabel("Throughput (tok/s)")

    g.figure.subplots_adjust(top=0.85, wspace=0.3, hspace=0.35)
    sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, 0.98), ncol=2, title=None, frameon=False)
    g.figure.suptitle("Throughput vs. Concurrency: vLLM vs HuggingFace", y=1.05, fontsize=15, fontweight="bold")

    path = out_dir / "throughput_vs_concurrency.png"
    g.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved {path}")


def plot_memory(agg: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(9, 5.5))
    sns.barplot(data=agg, x="concurrency", y="peak_mem_mean", hue="Engine", hue_order=["HuggingFace", "vLLM"], palette=PALETTE)
    plt.title("Peak GPU Memory Usage vs. Concurrency", fontsize=14)
    plt.xlabel("Concurrency (batch size)")
    plt.ylabel("Peak memory in use (GB)")
    plt.legend(title=None)
    plt.tight_layout()
    path = out_dir / "memory_comparison.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved {path}")

    if (agg["peak_mem_mean"] == 0).all():
        print(
            "[plot_results] WARNING: every peak_mem_mean value is 0.0 -- re-run benchmark.py with the "
            "updated bench_utils.track_peak_memory_gb (nvidia-smi based) before trusting this chart."
        )


def write_latency_table(agg: pd.DataFrame, out_dir: Path, concurrency_levels) -> None:
    subset = agg[agg["concurrency"].isin(concurrency_levels)].copy()
    subset = subset.sort_values(["input_len_category", "output_len_tokens", "concurrency", "Engine"])
    cols = ["Engine", "input_len_category", "output_len_tokens", "concurrency",
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

    plot_hero_chart(agg, out_dir)
    plot_throughput(agg, out_dir)
    plot_memory(agg, out_dir)
    write_latency_table(agg, out_dir, levels)


if __name__ == "__main__":
    main()
