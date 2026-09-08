#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# Style constants — designed for Twitter (dark bg, large text, bold)

COLORS = {
    "bg":      "#0d1117",
    "surface": "#161b22",
    "text":    "#e6edf3",
    "muted":   "#7d8590",
    "grid":    "#21262d",
    # Context-length palette (cool → hot)
    "ctx_128":  "#58a6ff",
    "ctx_512":  "#3fb950",
    "ctx_2048": "#d29922",
    "ctx_8192": "#f85149",
    # Breakdown palette
    "attention":  "#f85149",
    "linear":     "#58a6ff",
    "norm_rope":  "#3fb950",
    "other":      "#7d8590",
    # Energy
    "energy_primary": "#bc8cff",
    "energy_cost":    "#d29922",
    # Busy fraction
    "busy":  "#58a6ff",
    "idle":  "#21262d",
}

CTX_COLORS = {
    128:  COLORS["ctx_128"],
    512:  COLORS["ctx_512"],
    2048: COLORS["ctx_2048"],
    8192: COLORS["ctx_8192"],
}

BATCH_SIZES = [1, 4, 16, 64]
CTX_LENGTHS = [128, 512, 2048, 8192]


def apply_dark_style(fig, axes):
    """Apply consistent dark theme to figure and axes."""
    fig.patch.set_facecolor(COLORS["bg"])
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(COLORS["surface"])
        ax.tick_params(colors=COLORS["text"], labelsize=11)
        ax.xaxis.label.set_color(COLORS["text"])
        ax.yaxis.label.set_color(COLORS["text"])
        ax.title.set_color(COLORS["text"])
        for spine in ax.spines.values():
            spine.set_color(COLORS["grid"])
        ax.grid(True, alpha=0.15, color=COLORS["muted"], linewidth=0.5)


# Analytical model (for roofline ceilings + mock data)

WEIGHT_BYTES = 15.2e9          # Qwen2.5-7B BF16
KV_BYTES_PER_TOKEN = 57_344    # 2 × 28 × 4 × 128 × 2
HBM_BW = 3.35e12              # H100 SXM5


def roofline_tok_s(batch: int, ctx: int) -> float:
    total_bytes = WEIGHT_BYTES + batch * ctx * KV_BYTES_PER_TOKEN
    floor_s = total_bytes / HBM_BW
    return batch / floor_s


# Mock data generators — predicted + realistic engineering loss

def mock_efficiency(batch: int, ctx: int) -> float:
    """Plausible fraction-of-roofline achieved. The story:
    B=1 is launch-bound (~45%), B=64 approaches BW-bound (~88%).
    Long context degrades slightly due to inefficient attention."""
    base = {1: 0.45, 4: 0.58, 16: 0.73, 64: 0.88}[batch]
    ctx_penalty = max(0, (ctx - 512) / 8192) * 0.08
    return base - ctx_penalty


def generate_mock_decode() -> list[dict]:
    """Mock decode results matching results_decode.json schema."""
    results = []
    for bs in BATCH_SIZES:
        for ctx in CTX_LENGTHS:
            floor = roofline_tok_s(bs, ctx)
            eff = mock_efficiency(bs, ctx)
            measured = floor * eff

            # Power model: idle ~120W, scales with utilisation
            power = 120 + 580 * (eff * 0.85)  # rough curve
            tok_j = measured / power if power > 0 else 0

            results.append({
                "batch_size": bs,
                "context_length": ctx,
                "measured_tok_s": round(measured, 1),
                "floor_tok_s": round(floor, 1),
                "pct_of_floor": round(eff * 100, 1),
                "avg_power_w": round(power, 0),
                "tokens_per_joule": round(tok_j, 3),
                "joules_per_1m_tokens": round(1e6 / tok_j, 1) if tok_j > 0 else 0,
            })
    return results


def generate_mock_busy() -> dict[int, float]:
    """GPU busy fraction by batch size (from nsys)."""
    return {
        1:  38.0,
        4:  58.0,
        16: 78.0,
        64: 93.0,
    }


def generate_mock_breakdown() -> dict[int, dict[str, float]]:
    """Time breakdown by context length (from nsys kernel analysis)."""
    return {
        128:  {"attention": 5,  "linear": 82, "norm_rope": 10, "other": 3},
        512:  {"attention": 14, "linear": 72, "norm_rope": 11, "other": 3},
        2048: {"attention": 30, "linear": 55, "norm_rope": 11, "other": 4},
        8192: {"attention": 50, "linear": 35, "norm_rope": 11, "other": 4},
    }


# Data loading from real files

def load_decode_results(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text())
    return data["results"]


def load_nsys_summary(path: str) -> tuple[dict[int, float], dict[int, dict]]:
    """
    Expected nsys_summary.json format:
    {
      "gpu_busy_fraction": {"1": 38.0, "4": 58.0, "16": 78.0, "64": 93.0},
      "time_breakdown": {
        "128":  {"attention": 5,  "linear": 82, "norm_rope": 10, "other": 3},
        "512":  {"attention": 14, "linear": 72, "norm_rope": 11, "other": 3},
        "2048": {"attention": 30, "linear": 55, "norm_rope": 11, "other": 4},
        "8192": {"attention": 50, "linear": 35, "norm_rope": 11, "other": 4}
      }
    }
    """
    data = json.loads(Path(path).read_text())
    busy = {int(k): v for k, v in data["gpu_busy_fraction"].items()}
    breakdown = {int(k): v for k, v in data["time_breakdown"].items()}
    return busy, breakdown


# Chart 1: Decode tok/s vs batch size (roofline comparison)

def plot_tok_s(ax, results: list[dict], is_mock: bool = False):
    ax.set_title("Decode Throughput vs. Roofline", fontsize=14, fontweight="bold",
                 pad=10)

    # Group by context length
    by_ctx: dict[int, tuple[list, list, list]] = {}
    for r in results:
        ctx = r["context_length"]
        if ctx not in by_ctx:
            by_ctx[ctx] = ([], [], [])
        by_ctx[ctx][0].append(r["batch_size"])
        by_ctx[ctx][1].append(r["measured_tok_s"])
        by_ctx[ctx][2].append(r["floor_tok_s"])

    for ctx in sorted(by_ctx.keys()):
        bs_list, measured, floor = by_ctx[ctx]
        color = CTX_COLORS.get(ctx, COLORS["muted"])

        # Roofline ceiling (dashed)
        ax.plot(bs_list, floor, "--", color=color, alpha=0.45, linewidth=1.5)
        # Measured (solid + markers)
        ax.plot(bs_list, measured, "-o", color=color, linewidth=2.2,
                markersize=7, markeredgewidth=0, label=f"ctx={ctx:,}",
                zorder=5)

    # Annotate the gap at B=1
    if results:
        b1 = [r for r in results if r["batch_size"] == 1 and r["context_length"] == 128]
        if b1:
            r = b1[0]
            ax.annotate(
                f'{r["pct_of_floor"]:.0f}% of\nroofline',
                xy=(1, r["measured_tok_s"]),
                xytext=(2.2, r["measured_tok_s"] * 0.4),
                fontsize=9, color=COLORS["muted"],
                arrowprops=dict(arrowstyle="->", color=COLORS["muted"],
                                lw=1.2),
                ha="left",
            )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(BATCH_SIZES)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Batch size", fontsize=12)
    ax.set_ylabel("tok/s", fontsize=12)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.3,
              edgecolor=COLORS["grid"], facecolor=COLORS["surface"],
              labelcolor=COLORS["text"])

    # Add "ROOFLINE" label on the dashed line
    ax.text(0.97, 0.97, "── roofline ceiling", transform=ax.transAxes,
            fontsize=8, color=COLORS["muted"], ha="right", va="top",
            fontstyle="italic")

    tag = "(predicted)" if is_mock else "(measured)"
    ax.text(0.97, 0.03, tag, transform=ax.transAxes, fontsize=8,
            color=COLORS["muted"], ha="right", va="bottom", fontstyle="italic")


# Chart 2: GPU busy fraction vs batch size

def plot_busy_fraction(ax, busy: dict[int, float], is_mock: bool = False):
    ax.set_title("GPU Busy Fraction", fontsize=14, fontweight="bold", pad=10)

    batches = sorted(busy.keys())
    fractions = [busy[b] for b in batches]
    idle = [100 - f for f in fractions]
    x = np.arange(len(batches))
    width = 0.55

    # Stacked: busy on bottom, idle on top
    bars_busy = ax.bar(x, fractions, width, color=COLORS["busy"], alpha=0.85,
                       label="GPU busy", edgecolor="none")
    bars_idle = ax.bar(x, idle, width, bottom=fractions, color=COLORS["idle"],
                       alpha=0.5, label="Idle (launch overhead + Python)",
                       edgecolor="none")

    # Percentage labels on busy bars
    for i, (b, f) in enumerate(zip(batches, fractions)):
        ax.text(i, f / 2, f"{f:.0f}%", ha="center", va="center",
                fontsize=12, fontweight="bold", color=COLORS["text"])
        if 100 - f > 8:
            ax.text(i, f + (100 - f) / 2, f"{100 - f:.0f}%\nidle",
                    ha="center", va="center", fontsize=9,
                    color=COLORS["muted"])

    # Dramatic annotation at batch 1
    if 1 in busy and busy[1] < 50:
        ax.annotate(
            f"GPU idle {100 - busy[1]:.0f}% of the time\n→ launch-bound, not memory-bound",
            xy=(0, busy[1] + (100 - busy[1]) * 0.6),
            xytext=(-0.1, 110),
            fontsize=8, color=COLORS["muted"],
            arrowprops=dict(arrowstyle="->", color=COLORS["muted"], lw=1),
            ha="left", va="bottom",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"B={b}" for b in batches])
    ax.set_ylabel("% of decode step", fontsize=12)
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])

    tag = "(predicted)" if is_mock else "(from nsys)"
    ax.text(0.97, 0.03, tag, transform=ax.transAxes, fontsize=8,
            color=COLORS["muted"], ha="right", va="bottom", fontstyle="italic")


# Chart 3: Time breakdown vs context length (stacked)

def plot_breakdown(ax, breakdown: dict[int, dict], is_mock: bool = False):
    ax.set_title("Decode Step Breakdown", fontsize=14, fontweight="bold",
                 pad=10)

    ctx_list = sorted(breakdown.keys())
    x = np.arange(len(ctx_list))
    width = 0.55

    categories = ["linear", "attention", "norm_rope", "other"]
    cat_colors = {
        "attention":  COLORS["attention"],
        "linear":     COLORS["linear"],
        "norm_rope":  COLORS["norm_rope"],
        "other":      COLORS["other"],
    }
    cat_labels = {
        "attention": "Attention",
        "linear":    "Linear (GEMM)",
        "norm_rope": "Norm + RoPE",
        "other":     "Sampling / other",
    }

    bottoms = np.zeros(len(ctx_list))
    for cat in categories:
        vals = [breakdown[c].get(cat, 0) for c in ctx_list]
        ax.bar(x, vals, width, bottom=bottoms, color=cat_colors[cat],
               alpha=0.85, label=cat_labels[cat], edgecolor="none")

        # Label the segment if it's big enough
        for i, v in enumerate(vals):
            if v >= 10:
                ax.text(i, bottoms[i] + v / 2, f"{v:.0f}%",
                        ha="center", va="center", fontsize=9,
                        fontweight="bold", color=COLORS["text"])
        bottoms += np.array(vals)

    # Arrow showing attention growth
    attn_first = breakdown[ctx_list[0]].get("attention", 0)
    attn_last = breakdown[ctx_list[-1]].get("attention", 0)
    if attn_last > attn_first * 2:
        # Place annotation pointing to the red section of the last bar
        # Bottom of attention is linear (plus anything below it)
        bottom_of_attn = breakdown[ctx_list[-1]].get("linear", 0)
        mid_y_last = bottom_of_attn + (attn_last / 2)
        ax.annotate(
            f"Attention: {attn_first}% → {attn_last}%\n→ justifies Day 5",
            xy=(len(ctx_list) - 1, mid_y_last),
            xytext=(len(ctx_list) - 2.8, mid_y_last + 20),
            fontsize=8, color=COLORS["attention"],
            arrowprops=dict(arrowstyle="->", color=COLORS["attention"],
                            lw=1.2),
            ha="center", va="center", fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c:,}" for c in ctx_list])
    ax.set_xlabel("Context length (KV cache tokens)", fontsize=12)
    ax.set_ylabel("% of decode step", fontsize=12)
    ax.set_ylim(0, 108)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.3,
              edgecolor=COLORS["grid"], facecolor=COLORS["surface"],
              labelcolor=COLORS["text"], ncol=2)

    tag = "(predicted)" if is_mock else "(from nsys)"
    ax.text(0.97, 0.03, tag, transform=ax.transAxes, fontsize=8,
            color=COLORS["muted"], ha="right", va="bottom", fontstyle="italic")


# Chart 4: Tokens/joule vs batch size (dual axis: £/1M tokens)

def plot_energy(ax, results: list[dict], grid_price: float = 0.30,
                is_mock: bool = False):
    ax.set_title("Energy Efficiency", fontsize=14, fontweight="bold", pad=10)

    # Average across context lengths for the main energy story
    # (also plot individual points faintly)
    by_batch: dict[int, list] = {}
    for r in results:
        bs = r["batch_size"]
        if bs not in by_batch:
            by_batch[bs] = []
        by_batch[bs].append(r)

    batches = sorted(by_batch.keys())
    avg_tok_j = []
    all_tok_j_points = []

    for bs in batches:
        vals = [r["tokens_per_joule"] for r in by_batch[bs]]
        avg_tok_j.append(np.mean(vals))
        for r in by_batch[bs]:
            all_tok_j_points.append((bs, r["tokens_per_joule"], r["context_length"]))

    # Individual points by context length
    for bs_val, tj, ctx in all_tok_j_points:
        color = CTX_COLORS.get(ctx, COLORS["muted"])
        ax.scatter(bs_val, tj, color=color, s=40, alpha=0.6, zorder=4,
                   edgecolors="none")

    # Average line
    ax.plot(batches, avg_tok_j, "-s", color=COLORS["energy_primary"],
            linewidth=2.5, markersize=9, markeredgewidth=0, zorder=5,
            label="tok/J (avg)")

    # Annotate the improvement factor
    if len(avg_tok_j) >= 2:
        ratio = avg_tok_j[-1] / avg_tok_j[0]
        ax.annotate(
            f"{ratio:.0f}× more\nenergy efficient",
            xy=(batches[-1], avg_tok_j[-1]),
            xytext=(batches[-1] * 0.35, avg_tok_j[-1] * 0.85),
            fontsize=10, color=COLORS["energy_primary"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COLORS["energy_primary"],
                            lw=1.5),
            ha="center", va="center",
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(BATCH_SIZES)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Batch size", fontsize=12)
    ax.set_ylabel("tok / joule", fontsize=12, color=COLORS["energy_primary"])
    ax.tick_params(axis="y", labelcolor=COLORS["energy_primary"])

    # Secondary axis: £/1M tokens
    ax2 = ax.twinx()
    ax2.set_facecolor("none")

    cost_per_1m = []
    for tj in avg_tok_j:
        if tj > 0:
            kwh = 1e6 / (tj * 3_600_000)
            cost_per_1m.append(kwh * grid_price)
        else:
            cost_per_1m.append(0)

    ax2.plot(batches, cost_per_1m, "--^", color=COLORS["energy_cost"],
             linewidth=1.8, markersize=7, markeredgewidth=0, alpha=0.8,
             label=f"£/1M tok @ {grid_price*100:.0f}p/kWh")

    ax2.set_ylabel(f"£ / 1M tokens ({grid_price*100:.0f}p/kWh)", fontsize=11,
                   color=COLORS["energy_cost"])
    ax2.tick_params(axis="y", labelcolor=COLORS["energy_cost"], labelsize=10)
    ax2.set_yscale("log")
    for spine in ax2.spines.values():
        spine.set_color(COLORS["grid"])
    ax2.spines["right"].set_color(COLORS["energy_cost"])
    ax2.spines["right"].set_alpha(0.4)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8,
              loc="center left", framealpha=0.3,
              edgecolor=COLORS["grid"], facecolor=COLORS["surface"],
              labelcolor=COLORS["text"])

    tag = "(predicted)" if is_mock else "(measured)"
    ax.text(0.97, 0.03, tag, transform=ax.transAxes, fontsize=8,
            color=COLORS["muted"], ha="right", va="bottom", fontstyle="italic")

    return ax2  # caller needs this to style it


# Hero chart assembly (2×2 panel)

def make_hero_chart(
    results: list[dict],
    busy: dict[int, float],
    breakdown: dict[int, dict],
    output_dir: Path,
    is_mock: bool = False,
    grid_price: float = 0.30,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Individual charts
    for i, (plot_fn, fname, kwargs) in enumerate([
        (plot_tok_s,          "chart_1_tok_s.png",     {"results": results}),
        (plot_busy_fraction,  "chart_2_busy.png",      {"busy": busy}),
        (plot_breakdown,      "chart_3_breakdown.png", {"breakdown": breakdown}),
        (plot_energy,         "chart_4_energy.png",    {"results": results, "grid_price": grid_price}),
    ], 1):
        fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
        apply_dark_style(fig, [ax])
        result = plot_fn(ax, is_mock=is_mock, **kwargs)
        # For the energy chart, style the twin axis too
        if result is not None and hasattr(result, "spines"):
            result.tick_params(colors=COLORS["text"])
        fig.tight_layout()
        fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        print(f"  ✓ Saved {fname}")

    # Combined 2×2 hero chart
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=150)
    apply_dark_style(fig, axes.flat)

    plot_tok_s(axes[0, 0], results, is_mock=is_mock)
    plot_busy_fraction(axes[0, 1], busy, is_mock=is_mock)
    plot_breakdown(axes[1, 0], breakdown, is_mock=is_mock)
    ax2 = plot_energy(axes[1, 1], results, grid_price=grid_price, is_mock=is_mock)
    if ax2 is not None:
        ax2.tick_params(colors=COLORS["text"])

    # Suptitle
    mock_tag = "  ·  PREDICTED (pre-measurement)" if is_mock else ""
    fig.suptitle(
        f"Day 4/45 · Inference Engineering{mock_tag}\n"
        "Qwen2.5-7B BF16  ·  H100 SXM  ·  3.35 TB/s",
        fontsize=18, fontweight="bold", color=COLORS["text"],
        y=0.98,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    hero_path = output_dir / "hero_chart.png"
    fig.savefig(hero_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"  ✓ Saved hero_chart.png")

    return hero_path


# CLI

def parse_args():
    p = argparse.ArgumentParser(
        description="Day 4/45 hero chart for Twitter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mock", action="store_true",
                   help="Generate charts with predicted data (design preview)")
    p.add_argument("--data", type=str, default=None,
                   help="Path to results_decode.json")
    p.add_argument("--nsys", type=str, default=None,
                   help="Path to nsys_summary.json (for charts 2 & 3)")
    p.add_argument("--output-dir", type=str,
                   default="charts",
                   help="Directory to save chart PNGs")
    p.add_argument("--grid-price", type=float, default=0.30,
                   help="Electricity price in £/kWh for cost axis (default: 0.30)")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.mock and not args.data:
        print("ERROR: specify --mock or --data FILE", file=sys.stderr)
        sys.exit(1)

    is_mock = args.mock or args.data is None

    # Load or generate data
    if args.data:
        print(f"Loading decode data from {args.data}")
        results = load_decode_results(args.data)
        is_mock = False
    else:
        print("Generating mock data from analytical predictions …")
        results = generate_mock_decode()

    if args.nsys:
        print(f"Loading nsys summary from {args.nsys}")
        busy, breakdown = load_nsys_summary(args.nsys)
    else:
        if not is_mock:
            print("⚠ No --nsys file provided. Charts 2 & 3 will use ILLUSTRATIVE PLACEHOLDER "
                  "data (NOT measured, NOT analytically derived) — do not publish these two "
                  "panels without real data from profile_torch.py or nsys.")
        busy = generate_mock_busy()
        breakdown = generate_mock_breakdown()

    output_dir = Path(args.output_dir)
    print(f"\nGenerating charts → {output_dir}/")

    hero_path = make_hero_chart(
        results=results,
        busy=busy,
        breakdown=breakdown,
        output_dir=output_dir,
        is_mock=is_mock,
        grid_price=args.grid_price,
    )

    print(f"\n{'='*60}")
    print(f"  Hero chart: {hero_path}")
    print(f"  Individual charts: {output_dir}/chart_*.png")
    if is_mock:
        print(f"\n  ⚠ These use PREDICTED data.")
        print(f"    Re-run with --data results_decode.json after benchmarking.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
