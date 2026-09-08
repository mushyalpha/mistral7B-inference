"""
Parse nsys .nsys-rep traces into nsys_summary.json for plot_hero.py.

Exports a per-kernel GPU trace via `nsys stats`, categorizes kernels by name
(attention / linear / norm_rope / sampling / other), and computes:
  - GPU busy fraction = sum(kernel durations) / (profiled wall-clock span)
  - Time breakdown = per-category share of total kernel time

Usage:
    python parse_nsys.py \\
        --busy 1=nsys_busy_b1.nsys-rep 4=nsys_busy_b4.nsys-rep 16=nsys_busy_b16.nsys-rep 64=nsys_busy_b64.nsys-rep \\
        --breakdown 128=nsys_busy_b1.nsys-rep 512=nsys_brk_c512.nsys-rep 2048=nsys_brk_c2048.nsys-rep 8192=nsys_brk_c8192.nsys-rep \\
        --output nsys_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

CATEGORY_PATTERNS = [
    ("attention", re.compile(r"flash|attn|attention|fmha|sdpa", re.I)),
    ("linear",    re.compile(r"gemm|cutlass|matmul|linear|cublas", re.I)),
    ("norm_rope", re.compile(r"norm|rope|rotary|rms", re.I)),
    ("sampling",  re.compile(r"argmax|topk|softmax|sort|multinomial|index_select", re.I)),
]


def categorize(name: str) -> str:
    for cat, pat in CATEGORY_PATTERNS:
        if pat.search(name):
            return cat
    return "other"


def find_gputrace_csv(nsys_rep: Path) -> Path | None:
    """Export a per-kernel trace CSV via `nsys stats`, trying known report names."""
    base = nsys_rep.with_suffix("")
    for report in ("cuda_gpu_trace", "gputrace"):
        existing = list(base.parent.glob(f"{base.name}_{report}*.csv"))
        if existing:
            return existing[0]
        try:
            subprocess.run(
                ["nsys", "stats", "--report", report, "--format", "csv",
                 "--output", str(base), str(nsys_rep)],
                capture_output=True, text=True, timeout=120, check=True,
            )
        except Exception as e:
            print(f"  ({report} export failed: {e})", file=sys.stderr)
            continue
        found = list(base.parent.glob(f"{base.name}_{report}*.csv"))
        if found:
            return found[0]
    return None


def load_kernels(csv_path: Path) -> list[tuple[int, int, str]]:
    """Return list of (start_ns, duration_ns, kernel_name)."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        cols = {c.lower(): c for c in (reader.fieldnames or [])}
        start_col = next((cols[c] for c in cols if "start" in c), None)
        dur_col = next((cols[c] for c in cols if "dur" in c), None)
        name_col = cols.get("name")
        if name_col is None:
            name_col = next(
                (cols[c] for c in cols if "name" in c and "process" not in c and "thread" not in c),
                None,
            )
        if not (start_col and dur_col and name_col):
            raise RuntimeError(f"Missing start/dur/name columns in {csv_path}: {reader.fieldnames}")
        for row in reader:
            try:
                start = int(float(row[start_col]))
                dur = int(float(row[dur_col]))
            except (ValueError, KeyError):
                continue
            rows.append((start, dur, row[name_col]))
    return rows


def busy_fraction(rows: list[tuple[int, int, str]]) -> float:
    if not rows:
        return 0.0
    total_dur = sum(d for _, d, _ in rows)
    span = max(s + d for s, d, _ in rows) - min(s for s, _, _ in rows)
    if span <= 0:
        return 0.0
    return min(100.0, 100.0 * total_dur / span)


def breakdown_pct(rows: list[tuple[int, int, str]]) -> dict[str, float]:
    totals: dict[str, int] = {}
    for _, dur, name in rows:
        cat = categorize(name)
        totals[cat] = totals.get(cat, 0) + dur
    grand = sum(totals.values()) or 1
    return {cat: round(100.0 * v / grand, 1) for cat, v in totals.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--busy", nargs="+", required=True,
                   help="batch=path.nsys-rep pairs")
    p.add_argument("--breakdown", nargs="+", required=True,
                   help="ctx=path.nsys-rep pairs")
    p.add_argument("--output", default="nsys_summary.json")
    args = p.parse_args()

    gpu_busy_fraction: dict[str, float] = {}
    for pair in args.busy:
        batch, path = pair.split("=", 1)
        rep = Path(path)
        if not rep.exists():
            print(f"SKIP batch={batch}: {path} not found", file=sys.stderr)
            continue
        csv_path = find_gputrace_csv(rep)
        if csv_path is None:
            print(f"SKIP batch={batch}: could not export gputrace for {path}", file=sys.stderr)
            continue
        rows = load_kernels(csv_path)
        gpu_busy_fraction[batch] = round(busy_fraction(rows), 1)
        print(f"batch={batch}: busy={gpu_busy_fraction[batch]}%  ({len(rows)} kernel launches)")

    time_breakdown: dict[str, dict[str, float]] = {}
    for pair in args.breakdown:
        ctx, path = pair.split("=", 1)
        rep = Path(path)
        if not rep.exists():
            print(f"SKIP ctx={ctx}: {path} not found", file=sys.stderr)
            continue
        csv_path = find_gputrace_csv(rep)
        if csv_path is None:
            print(f"SKIP ctx={ctx}: could not export gputrace for {path}", file=sys.stderr)
            continue
        rows = load_kernels(csv_path)
        time_breakdown[ctx] = breakdown_pct(rows)
        print(f"ctx={ctx}: {time_breakdown[ctx]}")

    if not gpu_busy_fraction and not time_breakdown:
        print("\nERROR: no traces parsed successfully. nsys_summary.json not written.", file=sys.stderr)
        sys.exit(1)

    Path(args.output).write_text(json.dumps(
        {"gpu_busy_fraction": gpu_busy_fraction, "time_breakdown": time_breakdown},
        indent=2,
    ))
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
