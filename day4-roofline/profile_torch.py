"""
GPU busy fraction + kernel-category time breakdown using torch.profiler.

No nsys, no apt, no root/SYS_ADMIN capability required — uses PyTorch's own
CUPTI-backed profiler, which ships with torch and works under normal
container permissions.

Writes nsys_summary.json in the same schema plot_hero.py expects:
    {"gpu_busy_fraction": {...}, "time_breakdown": {...}}

Usage:
    python profile_torch.py --model Qwen/Qwen2.5-7B --output nsys_summary.json
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from bench_decode import _forward, load_model

CATEGORY_PATTERNS = [
    ("attention", re.compile(r"attn|attention|sdpa|scaled_dot_product|fmha|flash", re.I)),
    ("linear",    re.compile(r"linear|matmul|\bmm\b|addmm|bmm|gemm", re.I)),
    ("norm_rope", re.compile(r"norm|rope|rotary|rms", re.I)),
    ("sampling",  re.compile(r"argmax|topk|softmax|sort|multinomial|embedding", re.I)),
]


def categorize(name: str) -> str:
    for cat, pat in CATEGORY_PATTERNS:
        if pat.search(name):
            return cat
    return "other"


def _cuda_time(event) -> float:
    for attr in ("self_cuda_time_total", "self_device_time_total"):
        v = getattr(event, attr, None)
        if v is not None:
            return float(v)
    return 0.0


@torch.no_grad()
def profile_case(model, batch_size: int, context_length: int,
                  warmup_steps: int = 10, profiled_steps: int = 20):
    device = next(model.parameters()).device

    input_ids = torch.randint(100, 10_000, (batch_size, context_length),
                               device=device, dtype=torch.long)
    out = _forward(model, input_ids)
    past_kv = out.past_key_values
    next_token = out.logits[:, -1:, :].argmax(dim=-1)
    del out, input_ids
    torch.cuda.synchronize()

    for _ in range(warmup_steps):
        out = _forward(model, next_token, past_kv)
        past_kv = out.past_key_values
        next_token = out.logits[:, -1:, :].argmax(dim=-1)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        start = time.perf_counter()
        for _ in range(profiled_steps):
            out = _forward(model, next_token, past_kv)
            past_kv = out.past_key_values
            next_token = out.logits[:, -1:, :].argmax(dim=-1)
        torch.cuda.synchronize()
        wall_s = time.perf_counter() - start

    events = prof.key_averages()
    total_cuda_us = sum(_cuda_time(e) for e in events)
    busy_pct = min(100.0, 100.0 * (total_cuda_us / 1e6) / wall_s) if wall_s > 0 else 0.0

    cat_totals: dict[str, float] = {}
    for e in events:
        cat = categorize(e.key)
        cat_totals[cat] = cat_totals.get(cat, 0.0) + _cuda_time(e)
    grand = sum(cat_totals.values()) or 1.0
    breakdown = {cat: round(100.0 * v / grand, 1) for cat, v in cat_totals.items()}

    del past_kv, next_token
    torch.cuda.empty_cache()
    return round(busy_pct, 1), breakdown


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B")
    p.add_argument("--busy-batches", default="1,4,16,64",
                   help="Batch sizes to profile busy-fraction at (fixed ctx)")
    p.add_argument("--busy-ctx", type=int, default=128)
    p.add_argument("--breakdown-ctx", default="128,512,2048,8192",
                   help="Context lengths to profile breakdown at (fixed batch)")
    p.add_argument("--breakdown-batch", type=int, default=1)
    p.add_argument("--output", default="nsys_summary.json")
    args = p.parse_args()

    model, _ = load_model(args.model)

    gpu_busy_fraction: dict[str, float] = {}
    for b in [int(x) for x in args.busy_batches.split(",")]:
        try:
            busy, _ = profile_case(model, b, args.busy_ctx)
            gpu_busy_fraction[str(b)] = busy
            print(f"batch={b}, ctx={args.busy_ctx}: busy={busy}%")
        except torch.cuda.OutOfMemoryError:
            print(f"batch={b}: OOM, skipped")
            torch.cuda.empty_cache()

    time_breakdown: dict[str, dict[str, float]] = {}
    for c in [int(x) for x in args.breakdown_ctx.split(",")]:
        try:
            busy, breakdown = profile_case(model, args.breakdown_batch, c)
            time_breakdown[str(c)] = breakdown
            print(f"batch={args.breakdown_batch}, ctx={c}: {breakdown}")
            if str(args.breakdown_batch) not in gpu_busy_fraction and c == args.busy_ctx:
                gpu_busy_fraction[str(args.breakdown_batch)] = busy
        except torch.cuda.OutOfMemoryError:
            print(f"ctx={c}: OOM, skipped")
            torch.cuda.empty_cache()

    Path(args.output).write_text(json.dumps(
        {"gpu_busy_fraction": gpu_busy_fraction, "time_breakdown": time_breakdown},
        indent=2,
    ))
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
