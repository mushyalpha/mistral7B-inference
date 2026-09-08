from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# GPU helpers

def get_sm_clock_mhz() -> int:
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
    except Exception:
        pass
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.current.sm",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return int(r.stdout.strip().split("\n")[0])
    except Exception:
        return -1


def get_gpu_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"


# Model loading

def load_model(model_id: str, dtype: torch.dtype = torch.bfloat16):
    print(f"Loading {model_id} ({dtype}) …")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {n_params / 1e9:.2f} B")
    print(f"  Vocab      : {model.config.vocab_size:,}")
    print(f"  GPU        : {get_gpu_name()}")
    return model, tokenizer, n_params


# Prefill forward — logits_to_keep=1 to dodge the vocab OOM trap

def prefill_forward(model, input_ids):
    """
    Single forward pass with num_logits_to_keep=1.

    WHY: Qwen2.5 vocab = 152,064.  Full logits at 8K context =
    8192 × 152064 × 2 B ≈ 2.5 GB per sequence.  Softmax upcasts to FP32
    → ~5 GB.  At batch 8 that's 40 GB of logits alone.
    """
    try:
        out = model(input_ids, use_cache=False, num_logits_to_keep=1)
    except TypeError:
        # Older transformers without num_logits_to_keep
        out = model(input_ids, use_cache=False)
    return out


# Core benchmark

def benchmark_prefill(
    model: torch.nn.Module,
    n_params: int,
    batch_size: int,
    seq_length: int,
    warmup_runs: int = 3,
    timed_runs: int = 10,
    enable_profile: bool = False,
) -> dict:
    """
    Time prefill: one forward pass over [B, L] with no KV cache.
    """
    device = next(model.parameters()).device

    input_ids = torch.randint(100, 10_000, (batch_size, seq_length),
                              device=device, dtype=torch.long)

    # Warmup (untimed)
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = prefill_forward(model, input_ids)
        torch.cuda.synchronize()

    sm_clock_before = get_sm_clock_mhz()

    # Timed runs
    if enable_profile:
        torch.cuda.profiler.start()

    timings_ms: list[float] = []
    for _ in range(timed_runs):
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_ev.record()
        with torch.no_grad():
            _ = prefill_forward(model, input_ids)
        end_ev.record()
        torch.cuda.synchronize()

        timings_ms.append(start_ev.elapsed_time(end_ev))

    if enable_profile:
        torch.cuda.profiler.stop()

    sm_clock_after = get_sm_clock_mhz()

    del input_ids
    torch.cuda.empty_cache()

    t = np.array(timings_ms)

    # Prefill FLOPs ≈ 2 × P × tokens  (each param does one multiply-add per token)
    total_tokens = batch_size * seq_length
    flops = 2 * n_params * total_tokens
    median_s = np.median(t) / 1e3
    tflops = (flops / median_s) / 1e12 if median_s > 0 else 0

    return {
        "batch_size": batch_size,
        "seq_length": seq_length,
        "total_tokens": total_tokens,
        "warmup_runs": warmup_runs,
        "timed_runs": timed_runs,
        # Timings
        "median_ms": float(np.median(t)),
        "mean_ms": float(np.mean(t)),
        "p99_ms": float(np.percentile(t, 99)),
        "min_ms": float(np.min(t)),
        "max_ms": float(np.max(t)),
        "std_ms": float(np.std(t)),
        # Compute
        "tflops": round(tflops, 1),
        "flops": flops,
        "tokens_per_ms": round(total_tokens / np.median(t), 1),
        # Clock
        "sm_clock_mhz_before": sm_clock_before,
        "sm_clock_mhz_after": sm_clock_after,
        # Raw
        "all_timings_ms": [round(x, 4) for x in t.tolist()],
    }


# CLI

def parse_args():
    p = argparse.ArgumentParser(
        description="Prefill latency benchmark (compute roofline)",
    )
    p.add_argument("--model", default="Qwen/Qwen2.5-7B")
    p.add_argument("--batch-sizes", default="1,2,4,8",
                   help="Comma-separated batch sizes (keep small — prefill story is seq len)")
    p.add_argument("--seq-lengths", default="128,512,2048,8192",
                   help="Comma-separated sequence lengths")
    p.add_argument("--warmup-runs", type=int, default=3)
    p.add_argument("--timed-runs", type=int, default=10)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--output", default="results_prefill.json")
    p.add_argument("--profile", action="store_true",
                   help="Wrap timed region with cudaProfilerApi for nsys capture")
    p.add_argument("--max-tokens-per-forward", type=int, default=100_000,
                   help="Skip combos where B×L exceeds this (OOM guard)")
    return p.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.", file=sys.stderr)
        sys.exit(1)

    dtype = getattr(torch, args.dtype)
    model, tokenizer, n_params = load_model(args.model, dtype)

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    seq_lengths = [int(x) for x in args.seq_lengths.split(",")]

    # H100 BF16 peak: 990 TFLOPS (tensor core)
    peak_tflops = 990.0

    print(f"\n{'='*80}")
    print(f"  Prefill Benchmark — {args.model}")
    print(f"  Sweep: B ∈ {batch_sizes}, L ∈ {seq_lengths}")
    print(f"  Max tokens/forward guard: {args.max_tokens_per_forward:,}")
    print(f"  H100 BF16 peak: {peak_tflops} TFLOPS")
    print(f"{'='*80}\n")

    all_results = []

    for bs in batch_sizes:
        for sl in seq_lengths:
            total_tokens = bs * sl
            label = f"B={bs}, L={sl}"

            if total_tokens > args.max_tokens_per_forward:
                print(f"▸ {label} … ✗ skipped ({total_tokens:,} tokens > "
                      f"{args.max_tokens_per_forward:,} guard)")
                continue

            print(f"▸ {label} ({total_tokens:,} tokens) …", end="", flush=True)

            try:
                result = benchmark_prefill(
                    model, n_params, bs, sl,
                    warmup_runs=args.warmup_runs,
                    timed_runs=args.timed_runs,
                    enable_profile=args.profile,
                )
                all_results.append(result)

                mfu = (result["tflops"] / peak_tflops) * 100
                print(
                    f"  median {result['median_ms']:.2f} ms  "
                    f"| {result['tflops']:.1f} TFLOP/s  "
                    f"({mfu:.0f}% MFU)  "
                    f"| SM {result['sm_clock_mhz_after']} MHz"
                )
            except torch.cuda.OutOfMemoryError:
                print(f"  ✗ OOM — skipped")
                torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'─'*90}")
    print(f"{'Case':<16} {'Tokens':>8} {'Median ms':>10} {'TFLOP/s':>9} "
          f"{'MFU %':>7} {'tok/ms':>8} {'SM MHz':>8}")
    print(f"{'─'*90}")
    for r in all_results:
        label = f"B={r['batch_size']}, L={r['seq_length']}"
        mfu = (r["tflops"] / peak_tflops) * 100
        print(f"{label:<16} {r['total_tokens']:>8,} {r['median_ms']:>10.2f} "
              f"{r['tflops']:>9.1f} {mfu:>6.0f}% "
              f"{r['tokens_per_ms']:>8.1f} {r['sm_clock_mhz_after']:>8}")
    print(f"{'─'*90}")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_data = {
        "model": args.model,
        "gpu": get_gpu_name(),
        "dtype": args.dtype,
        "n_params": n_params,
        "results": all_results,
    }
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
