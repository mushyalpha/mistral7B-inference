
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from energy import EnergyTracker, cost_per_million_tokens


# Analytical floor

@dataclass
class ModelSpec:
    """Architecture constants needed for roofline math."""
    n_params: int          # total parameter count
    n_layers: int
    n_kv_heads: int
    head_dim: int
    bytes_per_param: int   # 2 for BF16/FP16

    @property
    def weight_bytes(self) -> float:
        return self.n_params * self.bytes_per_param

    def kv_bytes_per_token(self) -> int:
        """K + V, all layers, one token."""
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * self.bytes_per_param

    def decode_floor_ms(self, batch: int, ctx_len: int, bw_tb_s: float) -> float:
        """Minimum decode-step latency (ms) — pure bandwidth floor."""
        total_bytes = self.weight_bytes + batch * ctx_len * self.kv_bytes_per_token()
        return (total_bytes / (bw_tb_s * 1e12)) * 1e3

    def decode_floor_tok_s(self, batch: int, ctx_len: int, bw_tb_s: float) -> float:
        t_ms = self.decode_floor_ms(batch, ctx_len, bw_tb_s)
        return batch / (t_ms / 1e3)


def extract_model_spec(model: torch.nn.Module) -> ModelSpec:
    """Pull architecture constants from a loaded model."""
    cfg = model.config
    n_params = sum(p.numel() for p in model.parameters())
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    dtype = next(model.parameters()).dtype
    bpp = 2 if dtype in (torch.bfloat16, torch.float16) else 4
    return ModelSpec(
        n_params=n_params,
        n_layers=cfg.num_hidden_layers,
        n_kv_heads=n_kv,
        head_dim=head_dim,
        bytes_per_param=bpp,
    )


# GPU clock query

def get_sm_clock_mhz() -> int:
    """Current SM clock in MHz. Returns -1 if unavailable."""
    # Try pynvml first (no subprocess overhead)
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
    except Exception:
        pass
    # Fallback: nvidia-smi
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
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "unknown"


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
    print(f"  Parameters : {sum(p.numel() for p in model.parameters()) / 1e9:.2f} B")
    print(f"  Vocab      : {model.config.vocab_size:,}")
    print(f"  GPU        : {get_gpu_name()}")
    return model, tokenizer


# Forward helpers — logits_to_keep=1 to avoid the vocab OOM trap

def _forward(model, input_ids, past_key_values=None):
    """
    Single forward call with num_logits_to_keep=1.

    Qwen2.5's vocab is 152,064.  Without this flag, logits at 8K context =
    8192 × 152064 × 2 B ≈ 2.5 GB per sequence (and softmax upcasts to FP32
    → ~5 GB).  At batch 8 that's 40 GB of logits alone → OOM.
    """
    kwargs = dict(use_cache=True)
    if past_key_values is not None:
        kwargs["past_key_values"] = past_key_values
    # num_logits_to_keep was added in transformers ~4.43
    try:
        out = model(input_ids, num_logits_to_keep=1, **kwargs)
    except TypeError:
        out = model(input_ids, **kwargs)
    return out


# Core benchmark

@torch.no_grad()
def benchmark_decode(
    model: torch.nn.Module,
    spec: ModelSpec,
    batch_size: int,
    context_length: int,
    warmup_steps: int = 10,
    timed_steps: int = 50,
    hbm_bw_tb_s: float = 3.35,
    enable_profile: bool = False,
    grid_prices: list[float] | None = None,
) -> dict:
    """
    1. Prefill with random tokens to fill KV cache to `context_length` (untimed).
    2. 10 warmup decode steps (untimed — lets SM clocks settle).
    3. ≥50 timed single-token decode steps with CUDA events.
    """
    device = next(model.parameters()).device

    # 1. Prefill (untimed)
    input_ids = torch.randint(100, 10_000, (batch_size, context_length),
                              device=device, dtype=torch.long)
    torch.cuda.synchronize()
    out = _forward(model, input_ids)
    past_kv = out.past_key_values
    next_token = out.logits[:, -1:, :].argmax(dim=-1)  # [B, 1]
    del out, input_ids
    torch.cuda.synchronize()

    # 2. Warmup decode (untimed)
    for _ in range(warmup_steps):
        out = _forward(model, next_token, past_kv)
        past_kv = out.past_key_values
        next_token = out.logits[:, -1:, :].argmax(dim=-1)
    torch.cuda.synchronize()

    sm_clock_before = get_sm_clock_mhz()

    # 3. Timed decode steps (with energy measurement)
    energy = EnergyTracker()
    energy.start()

    if enable_profile:
        torch.cuda.profiler.start()

    timings_ms: list[float] = []
    for _ in range(timed_steps):
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_ev.record()
        out = _forward(model, next_token, past_kv)
        past_kv = out.past_key_values
        next_token = out.logits[:, -1:, :].argmax(dim=-1)
        end_ev.record()
        torch.cuda.synchronize()

        timings_ms.append(start_ev.elapsed_time(end_ev))

    if enable_profile:
        torch.cuda.profiler.stop()

    total_tokens_generated = batch_size * timed_steps
    energy_snap = energy.stop(n_tokens=total_tokens_generated)

    sm_clock_after = get_sm_clock_mhz()

    # Clean up
    del past_kv, next_token
    torch.cuda.empty_cache()

    # Compute stats and roofline comparison
    t = np.array(timings_ms)
    floor_ms = spec.decode_floor_ms(batch_size, context_length, hbm_bw_tb_s)
    floor_tok_s = spec.decode_floor_tok_s(batch_size, context_length, hbm_bw_tb_s)
    measured_tok_s = batch_size / (np.median(t) / 1e3)

    kv_bytes = batch_size * context_length * spec.kv_bytes_per_token()
    weight_bytes = spec.weight_bytes
    kv_frac = kv_bytes / (kv_bytes + weight_bytes)

    # Energy cost at grid prices
    prices = grid_prices or [0.08, 0.30]  # £/kWh: 8p off-peak, 30p peak
    cost_per_1m = {
        f"cost_per_1m_tokens_at_{p}": round(
            cost_per_million_tokens(energy_snap.joules, total_tokens_generated, p), 4
        )
        for p in prices
    }

    return {
        "batch_size": batch_size,
        "context_length": context_length,
        "warmup_steps": warmup_steps,
        "timed_steps": timed_steps,
        # Timings
        "median_ms": float(np.median(t)),
        "mean_ms": float(np.mean(t)),
        "p99_ms": float(np.percentile(t, 99)),
        "min_ms": float(np.min(t)),
        "max_ms": float(np.max(t)),
        "std_ms": float(np.std(t)),
        # Roofline
        "floor_ms": round(floor_ms, 3),
        "floor_tok_s": round(floor_tok_s, 1),
        "measured_tok_s": round(measured_tok_s, 1),
        "pct_of_floor": round((floor_ms / np.median(t)) * 100, 1),
        "slowdown_vs_floor": round(np.median(t) / floor_ms, 2),
        # Traffic breakdown
        "weight_gb": round(weight_bytes / 1e9, 2),
        "kv_traffic_gb": round(kv_bytes / 1e9, 3),
        "kv_fraction_pct": round(kv_frac * 100, 1),
        # Clock
        "sm_clock_mhz_before": sm_clock_before,
        "sm_clock_mhz_after": sm_clock_after,
        # Energy (NVML integrating counter — no sampling aliasing)
        "energy_joules": energy_snap.joules,
        "avg_power_w": energy_snap.avg_power_w,
        "tokens_per_joule": energy_snap.tokens_per_joule,
        "joules_per_1m_tokens": energy_snap.joules_per_1m_tokens,
        "throttle_reasons": energy_snap.throttle_reasons,
        **cost_per_1m,
        # Raw timings for later analysis
        "all_timings_ms": [round(x, 4) for x in t.tolist()],
    }


# CLI

def parse_args():
    p = argparse.ArgumentParser(
        description="Decode-step latency vs. bandwidth floor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python bench_decode.py --model Qwen/Qwen2.5-7B
  python bench_decode.py --model Qwen/Qwen2.5-7B --profile  # for nsys capture
        """,
    )
    p.add_argument("--model", default="Qwen/Qwen2.5-7B")
    p.add_argument("--batch-sizes", default="1,4,16,64",
                   help="Comma-separated batch sizes")
    p.add_argument("--context-lengths", default="128,512,2048,8192",
                   help="Comma-separated KV-cache context lengths")
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--timed-steps", type=int, default=50)
    p.add_argument("--hbm-bw", type=float, default=3.35,
                   help="HBM bandwidth in TB/s (H100 SXM5=3.35, PCIe=2.0, H200=4.8)")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--output", default="results_decode.json",
                   help="Path to write JSON results")
    p.add_argument("--profile", action="store_true",
                   help="Wrap timed region with cudaProfilerApi for nsys capture")
    p.add_argument("--grid-prices", default="0.08,0.30",
                   help="Comma-separated electricity prices in £/kWh for cost calc "
                        "(default: 0.08,0.30 = 8p off-peak, 30p peak)")
    return p.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This benchmark requires a GPU.", file=sys.stderr)
        sys.exit(1)

    dtype = getattr(torch, args.dtype)
    model, tokenizer = load_model(args.model, dtype)
    spec = extract_model_spec(model)

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    context_lengths = [int(x) for x in args.context_lengths.split(",")]
    grid_prices = [float(p) for p in args.grid_prices.split(",")]

    print(f"\n{'='*80}")
    print(f"  Decode Benchmark — {args.model}")
    print(f"  HBM BW assumed: {args.hbm_bw} TB/s")
    print(f"  Weights: {spec.weight_bytes / 1e9:.2f} GB")
    print(f"  KV/token: {spec.kv_bytes_per_token():,} B ({spec.kv_bytes_per_token() / 1024:.0f} KiB)")
    print(f"  Sweep: B ∈ {batch_sizes}, ctx ∈ {context_lengths}")
    print(f"  Steps: {args.warmup_steps} warmup + {args.timed_steps} timed")
    print(f"  Grid prices: {grid_prices} £/kWh")
    print(f"{'='*80}\n")

    all_results = []

    for bs in batch_sizes:
        for ctx in context_lengths:
            label = f"B={bs}, ctx={ctx}"
            print(f"▸ {label} …", end="", flush=True)

            try:
                result = benchmark_decode(
                    model, spec, bs, ctx,
                    warmup_steps=args.warmup_steps,
                    timed_steps=args.timed_steps,
                    hbm_bw_tb_s=args.hbm_bw,
                    enable_profile=args.profile,
                    grid_prices=grid_prices,
                )
                all_results.append(result)

                throttle = ",".join(result["throttle_reasons"])
                print(
                    f"  median {result['median_ms']:.2f} ms  "
                    f"(floor {result['floor_ms']:.2f} ms → "
                    f"{result['pct_of_floor']:.0f}% of roofline)  "
                    f"| {result['measured_tok_s']:.0f} tok/s  "
                    f"| {result['avg_power_w']:.0f} W  "
                    f"| {result['tokens_per_joule']:.2f} tok/J  "
                    f"| throttle: {throttle}"
                )
            except torch.cuda.OutOfMemoryError:
                print(f"  ✗ OOM — skipped")
                torch.cuda.empty_cache()

    # Summary table: latency + roofline
    print(f"\n{'─'*110}")
    print(f"{'Case':<20} {'Median ms':>10} {'Floor ms':>10} {'% Floor':>9} "
          f"{'tok/s':>10} {'SM MHz':>8} {'KV %':>6} "
          f"{'Power W':>8} {'tok/J':>8}")
    print(f"{'─'*110}")
    for r in all_results:
        label = f"B={r['batch_size']}, ctx={r['context_length']}"
        print(f"{label:<20} {r['median_ms']:>10.2f} {r['floor_ms']:>10.2f} "
              f"{r['pct_of_floor']:>8.0f}% {r['measured_tok_s']:>10.0f} "
              f"{r['sm_clock_mhz_after']:>8} {r['kv_fraction_pct']:>5.1f}% "
              f"{r['avg_power_w']:>8.0f} {r['tokens_per_joule']:>8.2f}")
    print(f"{'─'*110}")

    # Energy summary: £/1M tokens at each grid price
    print(f"\n{'─'*90}")
    price_headers = "  ".join(f"{'£/1M@' + str(p):>12}" for p in grid_prices)
    print(f"{'Case':<20} {'tok/J':>8} {'J/1M tok':>10}  {price_headers}")
    print(f"{'─'*90}")
    for r in all_results:
        label = f"B={r['batch_size']}, ctx={r['context_length']}"
        costs = "  ".join(
            f"{r.get(f'cost_per_1m_tokens_at_{p}', 0):>12.4f}" for p in grid_prices
        )
        print(f"{label:<20} {r['tokens_per_joule']:>8.2f} "
              f"{r['joules_per_1m_tokens']:>10.1f}  {costs}")
    print(f"{'─'*90}")

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_data = {
        "model": args.model,
        "gpu": get_gpu_name(),
        "dtype": args.dtype,
        "hbm_bw_tb_s": args.hbm_bw,
        "grid_prices_per_kwh": grid_prices,
        "spec": {
            "n_params": spec.n_params,
            "n_layers": spec.n_layers,
            "n_kv_heads": spec.n_kv_heads,
            "head_dim": spec.head_dim,
            "weight_bytes": spec.weight_bytes,
            "kv_bytes_per_token": spec.kv_bytes_per_token(),
        },
        "results": all_results,
    }
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
