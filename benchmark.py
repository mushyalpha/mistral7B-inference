"""
vLLM vs. HuggingFace benchmark harness.

Sweeps the three axes called out in the experimental design:
  1. concurrency / batch size  (--concurrency)
  2. input length              (--input-lengths: short/long, tests prefill+KV cache)
  3. output length              (--output-lengths: max_new_tokens, tests decode)

...with warm-up runs excluded from timing, a fixed seed + greedy decoding for
reproducibility, the same prompt pool for both engines, and multiple trials so
variance can be reported.

Run once per engine (loading both HF and vLLM in the same process isn't
practical -- they each want to own the whole GPU):

    python benchmark.py --engine hf
    python benchmark.py --engine vllm

Both invocations append to the same CSV (results/benchmark_results.csv by
default) using identical default sweep parameters, so the two engines are
directly comparable in plot_results.py.
"""
import argparse
import itertools
import json
from pathlib import Path

import torch

from bench_utils import append_csv_row, now, percentiles, set_all_seeds, track_peak_memory_gb
from prompts import build_prompt_pools, sample_prompts

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_CONCURRENCIES = [1, 4, 16, 64]
DEFAULT_OUTPUT_LENS = [64, 512]        # decode-light vs. decode-heavy
DEFAULT_INPUT_LENS = ["short", "long"]  # prefill-light vs. prefill-heavy


def parse_args():
    parser = argparse.ArgumentParser(description="vLLM vs HuggingFace throughput/latency sweep")
    parser.add_argument("--engine", choices=["hf", "vllm"], required=True)
    parser.add_argument("--concurrency", type=str, default=",".join(map(str, DEFAULT_CONCURRENCIES)),
                         help="comma-separated batch sizes, e.g. 1,4,16,64")
    parser.add_argument("--input-lengths", type=str, default=",".join(DEFAULT_INPUT_LENS),
                         help="comma list from: short,long")
    parser.add_argument("--output-lengths", type=str, default=",".join(map(str, DEFAULT_OUTPUT_LENS)),
                         help="comma list of max_new_tokens values, e.g. 64,512")
    parser.add_argument("--trials", type=int, default=3, help="repeats per (input,output,concurrency) combo")
    parser.add_argument("--warmup-requests", type=int, default=4,
                         help="requests used to trigger CUDA graph capture / compilation before timing")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt-pool-size", type=int, default=200)
    parser.add_argument("--output-csv", type=str, default="results/benchmark_results.csv")
    parser.add_argument("--outputs-json", type=str, default=None,
                         help="optional path to dump sample generations per combo, for compare_outputs.py")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="vLLM only")
    return parser.parse_args()


def run_hf_batch(tokenizer, model, prompts, max_new_tokens, device):
    formatted = [f"[INST] {p} [/INST]" for p in prompts]
    unpadded_ids = tokenizer(formatted, padding=False)["input_ids"]
    input_token_counts = [len(ids) for ids in unpadded_ids]

    inputs = tokenizer(formatted, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(device)

    t0 = now()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,        # greedy -- deterministic
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )
    elapsed = now() - t0

    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    texts, token_counts = [], []
    for g in generated:
        n = int((g != tokenizer.pad_token_id).sum().item())
        token_counts.append(n)
        texts.append(tokenizer.decode(g, skip_special_tokens=True))

    # model.generate() is one synchronous call for the whole batch: every
    # request is blocked until the slowest sequence in the batch finishes
    # (head-of-line blocking under static batching), so every request's
    # latency genuinely IS the batch's total elapsed time. This is the
    # exact effect continuous batching is meant to avoid.
    latencies = [elapsed] * len(prompts)

    return {
        "elapsed": elapsed,
        "total_tokens": sum(token_counts),
        "token_counts": token_counts,
        "input_token_counts": input_token_counts,
        "latencies": latencies,
        "texts": texts,
    }


def run_vllm_batch(llm, prompts, max_new_tokens, seed):
    from vllm import SamplingParams

    formatted = [f"[INST] {p} [/INST]" for p in prompts]
    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, seed=seed)

    t0 = now()
    outputs = llm.generate(formatted, sampling, use_tqdm=False)
    elapsed = now() - t0

    texts, token_counts, input_token_counts, latencies = [], [], [], []
    for out in outputs:
        token_counts.append(len(out.outputs[0].token_ids))
        texts.append(out.outputs[0].text)
        input_token_counts.append(len(out.prompt_token_ids))

        # vLLM exposes real per-request arrival/finish timestamps even though
        # llm.generate() itself blocks until the whole batch is done -- these
        # reveal that continuous batching lets individual requests finish
        # earlier than the slowest one, unlike HF's static batching.
        metrics = getattr(out, "metrics", None)
        arrival = getattr(metrics, "arrival_time", None) if metrics else None
        finished = getattr(metrics, "finished_time", None) if metrics else None
        if arrival is not None and finished is not None:
            latencies.append(finished - arrival)
        else:
            latencies.append(elapsed)

    return {
        "elapsed": elapsed,
        "total_tokens": sum(token_counts),
        "token_counts": token_counts,
        "input_token_counts": input_token_counts,
        "latencies": latencies,
        "texts": texts,
    }


def load_hf(dtype: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch_dtype, device_map="auto")
    model.eval()
    return tokenizer, model


def load_vllm(dtype: str, gpu_memory_utilization: float):
    from vllm import LLM
    import os
    # On RunPod, /workspace is the large persistent disk; /root fills up fast.
    # Respect HF_HOME if set, otherwise fall back to /workspace cache.
    default_dl_dir = "/workspace/.hf_cache/hub"
    download_dir = os.environ.get("HUGGINGFACE_HUB_CACHE", default_dl_dir)
    os.makedirs(download_dir, exist_ok=True)
    return LLM(
        model=MODEL_ID,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        download_dir=download_dir,
    )


def main():
    args = parse_args()
    set_all_seeds(args.seed)

    concurrencies = [int(c) for c in args.concurrency.split(",") if c]
    input_lens = [s.strip() for s in args.input_lengths.split(",") if s.strip()]
    output_lens = [int(o) for o in args.output_lengths.split(",") if o]

    print(f"Engine        : {args.engine}")
    print(f"Device        : {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU           : {torch.cuda.get_device_name(0)}")
        print(f"VRAM          : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Concurrency   : {concurrencies}")
    print(f"Input lengths : {input_lens}")
    print(f"Output lengths: {output_lens}")
    print(f"Trials        : {args.trials}  |  Warm-up requests: {args.warmup_requests}")
    print(f"Seed          : {args.seed}\n")

    short_pool, long_pool = build_prompt_pools(pool_size=args.prompt_pool_size, seed=args.seed)
    pools = {"short": short_pool, "long": long_pool}

    if args.engine == "hf":
        tokenizer, model = load_hf(args.dtype)

        def run_batch(prompts, max_new_tokens):
            return run_hf_batch(tokenizer, model, prompts, max_new_tokens, DEVICE)
    else:
        llm = load_vllm(args.dtype, args.gpu_memory_utilization)

        def run_batch(prompts, max_new_tokens):
            return run_vllm_batch(llm, prompts, max_new_tokens, args.seed)

    # Warm-up: exercise the engine (CUDA graph capture, kernel autotuning,
    # vLLM's memory-profiling pass, etc.) BEFORE any timing starts, and run it
    # twice so the second pass reflects steady state, not first-call setup.
    warmup_prompts = sample_prompts(short_pool, max(args.warmup_requests, 1), seed=args.seed)
    print("Warming up (excluded from all timing)...")
    run_batch(warmup_prompts, min(output_lens))
    run_batch(warmup_prompts, min(output_lens))
    print("Warm-up complete.\n")

    outputs_dump = []
    combo_id = 0
    for input_len, output_len, concurrency in itertools.product(input_lens, output_lens, concurrencies):
        combo_id += 1
        pool = pools[input_len]

        for trial in range(args.trials):
            trial_seed = args.seed * 100_000 + combo_id * 100 + trial
            trial_prompts = sample_prompts(pool, concurrency, seed=trial_seed)

            with track_peak_memory_gb() as mem:
                result = run_batch(trial_prompts, output_len)

            lat_stats = percentiles(result["latencies"])
            throughput = result["total_tokens"] / result["elapsed"] if result["elapsed"] > 0 else 0.0

            row = {
                "engine": args.engine,
                "input_len_category": input_len,
                "output_len_tokens": output_len,
                "concurrency": concurrency,
                "trial": trial,
                "elapsed_s": round(result["elapsed"], 4),
                "total_tokens": result["total_tokens"],
                "throughput_tok_s": round(throughput, 2),
                "avg_input_tokens": round(sum(result["input_token_counts"]) / len(result["input_token_counts"]), 1),
                "avg_output_tokens": round(sum(result["token_counts"]) / len(result["token_counts"]), 1),
                "latency_mean_s": round(sum(result["latencies"]) / len(result["latencies"]), 4),
                "latency_p50_s": round(lat_stats["p50"], 4),
                "latency_p90_s": round(lat_stats["p90"], 4),
                "latency_p99_s": round(lat_stats["p99"], 4),
                "peak_mem_gb": round(mem["peak_gb"], 3),
            }
            append_csv_row(args.output_csv, row)

            print(
                f"[{input_len:5s} in | {output_len:4d} out | conc={concurrency:3d} | trial {trial + 1}/{args.trials}] "
                f"{throughput:8.1f} tok/s | p50={lat_stats['p50']:.2f}s p99={lat_stats['p99']:.2f}s | "
                f"peak_mem={mem['peak_gb']:.2f}GB"
            )

            if args.outputs_json and trial == 0:
                outputs_dump.append({
                    "input_len_category": input_len,
                    "output_len_tokens": output_len,
                    "concurrency": concurrency,
                    "prompts": trial_prompts,
                    "outputs": result["texts"],
                })

    if args.outputs_json:
        out_path = Path(args.outputs_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(outputs_dump, indent=2))
        print(f"\nSaved sample outputs to {out_path}")

    print(f"\nDone. Results appended to {args.output_csv}")


if __name__ == "__main__":
    main()
