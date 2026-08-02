"""
serve_vllm.py — vLLM offline inference
Direct comparison with baseline_inference.py
Uses vLLM's PagedAttention + continuous batching
"""
import time
import torch
from vllm import LLM, SamplingParams

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

TEST_PROMPTS = [
    "Between Messi and Ronaldo, who is the better player?",
    "Write a 500 word essay on the importance of AI in electricity grid management?",
    "What are the top 3 places in Botswana to visit for safari and why?",
]

# numbers from baseline_inference.py 
BASELINE_TOK_S   = 41.5
BASELINE_LATENCY = 5.56


def main():
    print(" Mistral 7B — vLLM Inference")
    print(f"Device : {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM   : {total_mem:.1f} GB")
    print()

    print("Loading model with vLLM...")
    llm = LLM(
        model=MODEL_ID,
        dtype="float16",
        gpu_memory_utilization=0.90,   # vLLM manages KV cache pages within this budget
    )

    sampling = SamplingParams(
        temperature=0.7,
        max_tokens=256,
    )

    total_latency = 0.0
    total_tokens  = 0

    for i, prompt in enumerate(TEST_PROMPTS, 1):
        # Mistral Instruct chat format
        formatted = f"[INST] {prompt} [/INST]"

        print(f"Prompt {i}")
        print(f"Q: {prompt}")

        t0      = time.perf_counter()
        outputs = llm.generate([formatted], sampling)
        latency = time.perf_counter() - t0

        response = outputs[0].outputs[0].text
        n_tokens = len(outputs[0].outputs[0].token_ids)

        print(f"A: {response.strip()}")
        print(f"   Latency : {latency:.2f}s")
        print(f"   Tokens  : {n_tokens}")
        print(f"   Speed   : {n_tokens / latency:.1f} tok/s")
        print()

        total_latency += latency
        total_tokens  += n_tokens

    avg_latency = total_latency / len(TEST_PROMPTS)
    avg_speed   = total_tokens  / total_latency

    print(" SUMMARY")
    print(f"  Avg latency : {avg_latency:.2f}s")
    print(f"  Avg speed   : {avg_speed:.1f} tok/s")
    print(f"  Total tokens: {total_tokens}")
    if DEVICE == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak VRAM   : {peak_gb:.2f} GB")

    print()
    print(" COMPARISON vs BASELINE")
    print(f"  {'Metric':<18} {'Baseline':>12} {'vLLM':>12} {'Speedup':>10}")
    print(f"  {'-'*54}")
    speedup = avg_speed / BASELINE_TOK_S
    lat_imp = BASELINE_LATENCY / avg_latency
    print(f"  {'Throughput (tok/s)':<18} {BASELINE_TOK_S:>12.1f} {avg_speed:>12.1f} {speedup:>9.1f}x")
    print(f"  {'Latency (s)':<18} {BASELINE_LATENCY:>12.2f} {avg_latency:>12.2f} {lat_imp:>9.1f}x")


if __name__ == "__main__":
    main()
