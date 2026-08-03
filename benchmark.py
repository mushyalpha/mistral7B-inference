import argparse
import time
import csv
import os
import torch

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
MAX_TOKENS = 256

def get_prompts(num_requests):
    base_prompts = [
        "Between Messi and Ronaldo, who is the better player?",
        "Write a 500 word essay on the importance of AI in electricity grid management?",
        "What are the top 3 places in Botswana to visit for safari and why?",
    ]
    prompts = []
    for i in range(num_requests):
        prompts.append(base_prompts[i % len(base_prompts)])
    # format for mistral
    return [f"[INST] {p} [/INST]" for p in prompts]


def run_vllm(prompts):
    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL_ID, dtype="float16", gpu_memory_utilization=0.90)
    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    elapsed = time.perf_counter() - t0
    
    total_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
    return total_tokens, elapsed


def run_hf(prompts):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    # Batch process
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
    
    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, 
            max_new_tokens=MAX_TOKENS, 
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    elapsed = time.perf_counter() - t0
    
    # Calculate generated tokens only (ignore input and padding)
    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    total_tokens = 0
    for g in generated_ids:
        total_tokens += (g != tokenizer.pad_token_id).sum().item()
        
    return total_tokens, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["hf", "vllm"], required=True)
    parser.add_argument("--num-requests", type=int, default=1)
    args = parser.parse_args()

    prompts = get_prompts(args.num_requests)
    
    print(f"\n Running {args.engine.upper()} with {args.num_requests} concurrent requests...\n")
    
    if args.engine == "vllm":
        total_tokens, elapsed = run_vllm(prompts)
    else:
        total_tokens, elapsed = run_hf(prompts)

    throughput = total_tokens / elapsed
    
    print(f"\n RESULTS:")
    print(f"Engine      : {args.engine.upper()}")
    print(f"Requests    : {args.num_requests}")
    print(f"Total Time  : {elapsed:.2f}s")
    print(f"Total Tokens: {total_tokens}")
    print(f"Throughput  : {throughput:.2f} tok/s\n")
    
    # Save to CSV
    csv_file = "benchmark_results.csv"
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Engine", "NumRequests", "TotalTime", "TotalTokens", "Throughput"])
        writer.writerow([args.engine, args.num_requests, round(elapsed, 2), total_tokens, round(throughput, 2)])
    print(f"Results appended to {csv_file}")

if __name__ == "__main__":
    main()
