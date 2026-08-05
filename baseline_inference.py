"""
doing a naive hugging face transformer inference as a deliberately unoptimised 
baseline so we compare it against vLLM and quantised variants
"""


import time
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from bench_utils import set_all_seeds

# Config
MODEL_ID    = "mistralai/Mistral-7B-Instruct-v0.3"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TOKENS  = 256
NUM_PROMPTS = 100
SEED        = 0

def load_sharegpt_prompts(num_prompts):
    """Downloads ShareGPT and extracts the first user message from conversations."""
    
    print(f" Loading {num_prompts} prompts from ShareGPT dataset")
    
    # Load a ShareGPT dataset split from Hugging Face
    dataset = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train", streaming=True)
    
    prompts = []
    for item in dataset:
        conversations = item.get("conversations", [])
        # Ensure there is a conversation and the first turn is from the user
        if conversations and conversations[0].get("from") == "human":
            prompts.append(conversations[0].get("value"))
        
        if len(prompts) >= num_prompts:
            break
            
    print(f"Successfully loaded {len(prompts)} real-world user prompts.")
    return prompts


def load_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Warm-up fix
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,   # BFP16 halve VRAM usage for RTX 3090
        device_map="auto",           # auto spread parts of the model across GPU
    )
    model.eval()
    return tokenizer, model


def generate(tokenizer, model, prompt: str) -> tuple[str, float]:
    """Returns (generated_text, latency_seconds)."""
    messages = [{"role": "user", "content": prompt}]

    # Mistral Instruct uses a chat template
    # Transformers 5.x returns BatchEncoding, not a raw tensor
    tokenized = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    )
    if hasattr(tokenized, 'input_ids'):
        input_ids = tokenized.input_ids.to(DEVICE)
    else:
        input_ids = tokenized.to(DEVICE)

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,        # greedy - deterministic
            temperature=None,
            top_p=None,
        )
    elapsed = time.perf_counter() - t0

    # Strip the input tokens from the output
    new_tokens   = output_ids[0][input_ids.shape[-1]:]
    output_text  = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return output_text, elapsed

def main():
    print(" Mistral 7B - Naive HuggingFace Inference")
    print(f"Device : {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM   : {vram_gb:.1f} GB")
    print()

    set_all_seeds(SEED)

    # Load data and model
    tokenizer, model = load_model()     
    test_prompts = load_sharegpt_prompts(NUM_PROMPTS)

    # GPU WARMUP
    print("\n Warming up GPU clocks.")
    for _ in range(3):
        _ = generate(tokenizer, model, "Warmup prompt.")
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats() 
    print("Warmup complete. Starting benchmark.\n")

    total_latency  = 0.0
    total_tokens   = 0

# Running inference sequentially
    for i, prompt in enumerate(test_prompts):
        # Truncate prompt display for cleaner logs if it's a massive ShareGPT prompt
        display_prompt = prompt if len(prompt) < 60 else f"{prompt[:60]}..."
        print(f"[{i+1}/{NUM_PROMPTS}] Q: {display_prompt}")
        
        response, latency = generate(tokenizer, model, prompt)
        n_tokens = len(tokenizer.encode(response))

        total_latency += latency
        total_tokens  += n_tokens

    print("\n FINAL BENCHMARK SUMMARY")
    print(f"  Total Prompts Evaluated: {NUM_PROMPTS}")
    print(f"  Avg latency per request: {total_latency / NUM_PROMPTS:.2f}s")
    print(f"  System Throughput      : {total_tokens / total_latency:.1f} tok/s")
    print(f"  Total tokens generated : {total_tokens}")
    if DEVICE == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak VRAM Usage        : {peak_gb:.2f} GB")


if __name__ == "__main__":
    main()
