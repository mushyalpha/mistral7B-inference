"""
doing a naive hugging face transformer inference as a deliberately unoptimised 
baseline so we compare it against vLLM and quatised variats
"""


import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Config 
MODEL_ID   = "mistralai/Mistral-7B-Instruct-v0.3"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TOKENS = 256

TEST_PROMPTS = [
    "Between Messi and Ronaldo, who is the better player?",
    "Write a 500 word essay on the importance of AI in electricity grid management?",
    "What are the top 3 places in Botswana to visit for safari and why?",
]


def load_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,         # halve VRAM usage for RTX 3090
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
            do_sample=False,        # greedy - deterministic, good for benchmarks
            temperature=None,
            top_p=None,
        )
    elapsed = time.perf_counter() - t0

    # Strip the input tokens from the output
    new_tokens   = output_ids[0][input_ids.shape[-1]:]
    output_text  = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return output_text, elapsed


def main():
    print(" Mistral 7B - Baseline HuggingFace Inference")
    print(f"Device : {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM   : {vram_gb:.1f} GB")
    print()

    tokenizer, model = load_model()

    total_latency  = 0.0
    total_tokens   = 0

    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"Prompt {i+1} ")
        print(f"Q: {prompt}")
        response, latency = generate(tokenizer, model, prompt)
        n_tokens = len(tokenizer.encode(response))

        print(f"A: {response}")
        print(f"   Latency : {latency:.2f}s")
        print(f"   Tokens  : {n_tokens}")
        print(f"   Speed   : {n_tokens / latency:.1f} tok/s")
        print()

        total_latency += latency
        total_tokens  += n_tokens

    print(" SUMMARY")
    print(f"  Avg latency : {total_latency / len(TEST_PROMPTS):.2f}s")
    print(f"  Avg speed   : {total_tokens / total_latency:.1f} tok/s")
    print(f"  Total tokens: {total_tokens}")
    if DEVICE == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak VRAM   : {peak_gb:.2f} GB")


if __name__ == "__main__":
    main()
