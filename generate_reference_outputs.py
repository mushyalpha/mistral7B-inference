"""
Generates greedy-decoded outputs for a small fixed prompt set with ONE engine.
Run once per engine, then feed both JSON files to compare_outputs.py to check
that HF and vLLM produce roughly equivalent text - directly addresses the
"forgetting to verify output quality" pitfall (speed means nothing if one
engine is silently producing garbage).

    python generate_reference_outputs.py - engine hf
    python generate_reference_outputs.py - engine vllm
    python compare_outputs.py
"""
import argparse
import json
from pathlib import Path

import torch

from bench_utils import set_all_seeds

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

REFERENCE_PROMPTS = [
    "What's the capital of Japan?",
    "Explain the difference between TCP and UDP in two sentences.",
    "Write a haiku about the ocean.",
    "List three uses of a hash map.",
    "What is 17 times 24?",
]


def run_hf(prompts, max_new_tokens):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    outputs = []
    for prompt in prompts:
        formatted = f"[INST] {prompt} [/INST]"
        inputs = tokenizer(formatted, return_tensors="pt").to(device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        outputs.append(text)
    return outputs


def run_vllm(prompts, max_new_tokens, seed):
    from vllm import LLM, SamplingParams

    llm = LLM(model=MODEL_ID, dtype="float16", gpu_memory_utilization=0.90)
    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, seed=seed)
    formatted = [f"[INST] {p} [/INST]" for p in prompts]
    results = llm.generate(formatted, sampling, use_tqdm=False)
    return [r.outputs[0].text for r in results]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["hf", "vllm"], required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    set_all_seeds(args.seed)
    out_path = Path(args.out or f"results/outputs_{args.engine}.json")

    if args.engine == "hf":
        outputs = run_hf(REFERENCE_PROMPTS, args.max_new_tokens)
    else:
        outputs = run_vllm(REFERENCE_PROMPTS, args.max_new_tokens, args.seed)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        [{"prompt": p, "output": o} for p, o in zip(REFERENCE_PROMPTS, outputs)],
        indent=2,
    ))
    print(f"Wrote {len(outputs)} reference outputs to {out_path}")


if __name__ == "__main__":
    main()
