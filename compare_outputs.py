"""
Compares reference outputs from two engines (produced by
generate_reference_outputs.py) to sanity-check that greedy decoding gives
roughly equivalent text -- catches silent correctness regressions that a
throughput number alone would never reveal.
"""
import argparse
import difflib
import json
from pathlib import Path


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default="results/outputs_hf.json")
    parser.add_argument("--b", default="results/outputs_vllm.json")
    parser.add_argument("--label-a", default="hf")
    parser.add_argument("--label-b", default="vllm")
    parser.add_argument("--threshold", type=float, default=0.6,
                         help="similarity ratio (0-1) below which a pair is flagged")
    args = parser.parse_args()

    a_path, b_path = Path(args.a), Path(args.b)
    if not a_path.exists() or not b_path.exists():
        missing = a_path if not a_path.exists() else b_path
        raise SystemExit(f"{missing} not found -- run generate_reference_outputs.py for both engines first.")

    outputs_a = json.loads(a_path.read_text())
    outputs_b = json.loads(b_path.read_text())

    if len(outputs_a) != len(outputs_b):
        print(f"WARNING: {args.label_a} has {len(outputs_a)} outputs, {args.label_b} has {len(outputs_b)}.")

    flagged = 0
    n = min(len(outputs_a), len(outputs_b))
    for i in range(n):
        item_a, item_b = outputs_a[i], outputs_b[i]
        if item_a["prompt"] != item_b["prompt"]:
            print(f"[{i}] WARNING: prompts differ between files; comparison invalid for this row.")
            continue

        score = similarity(item_a["output"], item_b["output"])
        flag = "  <-- FLAGGED" if score < args.threshold else ""
        if flag:
            flagged += 1

        print(f"[{i}] similarity={score:.2f}{flag}")
        print(f"    prompt : {item_a['prompt']}")
        print(f"    {args.label_a:5s}: {item_a['output'][:200]!r}")
        print(f"    {args.label_b:5s}: {item_b['output'][:200]!r}")
        print()

    print(f"Summary: {flagged}/{n} pairs below similarity threshold {args.threshold}")
    if flagged:
        print("Investigate flagged pairs before trusting the throughput numbers -- "
              "a speed win means nothing if the outputs have diverged.")


if __name__ == "__main__":
    main()
