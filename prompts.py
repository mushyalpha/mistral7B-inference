"""
Shared prompt-pool utilities for the benchmark harness.

Builds two length buckets ("short" and "long") from a real dataset (ShareGPT)
so prefill (long input) and decode (long output) are stressed realistically,
per the "same prompts across systems / real dataset" fairness requirement.
Falls back to a small deterministic synthetic pool (with a warning) if the
dataset can't be downloaded, so the harness still runs offline.

Results are cached to disk so that running the HF pass and the vLLM pass
separately samples from the *identical* prompt pool.
"""
import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

CACHE_DIR = Path(".cache/prompts")
SHORT_MAX_WORDS = 40   # short-input bucket: prompts with <= this many words
LONG_MIN_WORDS = 150   # long-input bucket: prompts with >= this many words
DATASET_ID = "anon8231489123/ShareGPT_Vicuna_unfiltered"
DATASET_SPLIT = "train"


def _cache_path(category: str) -> Path:
    return CACHE_DIR / f"sharegpt_{category}.json"


def _fetch_sharegpt_prompts(pool_size: int) -> Optional[List[str]]:
    """Streams ShareGPT and returns first-turn human messages, or None on failure."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[prompts] `datasets` not installed; using synthetic fallback prompts.")
        return None

    try:
        dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT, streaming=True)
        collected = []
        for item in dataset:
            conversations = item.get("conversations", [])
            if conversations and conversations[0].get("from") == "human":
                text = conversations[0].get("value", "").strip()
                if text:
                    collected.append(text)
            if len(collected) >= pool_size:
                break
        return collected or None
    except Exception as exc:  # network issues, schema changes, etc.
        print(f"[prompts] Could not load ShareGPT dataset ({exc}); using synthetic fallback prompts.")
        return None


# --- Deterministic offline fallback so the harness always runs -------------

_FALLBACK_SHORT = [
    "What's the capital of France?",
    "Give me a one-sentence summary of photosynthesis.",
    "Name three primary colors.",
    "How do I reverse a list in Python?",
    "What year did the Berlin Wall fall?",
    "Translate 'good morning' to Spanish.",
    "What's the boiling point of water in Celsius?",
    "Suggest a good name for a coffee shop.",
    "What does CPU stand for?",
    "List two benefits of regular exercise.",
    "How many continents are there?",
    "What's a good substitute for butter in baking?",
]

_FALLBACK_LONG_TEMPLATE = (
    "I'm building a {topic} and I need detailed help. Please walk me through the "
    "full reasoning step by step, covering the historical context, the key "
    "technical trade-offs involved, at least three concrete examples, common "
    "pitfalls that beginners run into, and a comparison against the two most "
    "popular alternative approaches. Also explain how this connects to broader "
    "trends in the field, describe the kind of evidence an expert would look "
    "for before trusting a result, and finish with a short set of actionable "
    "recommendations for someone who wants to get started this week. Be "
    "thorough and don't skip steps -- I want to actually understand the "
    "reasoning, not just get a shallow answer."
)
_FALLBACK_LONG_TOPICS = [
    "distributed key-value database", "renewable energy microgrid", "large language model",
    "supply chain optimisation system", "autonomous drone delivery fleet", "recommendation engine",
    "genomic sequencing pipeline", "quantum error-correcting code", "city traffic simulation model",
    "decentralised identity system", "real-time fraud detection system", "satellite ground station network",
]
_FALLBACK_LONG = [_FALLBACK_LONG_TEMPLATE.format(topic=t) for t in _FALLBACK_LONG_TOPICS]


def _synthetic_pool(category: str, pool_size: int) -> List[str]:
    base = _FALLBACK_SHORT if category == "short" else _FALLBACK_LONG
    reps = (pool_size // len(base)) + 1
    return (base * reps)[:pool_size]


def _word_count(text: str) -> int:
    return len(text.split())


def build_prompt_pools(pool_size: int = 200, seed: int = 0) -> Tuple[List[str], List[str]]:
    """Returns (short_pool, long_pool) -- deterministic (given `seed`) lists of
    prompt strings, shared across engines via an on-disk cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    short_cache_path, long_cache_path = _cache_path("short"), _cache_path("long")

    if short_cache_path.exists() and long_cache_path.exists():
        short_pool = json.loads(short_cache_path.read_text())
        long_pool = json.loads(long_cache_path.read_text())
    else:
        # Oversample the raw stream since only a fraction of turns will land
        # cleanly in either bucket.
        raw = _fetch_sharegpt_prompts(pool_size=pool_size * 20)
        if raw:
            short_pool = [p for p in raw if _word_count(p) <= SHORT_MAX_WORDS]
            long_pool = [p for p in raw if _word_count(p) >= LONG_MIN_WORDS]
            if not short_pool:
                print("[prompts] No ShareGPT prompts fit the short bucket; padding with synthetic prompts.")
                short_pool = _synthetic_pool("short", pool_size)
            if not long_pool:
                print("[prompts] No ShareGPT prompts fit the long bucket; padding with synthetic prompts.")
                long_pool = _synthetic_pool("long", pool_size)
        else:
            short_pool = _synthetic_pool("short", pool_size)
            long_pool = _synthetic_pool("long", pool_size)

        short_cache_path.write_text(json.dumps(short_pool))
        long_cache_path.write_text(json.dumps(long_pool))

    rng = random.Random(seed)
    short_pool = list(short_pool)
    long_pool = list(long_pool)
    rng.shuffle(short_pool)
    rng.shuffle(long_pool)
    return short_pool[:pool_size], long_pool[:pool_size]


def sample_prompts(pool: List[str], n: int, seed: int) -> List[str]:
    """Deterministically samples n prompts from pool (cycles with a shuffled
    copy of the pool if n exceeds the pool size, so it never crashes)."""
    rng = random.Random(seed)
    if n <= len(pool):
        return rng.sample(pool, n)
    shuffled = list(pool)
    rng.shuffle(shuffled)
    return [shuffled[i % len(shuffled)] for i in range(n)]
