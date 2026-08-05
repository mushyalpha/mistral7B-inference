"""
Shared helpers for the benchmark harness: seeding, GPU memory tracking,
percentile math, and incremental CSV writing. Kept dependency-free (besides
torch) so it can be imported by both the HF and vLLM code paths.
"""
import csv
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List

import torch


def set_all_seeds(seed: int) -> None:
    """Fixes every RNG we touch so runs are reproducible across engines/trials."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def track_peak_memory_gb():
    """Resets CUDA peak-memory stats on enter; yields a dict holding `peak_gb`
    once the block exits (0.0 on CPU-only machines)."""
    result = {"peak_gb": 0.0}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    try:
        yield result
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            result["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9


def percentiles(values: List[float], ps: Iterable[int] = (50, 90, 95, 99)) -> Dict[str, float]:
    """Nearest-rank percentiles. Returns 0.0 for every key if `values` is empty."""
    if not values:
        return {f"p{p}": 0.0 for p in ps}
    ordered = sorted(values)
    n = len(ordered)
    out = {}
    for p in ps:
        rank = max(1, int((p / 100) * n + 0.5))  # nearest-rank, 1-indexed
        idx = min(n, rank) - 1
        out[f"p{p}"] = ordered[idx]
    return out


def append_csv_row(path: str, row: Dict) -> None:
    """Appends a row to a CSV, writing the header first if the file is new/empty."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    file_exists = os.path.isfile(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def now() -> float:
    return time.perf_counter()
