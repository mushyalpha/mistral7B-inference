"""
Shared helpers for the benchmark harness: seeding, GPU memory tracking,
percentile math, and incremental CSV writing. Kept dependency-free (besides
torch) so it can be imported by both the HF and vLLM code paths.
"""
import csv
import os
import random
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional

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


def _nvidia_smi_used_mb(gpu_index: int = 0) -> Optional[float]:
    """Device-wide GPU memory currently in use, in MB, via `nvidia-smi`.

    This is process-agnostic: it reports actual memory committed on the GPU
    regardless of which process allocated it. That matters because vLLM's
    engine can run its forward passes in a separate worker process (its
    multiprocessing/Ray executor), in which case `torch.cuda.*` queried from
    the driver process that called `llm.generate()` sees zero allocations no
    matter how much VRAM vLLM is actually using.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(gpu_index)],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return float(out.decode().strip().splitlines()[0])
    except Exception:
        return None


class _GpuMemorySampler:
    """Background thread that polls `nvidia-smi` so we capture the peak
    device-wide memory usage over an arbitrary code block, independent of
    which process (or subprocess) did the allocating."""

    def __init__(self, gpu_index: int = 0, interval_s: float = 0.05):
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self._peak_mb = 0.0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.available = _nvidia_smi_used_mb(gpu_index) is not None

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            used = _nvidia_smi_used_mb(self.gpu_index)
            if used is not None:
                self._peak_mb = max(self._peak_mb, used)
            time.sleep(self.interval_s)

    def start(self) -> None:
        if not self.available:
            return
        self._peak_mb = _nvidia_smi_used_mb(self.gpu_index) or 0.0
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop_and_get_peak_gb(self) -> float:
        if not self.available:
            return 0.0
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        return self._peak_mb / 1024.0


@contextmanager
def track_peak_memory_gb(gpu_index: int = 0):
    """Yields a dict that holds `peak_gb` once the block exits: the highest
    device-wide GPU memory usage observed during the block.

    Prefers polling `nvidia-smi` (works correctly no matter which process
    does the allocating -- crucial for vLLM's worker-process executors).
    Falls back to torch's in-process allocator stats, then to 0.0, if
    `nvidia-smi` isn't on PATH (e.g. CPU-only dev boxes).
    """
    result = {"peak_gb": 0.0}
    sampler = _GpuMemorySampler(gpu_index=gpu_index)

    if sampler.available:
        sampler.start()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    try:
        yield result
    finally:
        if sampler.available:
            result["peak_gb"] = sampler.stop_and_get_peak_gb()
        elif torch.cuda.is_available():
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
