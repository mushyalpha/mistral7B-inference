# Traps That Will Eat Your Paid Hours

> Read this **before** you boot the instance.
> Each of these cost real people real money to learn the hard way.

---

## 1. The Vocab Trap (OOM in the first 10 minutes)

Qwen2.5's vocab is **152,064**. HuggingFace's forward computes logits for
**every position** by default.

Do the arithmetic:

| Context | Logits (BF16) | + FP32 softmax upcast | × Batch 8 |
|---|---|---|---|
| 128 | 37 MB | 74 MB | 0.6 GB |
| 2048 | 595 MB | 1.2 GB | 9.5 GB |
| 8192 | **2.38 GB** | **4.76 GB** | **38 GB** |

At 8K context, batch 8, you're at **38 GB of logits alone** — before weights,
KV cache, or activations. On an 80 GB H100 this OOMs instantly.

### Fix

Pass `num_logits_to_keep=1` (or `logits_to_keep=1` on older transformers).
This tells the model to only compute logits for the last token — the only one
you need during autoregressive decoding.

```python
# In forward() calls:
outputs = model(input_ids, use_cache=True, num_logits_to_keep=1)

# Or if using model.generate():
# generate() handles this internally, but during manual decode loops
# you MUST pass this yourself.
```

Both `bench_decode.py` and `bench_prefill.py` already do this. But if you write
any ad-hoc forward calls, **always include it**.

### Verify locally before you rent

```bash
# On your local machine with any tiny Qwen model:
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B', torch_dtype=torch.float32)
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')
x = torch.randint(0, 1000, (1, 512))

# This should use ~300 KB of logits, not 300 MB:
out = model(x, num_logits_to_keep=1)
print(f'Logits shape: {out.logits.shape}')  # Should be [1, 1, vocab_size]
assert out.logits.shape[1] == 1, 'num_logits_to_keep is not working!'
print('✓ Vocab trap mitigated')
"
```

---

## 2. nsys Will Produce a 4 GB Trace

An unrestricted `nsys profile` over 500 decode steps on a 7B model generates
**hundreds of thousands** of CUDA API calls. The `.nsys-rep` file hits 4+ GB
and Nsight Systems will choke trying to open it.

### Fix: Wrap only the steady-state region

In the benchmark code:

```python
torch.cuda.profiler.start()
# ... 20 decode steps, no more ...
torch.cuda.profiler.stop()
```

Then launch nsys with capture-range gating:

```bash
nsys profile \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    -t cuda,nvtx,osrt \
    --cuda-graph-trace=node \
    -o trace_b1_c8k \
    python bench_decode.py --model Qwen/Qwen2.5-7B \
        --batch-sizes 1 --context-lengths 8192 \
        --timed-steps 20 --profile
```

This gives you a ~200 MB trace covering exactly the 20 decode steps you care
about. Repeat for each case you need (batch 1, batch 64, short context, long
context).

### What to look for in the trace

1. **GPU idle gaps** — time between kernel launches where the GPU is doing
   nothing. At batch 1 this is often 40-60% of the step.
2. **Kernel launch count** — count the kernels per decode step. Expect ~400.
   Each launch costs ~5 μs of host overhead.
3. **Short kernels** — kernels under 10 μs are launch-overhead-dominated.
   These are candidates for kernel fusion or CUDA graphs.

---

## 3. Never `ncu --set full` on a Model

`ncu --set full` collects **every** hardware counter and replays the kernel
**multiple times**. On a model with 400+ kernel launches per step, this takes
**hours** and the output file is unusable.

### Fix: Target specific kernels

```bash
ncu --target-processes all \
    --kernel-name-base demangled \
    --kernel-name regex:"gemm|attention|flash" \
    --launch-skip 200 --launch-count 5 \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section Occupancy \
    --section WarpStateStats \
    -o ncu_b1_c8k \
    python bench_decode.py --model Qwen/Qwen2.5-7B \
        --batch-sizes 1 --context-lengths 8192 \
        --timed-steps 5
```

Key flags:

| Flag | Why |
|---|---|
| `--launch-skip 200` | Skip past model loading + prefill + warmup into steady-state decode |
| `--launch-count 5` | Profile only 5 kernel instances (enough to see the pattern) |
| `--kernel-name regex:"gemm\|attention\|flash"` | Only profile the kernels that matter |
| `--section` (4 specific) | Collect only the counter groups you need |

**Budget**: 4 kernels × 4 cases = 16 ncu runs. Even at 2-3 minutes each,
that's under an hour. Don't do more.

### Interpreting the output

| Section | Question it answers |
|---|---|
| `SpeedOfLight` | What % of peak compute and memory BW are you hitting? |
| `MemoryWorkloadAnalysis` | L2 hit rate, HBM read/write bytes, sector efficiency |
| `Occupancy` | Are you register/shared-memory limited? |
| `WarpStateStats` | Where are warps stalling? (memory, barrier, instruction fetch) |

---

## 4. The 5-Minute Smoke Test

Do this **the instant the instance boots**, before anything else:

```bash
# 1. Check that CUDA works at all
nvidia-smi

# 2. Compile and profile a trivial kernel — verifies hardware counters work
cat > /tmp/saxpy.cu << 'EOF'
#include <stdio.h>
__global__ void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
int main() {
    int N = 1 << 20;
    float *x, *y;
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));
    for (int i = 0; i < N; i++) { x[i] = 1.0f; y[i] = 2.0f; }
    saxpy<<<(N+255)/256, 256>>>(N, 2.0f, x, y);
    cudaDeviceSynchronize();
    printf("y[0] = %f (expected 4.0)\n", y[0]);
    cudaFree(x); cudaFree(y);
    return 0;
}
EOF

nvcc -o /tmp/saxpy /tmp/saxpy.cu && /tmp/saxpy
echo "✓ CUDA compilation and execution OK"

# 3. Verify ncu hardware counters are accessible
ncu --target-processes all \
    --section SpeedOfLight \
    --launch-count 1 \
    /tmp/saxpy
echo "✓ ncu hardware counters accessible"

# 4. Verify nsys works
nsys profile -t cuda -o /tmp/smoke_nsys /tmp/saxpy
echo "✓ nsys profiling OK"

# 5. Quick PyTorch + transformers check
python -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'BF16 supported: {torch.cuda.is_bf16_supported()}')
t = torch.randn(1024, 1024, device='cuda', dtype=torch.bfloat16)
_ = t @ t
print('✓ BF16 matmul OK')
"
```

> **If hardware counters are blocked, you want to know in minute 3, not hour 3.**
>
> Some cloud providers (RunPod, Lambda, etc.) block `ncu` by default unless you
> request a "bare metal" or "profiling-enabled" instance. If the smoke test
> fails on step 3, you need to either:
> - Request a different instance type with profiling enabled
> - Or run `ncu` inside a Docker container with `--privileged` and
>   `--cap-add=SYS_ADMIN`

---

## 5. Clock Speed Will Lie to You

The H100 does **not** run at the same SM clock for all workloads:

| Workload | Typical SM clock |
|---|---|
| Batch 1 decode (low power draw) | 1980 MHz (boost) |
| Batch 64 decode (near TDP) | 1590-1620 MHz (throttled) |
| Heavy prefill | 1410-1530 MHz |

This means a "2× batch, 2× throughput" scaling curve is actually
"2× batch, 2× throughput, 0.85× clock" — the real scaling is worse than it
looks, and the roofline comparison is wrong unless you normalize.

Both benchmark scripts log SM clock before and after timed regions. **Always
check these values.** If the clock drops >10% between cases, note it in your
results.

To lock clocks for fair comparison (requires root):

```bash
# Lock SM clock to a fixed frequency (e.g., 1410 MHz — sustainable under load)
sudo nvidia-smi -lgc 1410,1410
# Run benchmarks...
# Unlock when done
sudo nvidia-smi -rgc
```

---

## Quick Reference: Profiling Command Templates

### nsys — decode, batch 1, context 8K
```bash
nsys profile \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    -t cuda,nvtx,osrt --cuda-graph-trace=node \
    -o trace_b1_c8k \
    python bench_decode.py --model Qwen/Qwen2.5-7B \
        --batch-sizes 1 --context-lengths 8192 --timed-steps 20 --profile
```

### nsys — decode, batch 64, context 128
```bash
nsys profile \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    -t cuda,nvtx,osrt --cuda-graph-trace=node \
    -o trace_b64_c128 \
    python bench_decode.py --model Qwen/Qwen2.5-7B \
        --batch-sizes 64 --context-lengths 128 --timed-steps 20 --profile
```

### ncu — gemm/attention kernels, batch 1
```bash
ncu --target-processes all --kernel-name-base demangled \
    --kernel-name regex:"gemm|attention|flash" \
    --launch-skip 200 --launch-count 5 \
    --section SpeedOfLight --section MemoryWorkloadAnalysis \
    --section Occupancy --section WarpStateStats \
    -o ncu_b1_c8k \
    python bench_decode.py --model Qwen/Qwen2.5-7B \
        --batch-sizes 1 --context-lengths 8192 --timed-steps 5
```

### ncu — same but batch 64
```bash
ncu --target-processes all --kernel-name-base demangled \
    --kernel-name regex:"gemm|attention|flash" \
    --launch-skip 200 --launch-count 5 \
    --section SpeedOfLight --section MemoryWorkloadAnalysis \
    --section Occupancy --section WarpStateStats \
    -o ncu_b64_c128 \
    python bench_decode.py --model Qwen/Qwen2.5-7B \
        --batch-sizes 64 --context-lengths 128 --timed-steps 5
```
