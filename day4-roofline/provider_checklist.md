# Provider Checklist — Before You Rent

> Spend 10 minutes on this checklist before you spend \$15–25 on GPU time.
> Every item here has cost someone real money to learn by not checking.

---

## 1. Profiling Counters (This WILL Block You)

`ncu` (Nsight Compute) needs kernel-level hardware counters. Most cloud
providers disable these by default.

### What you need

| Requirement | How to check |
|---|---|
| **Driver flag** `NVreg_RestrictProfilingToAdminUsers=0` | `cat /proc/driver/nvidia/params \| grep Restrict` → should be `0` |
| **Container flag** `--cap-add=SYS_ADMIN` | If in Docker/Podman, this must be set at container launch |
| **No `nv-fabricmanager` interference** | `systemctl status nvidia-fabricmanager` — if active, it can block counters on multi-GPU nodes |

### Provider compatibility (as of mid-2025)

| Provider | ncu counters? | Notes |
|---|---|---|
| **Lambda on-demand** | ✅ Yes | Bare metal, no container. Works out of the box. |
| **Crusoe** | ✅ Yes | Bare metal. |
| **RunPod bare metal** | ✅ Yes | Must select "Bare Metal" template, not "Community" |
| **RunPod community templates** | ❌ No | Docker without `SYS_ADMIN`. ncu will silently return zeros. |
| **Vast.ai** | ⚠️ Varies | Depends on host config. Ask host before renting. |
| **AWS p5** | ✅ Yes | But requires `--cap-add=SYS_ADMIN` in ECS/Docker |
| **GCP a3** | ✅ Yes | Works with `--privileged` containers |
| **CoreWeave** | ⚠️ Varies | Some instances restrict counters |

### The smoke test (run in minute 1)

```bash
# Compile trivial kernel
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

# THIS is the test that matters:
ncu --target-processes all --section SpeedOfLight --launch-count 1 /tmp/saxpy

# If you see "ERR_NVGPUCTRPERM" or all counters are 0.00% → counters are blocked.
# Stop. Fix the instance. Don't waste an hour discovering this during the benchmark.
```

---

## 2. GPU Variant — SXM5 vs PCIe Is a 1.7× Difference

You are measuring **HBM bandwidth utilisation**. The GPU variant determines
the denominator of your entire experiment:

| Variant | HBM Bandwidth | Memory | Relative |
|---|---|---|---|
| **H100 PCIe** | 2.0 TB/s | 80 GB | 1.0× |
| **H100 SXM5** | 3.35 TB/s | 80 GB | 1.68× |
| **H200 SXM** | 4.8 TB/s | 141 GB | 2.4× |

If you rent an H100 PCIe thinking it's SXM5, your roofline floor is off by 68%.
Every measurement is wrong. Every conclusion is wrong.

### How to verify

```bash
# Record this in the repo — one command, full hardware spec
nvidia-smi -q > gpu_info.txt
git add gpu_info.txt && git commit -m "Record GPU hardware spec"

# Quick check for variant:
nvidia-smi --query-gpu=name,pci.bus_id,memory.total,clocks.max.sm \
    --format=csv,noheader
# "H100 SXM" vs "H100 PCIe" will be in the name
```

### H200 consideration

If an H200 is available at similar price, **take it**:
- 4.8 TB/s HBM bandwidth → lower floor, more headroom to see engineering losses
- 141 GB HBM → batch 64 × 8K context fits comfortably (44.6 GB traffic on 141 GB)
- The batch×context sweep gets much more headroom before OOM

---

## 3. Cost Reality Check

This experiment is **~4–6 GPU-hours**, not a research training run.

| Provider | GPU | Price/hr | Total (5 hrs) |
|---|---|---|---|
| Lambda | H100 SXM | \$2.49/hr | ~\$12.50 |
| RunPod | H100 SXM | \$3.29/hr | ~\$16.50 |
| Crusoe | H100 SXM | \$2.85/hr | ~\$14.25 |
| Vast.ai | H100 SXM | ~\$2.50/hr | ~\$12.50 |

**Budget: \$15–25, not \$150.**

### Minimise paid time — develop locally first

Do **all** development and debugging on cheap hardware:

```
Local (free):
  - Mac/CPU with Qwen2.5-0.5B → validates script logic, arg parsing, JSON output
  - Catches import errors, off-by-one bugs, file paths

4090 cloud ($0.40/hr):
  - Validates CUDA path, CUDA events, NVML energy counter
  - Run full sweep with 0.5B model → verifies OOM guard, profiler hooks

H100 window ($2.50–3.30/hr):
  - Nothing but: run, capture, rsync down
  - Should take 2–3 hours of actual GPU time
  - Budget 5 hours for safety (re-runs, debugging edge cases)
```

### Persistent volume — don't re-download the model

The Qwen2.5-7B weights are **~15 GB**. Downloading them takes 10-20 minutes on
cloud instances. If you restart the instance for Day 5, you lose them.

```bash
# On first boot: cache model to persistent volume
export HF_HOME=/workspace/.hf_cache   # or wherever the persistent vol is mounted

# Download once (this persists across instance restarts):
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B', torch_dtype='auto')
print('✓ Model cached')
"

# For Day 5+, re-spin the instance with the same persistent volume.
# Model loads from local cache in ~30 seconds, not 20 minutes.
```

Also `rsync` traces and results down immediately after each run:

```bash
# From your local machine:
rsync -avz user@gpu-instance:/workspace/day4-roofline/results_*.json ./results/
rsync -avz user@gpu-instance:/workspace/day4-roofline/*.nsys-rep ./traces/
rsync -avz user@gpu-instance:/workspace/day4-roofline/*.ncu-rep ./traces/
```

---

## 4. Pre-Flight Script

Run this as a single script the moment the instance boots.
If any step fails, **stop and fix before proceeding.**

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "═══ Day 4 Pre-Flight Check ═══"

# 1. GPU identity
echo -e "\n▸ GPU Info:"
nvidia-smi --query-gpu=name,pci.bus_id,memory.total,driver_version \
    --format=csv,noheader

# 2. Save full GPU spec to repo
nvidia-smi -q > gpu_info.txt
echo "  Saved gpu_info.txt"

# 3. CUDA compilation
echo -e "\n▸ CUDA compilation:"
cat > /tmp/saxpy.cu << 'SAXPY'
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
    printf("  y[0] = %f (expected 4.0)\n", y[0]);
    cudaFree(x); cudaFree(y);
}
SAXPY
nvcc -o /tmp/saxpy /tmp/saxpy.cu && /tmp/saxpy
echo "  ✓ CUDA OK"

# 4. ncu hardware counters
echo -e "\n▸ ncu hardware counters:"
ncu --target-processes all --section SpeedOfLight --launch-count 1 \
    /tmp/saxpy 2>&1 | head -20
echo "  ✓ ncu OK (check output above for ERR_NVGPUCTRPERM)"

# 5. nsys
echo -e "\n▸ nsys profiling:"
nsys profile -t cuda -o /tmp/smoke_nsys --force-overwrite true /tmp/saxpy >/dev/null 2>&1
echo "  ✓ nsys OK"

# 6. NVML energy counter
echo -e "\n▸ NVML energy counter:"
python3 -c "
import pynvml
pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(0)
e = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
p = pynvml.nvmlDeviceGetPowerUsage(h) / 1000
t = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(h)
print(f'  Energy counter: {e} mJ')
print(f'  Current power:  {p:.0f} W')
print(f'  Throttle mask:  0x{t:x}')
print('  ✓ NVML energy OK')
"

# 7. PyTorch + BF16
echo -e "\n▸ PyTorch:"
python3 -c "
import torch
print(f'  PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'  GPU: {torch.cuda.get_device_name(0)}')
print(f'  BF16: {torch.cuda.is_bf16_supported()}')
t = torch.randn(1024, 1024, device='cuda', dtype=torch.bfloat16)
_ = t @ t
print('  ✓ BF16 matmul OK')
"

# 8. Transformers + model access
echo -e "\n▸ Model access:"
python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B')
print(f'  Vocab size: {tok.vocab_size}')
print('  ✓ Model access OK')
"

echo -e "\n═══ All checks passed. Ready to benchmark. ═══"
```

Save as `day4-roofline/preflight.sh` and `chmod +x` it.
