# Roofline Predictions — Qwen2.5-7B BF16 Decode

> Committed **before** any measurement run.
> The value of the experiment is the gap between this table and reality.

---

## 1. Model Architecture (from [Qwen2.5-7B config](https://huggingface.co/Qwen/Qwen2.5-7B))

| Parameter | Value |
|---|---|
| Layers | 28 |
| Q heads | 28 |
| KV heads | 4 |
| Head dim | 128 |
| Hidden size | 3 584 |
| Vocab size | 152 064 |
| Precision | BF16 (2 bytes/param) |

## 2. Weight and KV Traffic

### Total weight bytes

The model has **~7.62 B parameters** (verify via `model.num_parameters()`).

$$
W = 7.62 \times 10^9 \times 2\;\text{B} = 15.2\;\text{GB}
$$

Every decode step at batch 1 must stream all weights through the GPU exactly once
(each linear layer is a matrix–vector product, arithmetic intensity ≈ 1).

### KV cache bytes per token

Each token stored in the KV cache occupies:

$$
\text{KV/token} = 2 \;\times\; n_\text{layers} \;\times\; n_\text{kv\_heads} \;\times\; d_\text{head} \;\times\; 2\;\text{B}
$$

$$
= 2 \times 28 \times 4 \times 128 \times 2 = 57{,}344\;\text{B} = 56\;\text{KiB/token}
$$

The **total KV traffic per decode step** for a batch of $B$ sequences, each at context
length $S$:

$$
\text{KV}_\text{total} = B \times S \times 56\;\text{KiB}
$$

## 3. Hardware Bandwidth

| GPU | HBM Bandwidth |
|---|---|
| H100 SXM5 | 3.35 TB/s |
| H100 PCIe | 2.0 TB/s |
| H200 SXM | 4.8 TB/s |

All floors below use the **H100 SXM5 @ 3.35 TB/s**.

## 4. Decode-Step Latency Floor

The minimum time for one decode step is set by the time to stream all
memory-bound traffic through HBM. 

$$
t_\text{floor} = \frac{W + \text{KV}_\text{total}}{\text{BW}_\text{HBM}}
$$

Throughput in tokens/second (batch of $B$):

$$
\text{tok/s} = \frac{B}{t_\text{floor}}
$$

### Prediction table

| Case | Weights | KV traffic | Total bytes | $t_\text{floor}$ | tok/s |
|---|---|---|---|---|---|
| B=1, ctx=128 | 15.2 GB | 7 MB | 15.21 GB | 4.54 ms | **~220** |
| B=1, ctx=8K | 15.2 GB | 0.47 GB | 15.67 GB | 4.68 ms | **~214** |
| B=64, ctx=128 | 15.2 GB | 0.46 GB | 15.66 GB | 4.67 ms | **~13,700** |
| B=64, ctx=8K | 15.2 GB | 29.4 GB | 44.6 GB | 13.3 ms | **~4,800** |

### What this table already tells us

1. **B=1: KV traffic is a rounding error.** Weight traffic dominates. Adding
   context barely changes latency. Batching is nearly free — you're paying for
   the weight read regardless.

2. **B=64, ctx=128: batching is almost free.** KV traffic is still only 3% of
   total. You get ~64× the tokens for ~1.03× the time. This is the regime where
   the GPU looks efficient and throughput tracks the bandwidth roofline.

3. **B=64, ctx=8K: KV traffic is 2× the weight traffic.** This is the
   phase transition. Attention becomes the bottleneck; batching stops being
   free; latency per token degrades. This is why:
   - **FP8 KV cache** halves KV traffic → reclaims the short-context floor.
   - **Paged attention** avoids wasting bandwidth on padding.
   - **Sliding window / sparse attention** limits $S$ in the KV term.

## 5. Compute Check. Are We Actually Memory-Bound?

A decode step is memory-bound when the arithmetic intensity (FLOPs/byte) is
below the machine's ridge point.

**FLOPs per decode step** (batch 1, ignoring attention):

$$
\text{FLOPs} \approx 2 \times P = 2 \times 7.62 \times 10^9 = 15.24\;\text{TFLOP}
$$

Wait — that's **15.24 GFLOPs**, not TFLOPs. The H100 does **990 TFLOPS** BF16.

$$
\text{Arithmetic intensity} = \frac{15.24 \times 10^9\;\text{FLOPs}}{15.2 \times 10^9\;\text{B}} \approx 1\;\text{FLOP/B}
$$

$$
\text{Ridge point} = \frac{990 \times 10^{12}}{3.35 \times 10^{12}} \approx 295\;\text{FLOP/B}
$$

At AI ≈ 1, we are **~295× below the ridge point**. Decode at batch 1 is
unambiguously memory-bandwidth-bound. Compute is irrelevant. The only number
that matters is how fast you can stream weights.

At **batch 64**, AI rises to ~64 — still well below the ridge point. Even large
batches don't make decode compute-bound on an H100; memory bandwidth remains
king.

## 6. Energy Prediction

H100 SXM5 TDP: **700 W**.

| Case | tok/s | tokens/joule |
|---|---|---|
| B=1 | ~220 | ~0.31 |
| B=64, ctx=128 | ~13,700 | ~19.6 |
| B=64, ctx=8K | ~4,800 | ~6.9 |

At batch 1, the GPU spends 700 W to produce 220 tokens/s — **0.31 tokens per
joule**. At batch 64 short-context, the same 700 W produces 13,700 tok/s —
**63× more energy-efficient**. Batching is an energy strategy as much as a
throughput one.

## 7. What the Measurement Will Reveal

The measured decode latency will be **slower** than the floor. The gap is the
engineering loss — the part you can actually fix. Expected sources:

| Source | Expected impact |
|---|---|
| **Kernel launch overhead** | ~400 launches/token × ~5 μs each = ~2 ms idle time per step |
| **Python/framework overhead** | GIL contention, tensor allocation, scheduler bookkeeping |
| **Memory-access inefficiency** | Non-coalesced reads, TLB misses, bank conflicts |
| **Attention kernel** | Naive attention isn't bandwidth-optimal for GQA's 7:1 ratio |
| **Quantization headroom** | BF16 weights are 2× larger than FP8; KV cache 2× larger than FP8 |

The Day 4 result will be:

> *"Measured X ms per decode step — Y% of the 4.5 ms analytical floor.*
> *Nsight showed the GPU idle Z% of the time. Here's where the rest went."*

Three bottlenecks → three fixes:
- **CUDA graphs** → eliminate launch overhead (Day 5)
- **Paged attention kernel** → bandwidth-efficient GQA attention (Day 6)
- **FP8 KV cache** → halve KV traffic, reclaim the roofline at long context (Day 8)

---

*Predictions computed on paper. No GPU was rented to produce this file.*
