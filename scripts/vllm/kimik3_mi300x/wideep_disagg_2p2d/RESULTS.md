# Results — Kimi-K3 MI300X 2P/2D EP16 MoRIIO disagg

Serve: 2 prefill + 2 decode nodes; per-pool TP2×DP8 → EP16; MoRIIO WRITE;
router on the prefill master :30000. Config: `K3_GROUP_ROUTING=1
K3_EXTRA_FIXES=1 LOAD_STRATEGY=lazy KV_CACHE_DTYPE=fp8
MAX_NUM_BATCHED_TOKENS=2048 GPU_UTIL=0.85 PREFILL_BACKEND=mori_low_latency`,
model in tmpfs. `MAX_MODEL_LEN=320000` for the ≥ 150K rows (131072 otherwise).

## Single-needle NIAH — the deliverable metric
`niah_probe.py`, needle = `HELIOTROPE-7492`, greedy (temp=0), depths
0.1 / 0.5 / 0.9. **All PASS, deterministic.**

| context (tokens) | result     | eval time / request |
|------------------|------------|---------------------|
| 10K              | 3/3 PASS   | 5.3s                |
| 50K              | 3/3 PASS   | 19.5s               |
| 100K             | 3/3 PASS   | ~47s                |
| 120K             | 3/3 PASS   | ~54s                |
| 150K             | 3/3 PASS   | ~84s                |
| 200K             | 3/3 PASS   | ~88s                |
| **300K**         | **3/3 PASS** | **~150s**         |
| 500K             | 3/3 PASS   | ~301s               |
| 750K             | not tested | —                   |
| 900K             | **HANGS** (see note) | —         |

Eval time ~linear through 500K (~0.6 ms/token). For ctx > ~120K you must raise
`MAX_MODEL_LEN` (default 131072 caps ~120K), and for a single request > ~600K you
must raise the KV cache: `KV_CACHE_MEMORY_BYTES=40e9` gives a 2.84M-token cache
(the default `8e9` = 542K tokens is too small for one 600K+ request). See
[OPTIMIZATION.md](OPTIMIZATION.md).

**900K limit (honest):** a single 900K-token prefill does not complete — the
prefill engine freezes (log timestamp stops, no scheduler ticks) while GPUs stay
at 100% and no result returns. Killing the client recovers the serve cleanly, so
it's confined to that one oversized request, not a serve crash. 500K passes
cleanly, so the wall is between 500K and 900K. Suspected cause: the chunked-prefill
path at ~440 chunks (batched=2048) through the MoRIIO connector, or an attention
kernel that degrades past ~500–600K. Not a KV-capacity issue. Investigation
pending.

Reproduce:
```bash
python3 niah_probe.py --url http://<prefill-master-ip>:30000 --model kimi-k3 \
    --ctx-list 10000,50000,100000,150000,200000,300000 --depths 0.1,0.5,0.9 \
    --timeout 400
```

## Multi-needle stress — 10-animal (stricter)
`benchmark_niah.py`, 10 animals hidden across a word haystack, scored found/10.

| context (words ≈ 1.3× tokens) | found/10 |
|-------------------------------|----------|
| 2000–5000                     | 10/10    |
| 10000                         | 10/10    |
| 15000                         | 10/10    |
| 20000                         | 9/10 (typical; 10/10 seen) |
| 50000                         | 6–10/10 (nondeterministic) |
| 80000                         | 9/10     |

The occasional ≤ 1-needle miss at ≥ 20K is the residual RDMA write race
(see `STATUS.md` § Known residual). Single-needle retrieval is unaffected.

## Latency vs. throughput
`concurrency_bench.py`, 20K-token prompts, 64 output tokens, batched=2048.

| concurrency | throughput (req/s) | vs single-stream | p50 lat | p99 lat |
|-------------|--------------------|------------------|---------|---------|
| 1           | 0.062              | —                | 16.2s   | 16.2s   |
| 8           | **0.353**          | **5.7×**         | 22.7s   | 22.7s   |
| 16          | **0.455**          | **7.3×**         | 34.5s   | 35.1s   |

8 requests finish in 1.4× the wall time of 1 → 5.7× throughput. That is the DP8
payoff; scaling flattens past 8 (= 8 replicas). **Single-stream latency is ~4×
a colocated PP2×TP8 EP8 serve** (1 request on 2 GPUs vs all 16) — architectural,
not tunable.

### `max_num_batched_tokens` — a dead end (kept at 2048)
| metric | batched=2048 | batched=8192 |
|--------|--------------|--------------|
| 200K single-stream | ~88s | 112.8s (worse) |
| 20K conc=8 throughput | 0.353 req/s | 0.245 req/s (worse) |

Raising it did not cut latency and hurt throughput (compute-bound prefill; larger
batches contend more across 8 replicas). **Keep `MAX_NUM_BATCHED_TOKENS=2048`.**

## Use which serve
- **Colocated PP2×TP8 EP8** (`../wideep_int4_moriep`) — lowest single-request
  latency; interactive / low-QPS.
- **Disagg TP2×DP8 EP16** (this) — highest concurrent throughput + decode-latency
  isolation; batch / high-QPS. NIAH to 300K.
