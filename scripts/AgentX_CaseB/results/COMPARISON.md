# GLM-5.2-MXFP4 — Case-B conformance trace: serving results (single-node, MI355X)

**Trace:** the deterministic Case-B conformance corpus (`../caseB_conformance_corpus.tar.gz`,
seed 42, 300 sessions) — verified **13/13 axes** against the customer Case-B parameters
(ISL 62K/220K/500K, OSL 180/1.4K/7K, turns 5/82/144, delay 3.6/23/240s, cache 88%). This is the
**full Case-B distribution including the 500K-token input tail**.

- **Engine:** GLM-5.2-MXFP4, TP8, single node (8× MI355X), `--max-model-len 524288` (512K window).
- **vLLM:** `vllm-openai-rocm:v0.25.1-aiter-0.1.18`, `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1`,
  `--kv-cache-dtype fp8_e4m3`, `--max-num-seqs 32`, `--linear/--moe-backend aiter`.
- **Replay:** aiperf `inferencex-agentx-mvp` + weka_trace, conc 2/4/8, 300s/point.

## vLLM single-node — results

| Conc | Output tok/s (system) | tok/s/user | Req/s | TTFT p50 | TTFT p99 | ITL p50 | E2E p50 | E2E p90 | E2E p99 | ISL mean | OSL mean | Cache% |
|-----:|----------------------:|-----------:|------:|---------:|---------:|--------:|--------:|--------:|--------:|---------:|---------:|-------:|
| 2 | 36.2 | 59.3 | 0.07 | 6.4s | 30.6s | 14.9ms | 17.0s | 54.8s | 82.9s | 97,014 | 553 | 30.2 |
| 4 | 45.5 | 43.4 | 0.11 | 4.2s | 29.8s | 20.0ms | 15.7s | 60.0s | 127.8s | 81,297 | 429 | 32.7 |
| 8 | 58.5 | 27.3 | 0.15 | 4.7s | 32.3s | 51.1ms | 21.9s | 74.5s | 256.2s | 76,320 | 402 | 34.4 |

**Two throughput views:**
- **System output tok/s (36 → 45 → 58)** scales up with concurrency — aggregate decode work.
- **Per-user tok/s (59 → 43 → 27)** drops with concurrency — each stream slows as the node fills.
- **Req/s is very low (0.07-0.15)** because Case-B requests are huge (62-500K input) — minutes each.

## ATOM single-node — NOT MEASURED (environmental blocker)

ATOM (`atom.entrypoints.openai_server`, TP8, MTP×3 per the Case-B spec) **could not be
benchmarked on the current MI355X nodes.** The engine boots (health 200) but the **aiter JIT
compiler deadlocks (torch FileBaton) on cold kernel compilation** — an orphaned lock on
`module_gemm_a8w8_bpreshuffle` (and, per-request, `mha_varlen_fwd`), GPU drops to 0%, no recovery.

Reproduced across **2 image digests** (`:latest` = `a8539be1…`, and the pinned known-good
`280d2fe1…` from the Case-A work), **4 MAX_JOBS settings** (16/8/4), and **5 nodes** (several of
which were independently broken: NCCL init failure, docker daemon down, full root disk shared
with another tenant). This is a **serving-stack / node-environment issue, not a trace issue**:
the Case-A ATOM numbers succeeded only on a persistent node whose aiter JIT cache was already
**warm**; a cold compile on the current nodes hits the FileBaton stale-lock bug (FileBaton has no
stale-lock breaking, so the other TP ranks spin forever).

**To obtain ATOM Case-B numbers**, one of: (a) a node with a pre-warmed GLM-5.2 aiter JIT cache,
(b) an atom-dev build that fixes FileBaton stale-lock handling, or (c) a MAX_JOBS=1 fully-serial
JIT (very slow but avoids the concurrent-builder kill that orphans the lock). None were reliably
achievable in-session.

## Honest caveats
- **vLLM only** — no ATOM comparison point (blocker above).
- **Output is small** (OSL mean 400-550) because Case-B is prefill-dominated (huge input, modest
  output) and the 7K-output tail sessions rarely finish inside a 300s window. tok/s here is
  therefore *decode throughput on a prefill-heavy workload* — the engine spends most time on the
  62-500K prefills. A ≥900s window would let more long requests complete.
- **conc=1** not captured (aiperf agentic-replay needs conc≥2).
- Cache-hit measured 30-34% server-side (the constructed trace's reuse structure is 88%; realized
  hit depends on the engine's prefix-cache and the trajectory-start sampling).

## Reproduce
```
python3 ../gen_caseB_conformance.py corpus 300 42     # identical trace every time
python3 ../verify_caseB.py corpus                     # 13/13 conformance
URL=http://<endpoint> ../replay_caseB.sh              # sweep any engine/topology
```

## Files
- `vllm_summary.csv` — the sweep above
- `raw/vllm_caseB_c{2,4,8}.csv` — full aiperf metric CSVs
