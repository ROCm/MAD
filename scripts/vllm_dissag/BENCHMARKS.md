# Long-context and SLO benchmarks

Three benchmarks that answer three different questions about a long-context serving stack.
All of them run against **any** live OpenAI-compatible endpoint — a cluster job, a single
container, or a hosted URL. None of them require the launcher.

| Benchmark | Question it answers | Entry point |
|---|---|---|
| **NIAH** | Does retrieval still work at the context length we are claiming? | `benchmark_niah_long.sh` / `benchmark_niah.py` |
| **avg 80K / 256K ctx** | Do we meet the TTFT/TPOT SLO under the customer's smaller workload? | `benchmark_avg_80K.sh` |
| **avg 200K / 1M ctx** | Same, under the larger one. | `benchmark_avg_200K.sh` |

Run NIAH **first**. A throughput number from a server that has silently stopped attending
past 500K is not a result — it is a fast wrong answer. `BENCHMARK_SCRIPT=niah_perf` encodes
that ordering by skipping the perf phase when NIAH fails.

---

## Contents

- [What you need](#what-you-need)
- [1. NIAH — long-context retrieval](#1-niah--long-context-retrieval)
- [2. The avg-ISL SLO benchmarks](#2-the-avg-isl-slo-benchmarks)
- [Reading the output](#reading-the-output)
- [Running under the launcher](#running-under-the-launcher)
- [Running against a non-vLLM server](#running-against-a-non-vllm-server)
- [Failure modes worth recognising](#failure-modes-worth-recognising)

---

## What you need

| Tool | Needs | Runs anywhere? |
|---|---|---|
| `benchmark_niah.py` | Python 3 stdlib only | **Yes.** No vLLM, no `transformers`, no GPU. |
| `benchmark_niah.py` with `NIAH_TOKENS` | `transformers` (for exact token calibration) | Degrades loudly to a words×1.30 estimate without it |
| `gen_workload.py` | `transformers` + a local tokenizer directory | Tokenizer must be on disk |
| `benchmark_customer_slo.sh` | `vllm bench serve` on PATH | Driver only; the *server* can be anything |
| `pool_workload.py`, `slo_report.py` | Python 3 stdlib only | Yes |

The server under test needs **no** special build. It needs to admit the context lengths you
are testing — see the `--max-model-len` note below.

---

## 1. NIAH — long-context retrieval

Ten animal names are planted in filler text at ~9% depth intervals; the model is asked to
list them back. The score is how many of the ten came back, and — new — **the depth of every
miss**.

That depth is the entire diagnostic. A bare `7/10` says retrieval is degraded. `7/10 with
all three misses past 80% depth` says the **tail of the context is being dropped**, which
points at a completely different bug from three scattered misses.

### Standalone, against any endpoint

```bash
NIAH_URL=https://my-endpoint/v1/chat/completions \
NIAH_MODEL=my-model \
NIAH_TOKENS=32768,131072,262144 \
NIAH_TOKENIZER=/models/GLM-5.2-FP8 \
  python3 benchmark_niah.py
```

### Words vs tokens — read this before setting a target

`NIAH_WORDS` is the historical knob and it measures **words**. The customer's numbers are
**tokens**. For this filler the ratio is ~1.30, so `NIAH_WORDS=950000` is about **1.24M
tokens** — *above* a 1,048,576 window. vLLM **rejects** an over-length request with a 400
rather than truncating it, and that arrives at the client as a transport error that reads
exactly like a dead server.

Use `NIAH_TOKENS` instead. It calibrates against the real tokenizer and deliberately
approaches the target **from below**: overshooting a window is a hard failure, undershooting
is merely a slightly shorter test.

Without `transformers` installed it falls back to `NIAH_TOKENS_PER_WORD` (default 1.30) and
prints a loud `UNCALIBRATED` warning. **Do not quote those lengths as exact token counts.**

### The full ladder to 950K

```bash
MODEL_PATH=/models/GLM-5.2-FP8 BENCHMARK_PORT=30000 ./benchmark_niah_long.sh
```

Default ladder: `32768,65536,131072,262144,393216,524288,786432,950000`.

**Budget the wall clock before you start.** Prefill is roughly **quadratic** in context
length here — the sparse indexer scores each 8,192-token prefill chunk against all preceding
keys (`index_topk=2048` caps the *decode* key set, not the prefill scan). Scaling from a
measured 4.80 s TTFT at 28,672 tokens:

```
 32,768 ->  0.1 min      393,216 -> 15.0 min
 65,536 ->  0.4 min      524,288 -> 26.7 min
131,072 ->  1.7 min      786,432 -> 60.2 min
262,144 ->  6.7 min      950,000 -> 87.8 min
```

| configuration | total |
|---|---|
| 1 seed, no warmup (**the default**) | **3.3 h** |
| 1 seed + warmup | 6.6 h |
| 3 seeds + warmup | 13.2 h |

The last two rungs are **74%** of the total. If the budget gets tight, dropping `950000`
alone buys back 44% of the ladder.

This is why `benchmark_niah_long.sh` defaults to `NIAH_SEEDS=0` and `NIAH_WARMUP=0`, unlike
`benchmark_niah.py` (which defaults to three seeds with warmup). Override when you have the
budget.

Timeouts scale **per rung**, quadratically, floored at 300 s. `NIAH_TIMEOUT` means "the
budget for the longest rung." A single flat timeout sized for 950K would mean a dead server
burns 4.4 h on the 32K rung and eight rungs blow the job wall clock without printing a line.

### NIAH variables

| Variable | Default | Meaning |
|---|---|---|
| `NIAH_URL` | `http://127.0.0.1:30000/v1/chat/completions` | endpoint |
| `NIAH_MODEL` | *(empty)* | model name sent in the request |
| `NIAH_TOKENS` | *(empty)* | ladder in **tokens** — preferred |
| `NIAH_WORDS` | `2000,8000,20000,35000` | ladder in **words** — legacy |
| `NIAH_TOKENIZER` | `$NIAH_MODEL` | tokenizer for calibration |
| `NIAH_TOKENS_PER_WORD` | `1.30` | fallback ratio when `transformers` is absent |
| `NIAH_SEEDS` | `0,1,2` (`0` in `_long`) | needle-placement seeds. Seed 0 = evenly spaced |
| `NIAH_WARMUP` | `1` (`0` in `_long`) | warm each shape before scoring |
| `NIAH_TIMEOUT` | `1800` | seconds, **for the longest rung** |
| `NIAH_TIMEOUT_SCALE` | `1` | scale shorter rungs down quadratically |
| `NIAH_MIN_SCORE` | `8.0` | pass threshold, out of 10 |

### Exit codes

| Code | Meaning |
|---|---|
| `0` | pass |
| `3` | some rung produced **no usable result** — dead server, timeout, or rejection |
| `4` | every rung scored, but some mean was below `NIAH_MIN_SCORE` — retrieval is broken |

`3` and `4` are deliberately distinct: "we could not measure it" and "we measured it and it
is bad" call for completely different next steps.

---

## 2. The avg-ISL SLO benchmarks

```bash
MODEL_PATH=/models/GLM-5.2-FP8 BENCHMARK_PORT=30000 ./benchmark_avg_80K.sh
MODEL_PATH=/models/GLM-5.2-FP8 BENCHMARK_PORT=30000 ./benchmark_avg_200K.sh
```

Each drives one customer row: an **average** ISL against a **context window**, at a finite
Poisson arrival rate, with `--goodput` thresholds and a pass/fail exit code.

### Why the workload is a distribution, not a fixed length

The sheet gives an *average* ISL and a *context window*. A fixed-length run at the average
never sends a single request near the window — so it would report a green result for a server
that cannot serve the window at all. `gen_workload.py` instead solves a lognormal whose mean
is the sheet's average and whose **p99 lands on the sheet's context window**, then emits it as
a `--dataset-name custom` JSONL.

### Why two passes: one warm-up, one measured

The default shape of a run is **two passes over the scenario**, not one:

| Pass | Controlled by | Results |
|---|---|---|
| warm-up | `SLO_WARMUP_ITERS` (default `1`) | written to `$RESULT_DIR/_warmup`, **discarded** |
| measured | `SLO_ITERS` (default `1`) | written to `$RESULT_DIR`, **judged** |

One pass is not enough, and the reason is not subtle. The first pass over a shape absorbs
every one-off cost in the stack: Triton/aiter JIT for shapes the engine has not seen, the
decode cudagraph captures, the MoRI dispatch buffers, and the first fill of the prefix cache.
Measured on this platform that inflated TPOT from **~89 ms steady state to 302 ms**. A
single-pass run reports 302 ms as the answer.

The warm-up runs the **full concurrency list** over the **same trace** the measurement will
use — same seed, same JSONL. Warming with a different draw warms the wrong prompt lengths and
leaves the shared agentic prefix cold, which is exactly what `PREFIX_FRAC` exists to exercise.
(This is also why `RESAMPLE_PER_ITER` must stay `0` for the warm-up to be meaningful; under
`RESAMPLE_PER_ITER=1` the engine warms on seed `SEED_BASE`, which is the trace measured
iteration 1 will use.)

There is a *separate*, much cheaper `[WARMUP]` step before this: 4 prompts at con=2 on the
mean shape. It stays because it fails fast — if the server cannot serve 4 prompts, learning
that in seconds beats learning it an hour into a 262K sweep — but it cannot warm the cudagraph
batch sizes for con=64, the long-tail prefill path, or the shared prefix.

Two independent guards keep warm-up numbers out of the verdict, neither of them a filename
convention someone can break by renaming a file:

- results are **routed to a different directory**, and `slo_report.py` globs `$RESULT_DIR/*.json`
  non-recursively;
- warm-up cells log `[WARMUP-PASS]`, never `[RUNNING]`, which is the anchor `benchmark_parser.py`
  keys on.

Set `SLO_WARMUP_ITERS=0` for a deliberate cold-start measurement — and report it as cold.

### What one sample can and cannot tell you

A few-hundred-request draw from a right-skewed distribution has a wobbly realised mean:

```
CV = sqrt(exp(sigma^2) - 1),   SE(mean)/mean = CV / sqrt(n)

256K row: sigma 0.5833 -> CV 0.637, n=256 -> 4.0%
1M   row: sigma 0.8778 -> CV 1.078, n=128 -> 9.5%
```

Measured across 12 seeds the realised mean actually spanned **14.2%** and **35.4%**. One
run's mean is a property of that seed, not of the workload.

**The default does not try to make that go away. It reports it.** Every run writes
`<workload>.meta.json` with the achieved mean, its deviation from target, and the seed, and
`pool_workload.py` recomputes percentiles from the raw lengths. So the claim to make is:

> our trace averaged 78,412 tokens, 2.0% below the 80,000 target, p99 262,144

and **not** "we ran the customer's 80K average" — the second claim is the one that gets
challenged, and a single seed cannot defend it.

If the target must be pinned, there is exactly one mechanism that works, and it is still here:

```bash
SLO_ITERS=10 RESAMPLE_PER_ITER=1 ./benchmark_avg_80K.sh
```

Ten runs at **distinct** seeds pool to `n×10` and the error falls as `1/sqrt(10)`: to **1.3%**
and **3.0%**. It costs ten times the wall clock, which is why it is not the default. (Reaching
1.3% inside a *single* run would need n=1,014 requests at ~80K tokens each — more than
`MAX_PROMPTS` and far more wall clock than ten smaller runs, for the same statistical answer.)

So `RESAMPLE_PER_ITER` selects which question the iterations answer:

- `RESAMPLE_PER_ITER=1` — fresh sample per iteration (seed = `SEED_BASE + iter - 1`).
  Iterations **pool**. Use when characterising the customer's average.
- `RESAMPLE_PER_ITER=0` *(default)* — one sample reused. Required for the warm-up pass to warm
  the right trace; with `SLO_ITERS>1` it also makes iterations measure **server variance on a
  fixed workload**, which is the right question when A/B testing a code change.

Both are correct; they answer different questions. Repeating one sample ten times cannot
reduce its sampling error no matter how many times you run it.

### Why not stratify

Inverse-CDF placement would pin the mean to 1–2% with no pooling at all. It is rejected
because it caps the sample at the `(n-0.5)/n` quantile, so the p99 would reach only ~225K
instead of 262,144. **The window is the number the customer actually stated.** We do not
trade it away to tidy up a number they did not state.

### SLO variables

| Variable | Default | Meaning |
|---|---|---|
| `SCENARIOS` | see below | `label:mean_isl:osl:context_window:concurrency-list`, `\|`-separated |
| `SLO_ITERS` | `1` | measured passes (judged) |
| `SLO_WARMUP_ITERS` | `1` | discarded passes before them; `0` = cold-start measurement |
| `RESAMPLE_PER_ITER` | `0` | `1` = fresh seed per measured iteration, so iterations pool |
| `SEED_BASE` | `0` | first seed; run 2 can use `SEED_BASE=10` to extend the pool |
| `SLO_TTFT_MS` | `7000` | TTFT SLO |
| `SLO_TPOT_MS` | `50` | TPOT SLO |
| `TAIL_FRAC` | `1.0` | p99 ISL as a fraction of the window. `1.0` = p99 fills it |
| `PREFIX_FRAC` | `0.5` | shared cacheable preamble fraction — **a declared assumption, sweep it** |
| `MAX_PROMPTS` | `512` | cap on requests per cell |
| `DP_RANKS` | `8` | to convert aggregate throughput to per-rank |
| `BURSTINESS` | `1.0` | 1.0 = Poisson |

Default scenarios:
```
256k-ctx:80000:1024:262144:16 32 64
1m-ctx:200000:1024:1048576:8 16 32
```

`PREFIX_FRAC=0.5` deserves a flag when you report: the customer said "agentic" and "interested
in long prefix caching" but gave **no reuse ratio**. 0.5 is our assumption, not their spec.

---

## Reading the output

Each wrapper ends by printing the distribution **actually served** — the one measured trace by
default, or the pool if you asked for ten seeds:

```
Per-run realised means (this spread IS the sampling error, not a bug):
  seed=0    n=256   mean=78967     p99=262144      [-1.3%]

POOLED over 1 run, n=256 requests:
  mean = 78967
  p50  = 67104      p95 = 181233      p99 = 262144      max = 262144
  vs target mean 80000: -1.3%
  vs context window 262144: p99 reaches 100.0% of it, 3 requests at the window
```

**Report this block, not the target** — it is what was actually served. With the default single
trace, the mean carries the full sampling error of one draw (4.0% on the 256K row, 9.5% on the
1M row), so quote it as *"our trace averaged 78,967, 1.3% below target"* rather than as the
customer's stated average.

Under `SLO_ITERS=10 RESAMPLE_PER_ITER=1` the same block pools ten seeds and the deviation
collapses. Measured that way:

| row | per-run spread | pooled mean (n×10) | pooled p99 |
|---|---|---|---|
| avg 80K / 256K | 10.4% | **+0.6%** | 262,144 (100.0% of window) |
| avg 200K / 1M | 31.2% | **−1.4%** | 1,048,576 (100.0% of window) |

The **per-run spread column is the point**: that is how far a single default run's mean can
land from the target, and it is why the deviation is reported rather than assumed away.

`pool_workload.py` pools **raw lengths and re-computes**, never averages summaries.
Percentiles do not average — the mean of ten p99s is not the p99 of the pool, and for a heavy
tail it is materially lower. It also **refuses** to pool samples with different targets,
because averaging an 80K sample with a 200K one describes no workload at all.

`slo_report.py` prints the verdict and exits `0` pass / `1` fail / `2` no results. It gates on
the **mean**, because the sheet says "avg". Its `prefill_pr` figure is an **end-to-end lower
bound**, not a kernel rate.

---

## Running under the launcher

```bash
BENCHMARK_SCRIPT=niah_long  sbatch run_xPyD_models.slurm
BENCHMARK_SCRIPT=avg_80k    sbatch run_xPyD_models.slurm
BENCHMARK_SCRIPT=avg_200k   sbatch run_xPyD_models.slurm
```

Valid tags: `sweep` (default), `long_context`, `keepalive`, `niah`, `niah_long`, `niah_perf`,
`customer_slo`, `avg_80k`, `avg_200k`. An unrecognised tag exits non-zero rather than falling
back to the default sweep.

Select via **`BENCHMARK_SCRIPT`**, never `BENCHMARK_SCRIPT_FILE`. The launcher's `case`
assigns `BENCHMARK_SCRIPT_FILE` with a plain `=`, so an inherited value is silently
discarded — several runs executed the default throughput sweep instead of the intended
benchmark before this was spotted.

Two container facts that bite:

- **Only `/run_logs` is host-backed** (`-v ${LOG_PATH}:/run_logs`). Anything written elsewhere
  lands on the container's writable overlay and is destroyed on exit. Both wrappers default
  their `LOG` under `/run_logs/${SLURM_JOB_ID}/`.
- **Any env var not in the launcher's docker `-e` block is silently dropped** inside the
  container. All variables in this document are listed there. If you add a new one, add it
  there too or it will simply not exist at runtime.

---

## Running against a non-vLLM server

`benchmark_niah.py` is stdlib-only and speaks plain OpenAI chat-completions — it works
against anything, unchanged.

The SLO benchmarks drive `vllm bench serve`, which defaults to `--backend vllm`: vLLM's own
completions dialect plus per-token timing extensions a generic OpenAI server does not have.
For SGLang, TGI, or a hosted endpoint:

```bash
BENCH_BACKEND=openai-chat \
BENCH_EXTRA_ARGS="--endpoint /v1/chat/completions" \
IGNORE_EOS=0 \
  ./benchmark_avg_80K.sh
```

`IGNORE_EOS=0` has a real cost, so state it when you report. `--ignore-eos` is what makes OSL
**exact**; without it the model stops when it wants to, and the measured TPOT is an average
over a length you did not choose. The 50 ms/token verdict is then approximate — **do not
quote it as the sheet's number.**

---

## Failure modes worth recognising

**An over-length request is rejected, not truncated.** vLLM answers 400. Rejected requests
vanish from the latency statistics and show up *only* as a lower `completed` count — so the
failure presents as a **suspiciously good** result. `slo_report.py` prints `completed`; check
it against `num-prompts` every time.

**`--max-model-len` is the prerequisite for the long rows, and it is NOT satisfied by default.**
GLM-5.2 declares `max_position_embeddings=1048576`, so an unmodified vLLM inherits the full
window. **This tree does not**: `models.yaml` passes `--max-model-len ${GLM_MAX_MODEL_LEN:-65536}`,
so out of the box every request over 65,536 tokens is 400'd with the silent signature above.
Export the window you intend to serve at launch — `GLM_MAX_MODEL_LEN=262144` for the 256K row —
and check `completed` against `num-prompts` in the report to confirm it took effect.

(An earlier version of this document asserted the opposite. It was true of upstream vLLM and
false here, and the failure it caused reads as a *better* result, not a worse one.)

**`/v1/models` is not a readiness probe on this router.** It has returned 200 while every
request 503'd, *and* returned 503 while completions succeeded — its listing path and its
forwarding path consult different state. Both drivers poll with a **real completion** instead.

**`| tee` always exits 0.** A failing benchmark piped through `tee` reports success. Both
drivers invoke Python directly and recover the real status via `set -o pipefail` +
`PIPESTATUS[0]`. Preserve that if you edit them.

**KV arithmetic, for sizing.** GLM-5.2's MLA stores `kv_lora_rank 512 + qk_rope_head_dim 64
= 576` elements per token per layer, FP8, over 78 layers → **43.88 KiB/token** by element
count. The figure to size with is **46.58 KiB/token**, which is what vLLM actually allocates
on MI308X — it rounds each layer's allocation to whole blocks, a 6.2% overhead the element
math does not see. Every script here defaults to the measured number; override with
`KV_BYTES_PER_TOKEN` if you re-measure on different hardware.

So one request costs:

| tokens | KV |
|---|---|
| 28,672 | 1.27 GiB |
| 80,000 | 3.55 GiB |
| 262,144 | **11.65 GiB** |
| 1,048,576 | **46.58 GiB** |

Measured per-rank pools on MI308X are **35.71 GiB at EP8** (1P/1D, `--gpu-memory-utilization
0.80`) and **64.19 GiB at EP16** (2P/2D, `0.72`). That makes the 256K row 2.5–3.1 requests per
rank at EP8 versus 4.9–5.5 at EP16 — **EP16 is necessary for that row, not merely faster** —
and puts the 1M row at 0.0–0.8 and 0.6–1.4, i.e. a single p99 request does not reliably fit in
a rank at either topology. (The ranges are the two defensible ways to charge the chunked-prefill
workspace; they are not error bars.) That is why the 1M row runs at half the concurrency of the
256K row: the KV pool sets the ceiling, not the statistics. The larger sampling error on that
row is a consequence of that hardware limit, which is exactly why it gets reported rather than
hidden.
