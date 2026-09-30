# Benchmarks and results

A multinode job starts its servers, then runs one benchmark against them from rank 0
and writes `perf.csv`. This page explains each benchmark, every knob, the files they
write, and how to read a result. For how to launch the job, see
[multinode-running.md](multinode-running.md).

## Contents

- [Choosing a benchmark](#choosing-a-benchmark)
- [Throughput sweep](#throughput-sweep)
- [Long context](#long-context)
- [NIAH: long-context retrieval](#niah-long-context-retrieval)
- [Agentic replay](#agentic-replay)
- [perf.csv](#perfcsv)
- [The CONCURRENCY log and CSV](#the-concurrency-log-and-csv)
- [How madengine collects the CSV](#how-madengine-collects-the-csv)
- [Reading a result](#reading-a-result)
- [What makes a run pass](#what-makes-a-run-pass)

## Choosing a benchmark

`BENCHMARK_SCRIPT` picks the benchmark. The launcher maps it to a script in
[`scripts/vllm_dissag/`](../scripts/vllm_dissag/) and runs it on rank 0 once the
servers and the router are ready.

| `BENCHMARK_SCRIPT` | Script | Measures | vLLM disagg | vLLM colocated | SGLang disagg |
|---|---|---|---|---|---|
| `sweep` (default) | `benchmark_xPyD.sh` | Throughput per (ISL, OSL, concurrency) cell | yes | yes | yes (its own `benchmark_xPyD.sh`) |
| `long_context` | `benchmark_long_context.sh` | Steady-state throughput and latency with per-shape warmup, concurrency 1 first | yes | yes | no |
| `niah` | `benchmark_niah.sh` | Retrieval accuracy: needles found out of 10 per context size | yes | yes | no |
| `agentic` | `benchmark_agentic.sh` | Replay of agentic coding traces; throughput, latency and prefix-cache hit rate | yes | no | yes |

Any other value stops the launcher with an error. The benchmark talks to the router or
proxy on `127.0.0.1:$BENCHMARK_PORT` (the colocated head's `SERVE_PORT`), so requests
take the same path a client's would.

The Kimi-K3 cards default to `niah`. The agentic cards set `agentic`. Every other card
runs the sweep.

## Throughput sweep

[`benchmark_xPyD.sh`](../scripts/vllm_dissag/benchmark_xPyD.sh) runs
`vllm bench serve` with the `random` dataset at every combination of input/output
length and concurrency.

### What it does

1. Waits 10 seconds.
2. Runs one **global warmup**: `WARMUP_PROMPTS` prompts at concurrency `WARMUP_CON`,
   ISL `WARMUP_ISL`, OSL `WARMUP_OSL`. The warmup is written to the log before the
   `Running the benchserving script for iter: 1` marker, so it never becomes a
   result row.
3. For each iteration (`BENCHMARK_ITR`), for each `ISL/OSL` pair in
   `BENCHMARK_COMBINATIONS`:
   - If `SHAPE_WARMUP=1`, runs a **per-shape warmup** at the real ISL/OSL and low
     concurrency, logged to `..._SHAPEWARMUP.log`.
   - For each concurrency in `BENCHMARK_CON`, runs one **cell**: `2 x concurrency`
     prompts (at least 16), request rate `inf`, `--ignore-eos`, `--max-concurrency`
     set to the concurrency. It prints `[RUNNING] prompts <N> isl <ISL> osl <OSL> con
     <CON> (timeout <T>s)` first, then waits 10 seconds after the cell.
4. Parses the log into `..._CONCURRENCY.csv` and `perf.csv` with `parse_to_csv.py`.

### Per-cell timeout and `[STALL]`

Each cell runs under `timeout`. The limit scales with the tokens per request:

```
timeout = max(STEP_TIMEOUT, STEP_TIMEOUT * (isl + osl) / 2048)
```

With the default `STEP_TIMEOUT=1800`, a 1024/1024 cell gets 1800 s and an 8192/1024
cell gets 8100 s. A cell that hits the limit (exit code 124) prints no result block.
The script writes instead:

```
[STALL] isl=<ISL> osl=<OSL> con=<CON> timed out after <T>s
```

to both `..._CONCURRENCY.log` and `..._STALLS.log`, and moves on to the next cell. The
parser turns that line into a `FAILURE` row, so a sweep that stopped answering cannot
pass by omission.

### Knobs

| Variable | Default | Meaning |
|---|---|---|
| `BENCHMARK_ITR` | `1` | Iterations of the whole sweep. The CSV keeps the maximum throughput per cell across iterations. |
| `BENCHMARK_CON` | `8 16 32 64 128 256 512` | Concurrency levels, space-separated. |
| `BENCHMARK_COMBINATIONS` | `1024/1024 8192/1024 1024/8192` | `ISL/OSL` pairs, space-separated. Most disagg cards set `1024/1024`. |
| `STEP_TIMEOUT` | `1800` | Base per-cell timeout in seconds, scaled as above. |
| `WARMUP_CON` | `1` | Global warmup concurrency. |
| `WARMUP_PROMPTS` | `16` | Global warmup prompts. |
| `WARMUP_ISL` / `WARMUP_OSL` | `32` / `32` | Global warmup lengths. |
| `SHAPE_WARMUP` | `0` | `1` adds a warmup at each real shape before its cells. |
| `SHAPE_WARMUP_CON` | `4` | Per-shape warmup concurrency. |
| `SHAPE_WARMUP_PROMPTS` | `8` | Per-shape warmup prompts. |
| `SHAPE_WARMUP_TIMEOUT` | `2400` | Per-shape warmup timeout in seconds. |
| `BENCHMARK_PORT` | set by the launcher | Router or proxy port. |

Why per-shape warmup is off by default: the global warmup is ISL=OSL=32 at concurrency
1, so it never exercises a shape's prefill path, its Triton or AITER kernel variants, or
the decode cudagraph batch sizes. But every model shares this sweep, and an A/B at
1024/1024 concurrency 8 measured it neutral, so recipes validated without it are not
shifted for no gain. GLM-5.1 opts in through its `models.yaml` `env:` block, because
its published latency numbers were measured that way.

Example: a short sweep of two concurrencies at one shape:

```bash
export BENCHMARK_CON="8 64" BENCHMARK_COMBINATIONS="1024/1024" BENCHMARK_ITR=1
```

## Long context

[`benchmark_long_context.sh`](../scripts/vllm_dissag/benchmark_long_context.sh) is a
steady-state serving benchmark. It differs from the sweep in three ways:

- **Per-shape warmup on every cell.** Each (ISL/OSL, concurrency) cell sends
  `--num-warmups` requests of the same shape first and discards them, so the measured
  result is not contaminated by first-hit JIT, cudagraph, kernel-autotune or
  connector-handshake costs.
- **Concurrency 1 first.** Concurrency 1 is the primary latency metric, so the default
  list starts with it, and it is measured cleanly warmed. The order follows
  `BENCHMARK_CON` as given.
- **Metrics:** total throughput per GPU, TTFT, ITL and TPOT.

| Variable | Default | Meaning |
|---|---|---|
| `BENCHMARK_CON` | `1 4 8` | Concurrency list. |
| `BENCHMARK_COMBINATIONS` | `1024/1024` | `ISL/OSL` list. |
| `WARMUPS` | `2` | `--num-warmups` per cell. |
| `NUM_PROMPTS_FACTOR` | `4` | Measured prompts per cell = factor x concurrency (at least 16). The sweep uses 2 x concurrency. |
| `STEP_TIMEOUT` | `2400` | Base per-cell timeout, scaled by tokens as in the sweep. Timeouts print `[STALL]`. |
| `GPUS_TOTAL` | `max(xP, yD) x GPUS_PER_NODE` | GPU count used in the log header. |

It writes `benchmark_long_context_<job>_<time>_xP<n>_yD<n>_<model>_CONCURRENCY.log`
(and `_STALLS.log`), then calls `parse_to_csv.py` exactly as the sweep does, so it writes
`/run_logs/<job>/perf.csv` with one row per cell. The parser reads this harness's
`[RUNNING] isl=... osl=... con=...` header form as well as the sweep's. `benchmark_parser.py`
also understands these lines (see [below](#the-concurrency-log-and-csv)).

## NIAH: long-context retrieval

NIAH (needle in a haystack) checks that the model still reads its whole context. It is
adapted from vllm-project/vllm#47042. [`benchmark_niah.py`](../scripts/vllm_dissag/benchmark_niah.py)
builds a haystack of filler words, plants ten animal names (the needles) at even
intervals, and asks the model to list the animals. The score is how many of the ten
come back.

### How a request is built and scored

- The haystack size is in **words** (`NIAH_WORDS`). Needle placement is seeded, so a
  run is deterministic: bit-exact on the same stack. Each size is run once per seed in
  `NIAH_SEEDS`.
- Requests go to `/v1/chat/completions` with `temperature 0` and
  `chat_template_kwargs.enable_thinking = false`, so a thinking model puts the answer
  in `content` rather than only in a reasoning field.
- Scoring reads `content` and any `reasoning_content` or `reasoning` field, so a
  thinking model is not mis-scored.
- With `NIAH_WARMUP=1` (the default), one throwaway request per size runs before
  scoring, with a timeout of at least 1800 s. On a freshly booted node the first
  request of a shape can take minutes to compile; without the warmup that lands on the
  first scored request and gives a false 0/10 or a timeout. Warmup failures are logged,
  never fatal.

### The served name must match

The model name in each request must equal the name the server registers, or every
request returns 404 and the whole run records nothing useful. The launchers handle it:

- **vLLM disagg:** `vllm_disagg.sh` resolves `SERVED_MODEL_NAME` from the recipe's
  `--served-model-name` flag when its flags set one (for example `kimi-k3` for
  `Kimi-K3-MXFP4`), else from `MODEL_PATH`, which is vLLM's default. It reads the
  recipe, not the router's `/v1/models`, because that endpoint returns 503 under MoRIIO
  service discovery.
- **vLLM colocated:** the server gets `--served-model-name "$MODEL_NAME"`, and the
  launcher exports `NIAH_MODEL` to match.
- `benchmark_niah.sh` then uses `NIAH_MODEL` if set, else `SERVED_MODEL_NAME`, else
  `MODEL_PATH`.

Before scoring, `benchmark_niah.sh` polls the router's `/v1/models` every 5 seconds
for up to 300 seconds, and proceeds with a warning if it never answers (the warmup
absorbs a late router).

### Knobs

| Variable | Default via the launcher | Meaning |
|---|---|---|
| `NIAH_WORDS` | `2000,8000,20000,35000` | Context sizes in words, comma-separated. The Kimi-K3 cards use `10000,50000,100000,200000`. |
| `NIAH_SEEDS` | `0,1,2` | Needle-layout seeds. The summary reports mean, min and max across seeds, to separate real accuracy from single-needle variance. |
| `NIAH_MAXTOK` | `2048` | `max_tokens` for the answer. Raise it for reasoning models. Standalone `benchmark_niah.py` defaults to `8192`. |
| `NIAH_TIMEOUT` | `1800` | Per-request timeout in seconds. |
| `NIAH_WARMUP` | `1` | `0` disables the warmup pass. |
| `NIAH_MODEL` | see above | Model name to request. |
| `NIAH_URL` | `http://127.0.0.1:$BENCHMARK_PORT/v1/chat/completions` | Endpoint. |

### Output

Per request:

```
words= 10000  seed=0  found=10/10  finish=stop  ['crocodile', 'dolphin', ...]
```

Per size, in the summary:

```
  words= 10000  mean=10.0/10  min=10  max=10  (n=3)
```

- A request that times out or errors prints `TIMEOUT/ERROR` and is excluded from the
  mean; the summary then adds `[k timeout/err excluded]`.
- A size where every request timed out or errored prints `NO-RESULT` instead of a mean.
- `finish=length` means the answer was cut off by `max_tokens`. The summary adds
  `(TRUNCATED in k/n: raise NIAH_MAXTOK)`. A truncated answer scores low for lack of
  room, not lack of retrieval: on Kimi-K3 at 2048 tokens, two of four sizes scored 1/10,
  both listing only the first animal.

The log is `niah_<job>_<time>_xP<n>_yD<n>_<model>.log`. `parse_to_csv.py --niah` then
writes one `perf.csv` row per size that has a summary line; see [perf.csv](#perfcsv).

### Running NIAH by hand

`benchmark_niah.py` runs against any live OpenAI-compatible endpoint:

```bash
NIAH_URL=http://<host>:<port>/v1/chat/completions \
NIAH_MODEL=<served model name> \
NIAH_WORDS=2000,8000 \
  python3 scripts/vllm_dissag/benchmark_niah.py
```

It exits with status 2 if `NIAH_MODEL` is empty.

### Words are not tokens

The haystack is sized in words, and for this filler a word is roughly 1.3 tokens. CSV
rows from a `niah` run are labelled in words. Do not compare them directly with results
quoted in tokens.

### Example: Kimi-K3 disaggregated 2P/2D

Single needle `HELIOTROPE-7492`, greedy (temperature 0), depths 0.1, 0.5 and 0.9. All
pass, deterministically, across the full native context range. Context sizes here are
in **tokens**, and the times are warm.

| Context | Result | Eval time per request |
|---|---|---|
| 10K to 200K | 3/3 PASS | 5 to 88 s |
| 300K | 3/3 PASS | about 150 s |
| 500K | 3/3 PASS | about 301 s |
| 750K | 3/3 PASS | about 542 s |
| 900K | 3/3 PASS | about 717 s |

Scaling is sub-quadratic. The top of the range needs `--max-model-len 1000000` and
`KV_CACHE_MEMORY_BYTES=40000000000`, both defaults in the `Kimi-K3` recipe. A fresh
server pays one AITER MLA kernel JIT compile (about 15 minutes) on the first request of
200K tokens or more; it is cached afterwards. The stricter 10-needle stress dips to
about 9/10 at 20K and above, an RDMA write-visibility race; single-needle retrieval is
unaffected. See [kimi-k3.md](kimi-k3.md) for the fixes behind these numbers.

## Agentic replay

`BENCHMARK_SCRIPT=agentic` replays real Claude Code agentic coding traces (the
`inferencex-agentx-mvp` scenario from
[SemiAnalysisAI aiperf](https://github.com/SemiAnalysisAI/aiperf)) against the router.
Each turn re-sends the accumulated context, so the workload is highly
prefix-repetitive: it exercises the prefix cache and reports a **measured**
`gpu_cache_hit_rate` next to the dataset's theoretical ceiling.

When it is selected, the connectors turn on server-side Prometheus metrics and remove
`--no-enable-prefix-caching` (vLLM) or `--disable-radix-cache` (SGLang), so the
prefix cache is on. For the vLLM-specific setup (the `/v1/models` shim, ports, the
validated cards and examples), see [vllm-disagg.md](vllm-disagg.md#agentic-replay-benchmark).
The harness itself is documented in
[scripts/common/agentx/README.md](../scripts/common/agentx/README.md).

### Outputs

Written to `/run_logs/<job>/` (`$LOG_PATH/<job>/` on the host):

```
agentic_<jobid>_xP<n>_yD<n>_<model>.json   # single-replay aggregate
suite_summary.json                         # multi-workload suite roll-up (workloads[])
<workload>/[conc<N>/]                      # per-workload (and per-concurrency) result dirs
aiperf_artifacts/                          # per-request logs, timeslices, server_metrics_export.{csv,json}
agentic_models_shim_NODE0.log              # vLLM only: the /v1/models shim log
```

Key fields in the aggregate JSON:

| Field | Meaning |
|---|---|
| `request_metrics.cache.theoretical_cache_hit_rate` | The dataset-derived ceiling. |
| `server_metrics.present`, `server_metrics.cache.gpu_cache_hit_rate` | The measured runtime GPU cache hit rate. |
| `request_accounting` | `records_profiled` versus `records_error_dropped`. The error rate is over measured requests, excluding warmup. |

`suite_summary.json` has a `workloads[]` array. Each entry carries `workload`,
`result_dir`, `status`, `theoretical_cache_hit_rate`, `gpu_cache_hit_rate` and
`error_rate`. A workload that exits non-zero is marked `INVALID`, with a
`RUN_INVALID.json` sidecar in its directory, and the suite continues to the next one.

### Validation

`scripts/common/validate_agentic_result.sh` runs at the end of the agentic benchmark
and prints, for example:

```
[validate] requests: 70/72 ok  error_rate=2.8%
[validate] gpu_cache_hit_rate: 32.4%
[validate] throughput_per_gpu: 6247 tok/s
[validate][PASS] error_rate within 10%
```

| Threshold | Default | Effect |
|---|---|---|
| `AGENTIC_MAX_ERROR_RATE` | `0.10` | Gates PASS or WARN. |
| `AGENTIC_MIN_CACHE_HIT` | `0.30` | Warn only. |

A low cache hit rate with `xP > 1` usually means the router is not keeping a
conversation's turns on the worker that holds its prefix (session affinity).

`DURATION` defaults to 120 seconds; use 900 or more for a canonical run.

## perf.csv

`perf.csv` is the file madengine reads. Every benchmark writes it to
`/run_logs/<job>/perf.csv`, and the launcher then copies it to
`perf_<MODEL_NAME>.csv` in its working directory for madengine to collect. If there is
no `perf.csv`, the launcher exits 1.

[`parse_to_csv.py`](../scripts/vllm_dissag/parse_to_csv.py) writes it, in one of two
schemas.

### Full schema (the default)

29 columns, the legacy madengine `perf.csv` layout:

```
model, n_gpus, nnodes, gpus_per_node, training_precision, pipeline, args, tags,
docker_file, base_docker, docker_sha, docker_image, git_commit, machine_name,
deployment_type, launcher, gpu_architecture, performance, metric, relative_change,
status, build_duration, test_duration, dataname, data_provider_type, data_size,
data_download_duration, build_number, additional_docker_run_options
```

`parse_to_csv.py` fills `model`, `performance`, `metric`, `status` and the run metadata
it can read from the environment:

| Column | Value |
|---|---|
| `pipeline` | `vllm` |
| `deployment_type` | `PERF_DEPLOYMENT_TYPE`, else `disagg_<xP>P<yD>D`. The colocated launcher sets `colocated_pp<PP>xtp<TP>`. |
| `tags` | `PERF_TAGS`, else `vllm_disagg,<backend>`, where backend is `mori` (`RUN_MORI=1`), `deepep` (`RUN_DEEPEP=1`) or `nixl`. The colocated launcher sets `vllm_multinode,colocated,<ep_...|noep>`. |
| `nnodes` | `NNODES`, else `xP + yD` |
| `gpus_per_node` | `GPUS_PER_NODE` (default 8) |
| `n_gpus` | `nnodes x gpus_per_node` |
| `docker_image` | `DOCKER_IMAGE_NAME` |
| `machine_name` | `SLURM_JOB_NODELIST` |
| `launcher` | `slurm_multi` |
| `gpu_architecture` | `PERF_GPU_ARCH` (the detected GPU), else `gfx942` |

Cards that declare no `multiple_results` rely on this schema: madengine reads their
CSV directly, with no metadata to merge.

### Narrow schema (`--narrow`)

The preferred contract: the benchmark reports only what it measured, and madengine
merges in the run metadata it already owns (nodes, GPUs, GPUs per node, launcher,
docker image, base image, image SHA, tags, pipeline, build number, machine name) through
the card's `multiple_results` declaration. This is the contract the templated launchers
use, so rows from different launchers describe themselves the same way. It exists
because a workload cannot reliably know its own topology: the full schema once reported
a colocated 2-node, 16-GPU run as a 1-node, 8-GPU `disagg_1P0D` on a `nixl` backend it
never used, since the colocated launcher sets `xP=1 yD=0` only to keep log file names
unique.

A narrow sweep CSV has these columns:

| Column | Value |
|---|---|
| `model` | `MODEL_NAME` |
| `benchmark` | `throughput_sweep` |
| `inp`, `out`, `max_concurrency` | The cell's ISL, OSL and concurrency |
| `performance` | Total token throughput |
| `metric` | `tok/s` |
| `tp` | `EP_TP_SIZE`, only when `WIDE_EP=1` |
| `pp` | `PP_SIZE`, when set |
| `ep_backend` | `ALL2ALL_BACKEND`, else `VLLM_ALL2ALL_BACKEND`, else `enabled`; only when `ENABLE_EP=1` or `WIDE_EP=1` |
| `prefill_decode` | `<xP>P<yD>D`, when `yD` is not 0 |

Empty configuration columns are left out. The narrow sweep CSV has no `status` column.
`BENCHMARK_SCRIPT=sweep` still uses the full schema by default; `benchmark_xPyD.sh`
does not pass `--narrow`. To migrate a card: declare `multiple_results` on it and pass
`--narrow`.

### Throughput rows

One row per sweep cell, that is per (ISL, OSL, concurrency):

- `performance`: the maximum `Total token throughput (tok/s)` for that cell across all
  iterations.
- `metric`: `tok/s (isl=<ISL> osl=<OSL> con=<CON>)`.
- `status`: `FAILURE` if the cell **stalled**, **printed no result** (for example its
  benchmark crashed, or its warmup got an error from a dead server), **lost any request**
  (`Failed requests` greater than 0), or **measured zero throughput**; else `SUCCESS`.
  A cell with no result has performance 0.

The log is read cell by cell: each `[RUNNING]` line owns the text up to the next one, and a
cell's result is the result block inside its own text. So a cell that printed nothing still
gets a row, and no result is ever filed under another cell's concurrency. The global warmup
before `iter: 1` is not a row.

The SGLang parser (`scripts/sglang_disagg/parse_to_csv.py`) applies the same rules.
`sglang.bench_serving` prints no failed count, so an SGLang cell has lost requests when its
`Successful requests` is below the prompt count on its `RUNNING` line.
`scripts/sglang_disagg/tests/parse_to_csv_assert.sh` checks it offline. `scripts/vllm_dissag/tests/parse_to_csv_assert.sh` checks all
of this offline:

```bash
bash scripts/vllm_dissag/tests/parse_to_csv_assert.sh
```

Its fixture produces, for example:

```
SUCCESS 100.00 tok/s (isl=1024 osl=1024 con=8)
SUCCESS 200.00 tok/s (isl=1024 osl=1024 con=16)
FAILURE 0.00 tok/s (isl=1024 osl=1024 con=32)     # stalled
FAILURE 300.00 tok/s (isl=1024 osl=1024 con=64)   # 10 failed requests
FAILURE 0.00 tok/s (isl=1024 osl=1024 con=128)    # zero throughput
FAILURE 0.00 tok/s (isl=1024 osl=1024 con=256)    # crashed, printed no result
```

### NIAH rows

One row per context size, in the full schema:

- `performance`: mean needles found out of 10 across the seeds that returned an answer,
  one decimal; 0.0 for a size with no answer at all.
- `metric`: `retrieval/10 (niah words=<N> seeds=<n>)`, where `n` is the number of
  seeds that returned an answer (for a `NO-RESULT` size, the number attempted).
- `status`: `FAILURE` if any request at that size timed out or errored: the summary line
  says `[k timeout/err excluded]`, or `NO-RESULT` when all of them did. Else `SUCCESS`.

A low score, or a `TRUNCATED` note, is a measurement rather than a harness failure, so it
stays `SUCCESS`: the score itself is what you judge. madengine counts `FAILURE` rows as
failed runs and exits non-zero, so a run with an errored size fails.

### Agentic rows

`benchmark_agentic.sh` runs the shared replay, then passes each
`agentic_<job>_*.json` to `parse_to_csv.py --agentic-json`. Each aggregate becomes up
to six rows, in the full schema:

| `metric` | `performance` |
|---|---|
| `tok/s total (agentic <workload>, <ok>/<total> requests)` | Total token throughput |
| `tok/s output (...)` | Output token throughput |
| `ms TTFT p50 (...)` | Median time to first token |
| `ms TPOT p50 (...)` | Median time per output token |
| `ms E2E latency p50 (...)` | Median end-to-end latency |
| `% GPU prefix-cache hit (...)` | Measured GPU prefix-cache hit rate, when present |

`status` follows the replay's own validator: `FAILURE` when no request succeeded, when
the error rate is above `AGENTIC_MAX_ERROR_RATE` (default 0.10), or when the run
marked itself invalid (`RUN_INVALID.json`); else `SUCCESS`.

## The CONCURRENCY log and CSV

The sweep's raw output is
`benchmark_<job>_<time>_xP<n>_yD<n>_<model>_CONCURRENCY.log`: the start time, the
warmup, then each cell's `[RUNNING]` line and `vllm bench serve` result block (or its
`[STALL]` line).

`parse_to_csv.py` also writes `..._CONCURRENCY.csv` next to it, one row per cell,
sorted by concurrency, then ISL, then OSL:

| Column | Meaning |
|---|---|
| `Concurrency` | Maximum concurrency of the cell |
| `Input tokens` | ISL |
| `Output tokens` | OSL |
| `Total Token throughput (tok/s)` | Maximum across iterations |

For the full per-cell picture, parse the log with
[`benchmark_parser.py`](../scripts/vllm_dissag/benchmark_parser.py):

```bash
python3 benchmark_parser.py <log_path>/benchmark_XXX_CONCURRENCY.log
python3 benchmark_parser.py <log_path>/benchmark_XXX_CONCURRENCY.log --compact
python3 benchmark_parser.py <log_path>/benchmark_XXX_CONCURRENCY.log --csv results.csv --no-screen
```

| Option | Effect |
|---|---|
| `--csv [FILE]` | Also save a CSV (default name `benchmark_results.csv`). |
| `--compact` | Fewer columns. |
| `--no-screen` | Only write the CSV; requires `--csv`. |

It reads both `[RUNNING]` formats (the sweep's `isl N osl N con N` and the long-context
`isl=N osl=N con=N`) and takes the model and `xP`/`yD` from the file name. It emits a
row only for a cell that produced a result block. Columns: `Model`, `xP_yD`, `ISL`,
`OSL`, `Concurrency`, `Prompts`, `Successful`, `Failed`, `Total_Input_Tokens`,
`Total_Output_Tokens`, `Request_Throughput_req_s`, `Output_Token_Throughput_tok_s`,
`Total_Token_Throughput_tok_s`, `Mean_TTFT_ms`, `Median_TTFT_ms`, `Mean_ITL_ms`,
`Median_ITL_ms`, `Mean_TPOT_ms`, `Median_TPOT_ms`. `--compact` keeps `Model`, `xP_yD`,
`ISL`, `OSL`, `Concurrency`, `Successful`, `Failed`, `Output_Token_Throughput_tok_s`,
`Median_TTFT_ms` and `Median_ITL_ms`.

## How madengine collects the CSV

A card may declare `multiple_results`, the file name its results CSV will have. The
Kimi-K3 cards declare `perf_<MODEL_NAME>.csv` (for example `perf_Kimi-K3.csv`), the
name the launcher copies `perf.csv` to.

madengine's `slurm_multi` collector looks for the CSV in this order:

1. The card's declared `multiple_results` file under the working directory and the
   output directory. When several match, it picks the one with the most non-empty
   `performance` values, so a header-only CSV from a failed run loses to a real one.
2. `perf*.csv` under `slurm.results_dir`.
3. `/shared_inference/$USER/<job>/perf.csv`, then
   `/shared_inference/$USER/model_blog_logs/<job>/perf.csv`.
4. `slurm_output/perf_csv/*<job>*.csv`, then `./perf.csv`.

Older madengine versions skipped step 1. That is why the site settings in
[multinode-running.md](multinode-running.md#site-settings-a-worked-example) set
`slurm.results_dir` to the launcher's own directory (`scripts/vllm_dissag` or
`scripts/vllm_multinode`): it makes the declared file findable. `results_dir` can be
set in `--additional-context` but not from a model card.

madengine's final `perf.csv` (the `-o` of `madengine run`) has one row per result
row the workload reported, and a `FAILURE` row with the reason when the run failed.

## Reading a result

A worked reading of a sweep:

1. **Did the job finish?** The launcher exits 1 when there is no `perf.csv`, and the
   job log says `ERROR: no perf CSV at ...`. Go to the node logs
   ([multinode-running.md](multinode-running.md#logs)) and read the first error lines
   the launcher printed.
2. **Is every row `SUCCESS`?** A `FAILURE` row names its cell in `metric`. Then:
   - performance 0 and a `[STALL]` line in `..._STALLS.log`: the cell timed out. Check
     the server logs for a hang or a fault at that time.
   - performance above 0: requests were lost. `benchmark_parser.py` shows the
     `Successful` and `Failed` counts for the cell.
   - performance 0 with no stall: the cell produced no throughput.
3. **Compare like with like.** A row's `metric` carries its ISL, OSL and concurrency.
   Compare only rows with the same shape, on the same topology (`deployment_type`,
   `nnodes`) and GPU (`gpu_architecture`).
4. **Look past total throughput.** `benchmark_parser.py` gives TTFT, ITL and TPOT per
   cell. Disaggregation trades single-stream latency for concurrent throughput, so read
   concurrency 1 or 8 for latency and the high concurrencies for throughput.
5. **Check the transport.** A number that is plausible but low can mean NCCL fell back
   to TCP. The job log's `RDMA rails:` line and any `not ACTIVE` warning tell you.

For NIAH, a size with `timeout/err excluded` or `NO-RESULT` is already a `FAILURE` row; for the
rest, each size should reach the score you expect, with no `TRUNCATED`. For agentic, read the `[validate]` lines.

## What makes a run pass

A run counts as passing when all of these hold:

- The SLURM job exits 0. The launchers exit non-zero when a server fails to start,
  when any node aborts the job, or when no `perf.csv` was produced.
- `perf.csv` has the rows you expect: one per sweep cell, one per NIAH size, or the
  agentic metrics per workload.
- Every row's `status` is `SUCCESS`. For the sweep that means no cell stalled, printed no
  result, lost a request, or measured zero throughput. For NIAH it means no request at any
  size timed out or errored. For agentic it means the error rate is within
  `AGENTIC_MAX_ERROR_RATE`.
- The numbers are in the range the recipe was validated at, on a shape and GPU its
  `models.yaml` entry names. NIAH scores deserve the same scrutiny: a `SUCCESS` NIAH row says
  every request answered, not that the answers were right, so the score itself is the pass
  criterion.
