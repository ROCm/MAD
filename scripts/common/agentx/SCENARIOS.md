# AgentX scenarios cookbook

Copy-paste `agentic.yaml` recipes for common AgentX cases. Each scenario shows
the config snippet (reusing [agentic.example.yaml](agentic.example.yaml)
entries), what happens step by step, and the resulting output tree.

- Concepts and schema: [README.md](README.md).
- Profile/preset authoring: [profiles/README.md](profiles/README.md).

**Where things land (read this once):**

- **Corpus** (the `session_XXXXX.json` files) lives under
  `SUITE_CORPUS_DIR/<name>` — default `/tmp/agentx_corpora/<name>` — a reusable
  cache. It is **not** written under `RESULT_DIR`.
- **`RESULT_DIR/<name>/`** holds only replay output: `aiperf_artifacts/`,
  `benchmark_command.txt`, `benchmark.log`, the aggregate `*.json` (name from
  `AGENTIC_RESULT_FILENAME`), and `RUN_INVALID.json` on failure. A concurrency
  **list** adds `conc<N>/` subdirs; a single concurrency writes flat.
- The suite also writes `RESULT_DIR/suite_summary.json`.

Run any of these with:

```bash
AGENTIC_CONFIG=agentic.yaml bash scripts/common/benchmark_agentic_suite.sh
```

---

## Scenario 1: single generated profile (`conformance_256k`)

```yaml
serving:
  model: auto
  max_model_len: 0            # auto-detect the served window
run:
  concurrency: 16
  duration: 900
workloads:
  - name: conformance_256k
    source: profile
    preset: conformance_256k
```

**What happens:**

1. `agentx_config.py` resolves the entry, inheriting
   `profiles/conformance_256k.yaml` via `_merge_preset()`.
2. `materialize_corpus()` generates the corpus into
   `/tmp/agentx_corpora/conformance_256k/` (cached), then runs
   `verify_agentx_profile.py` as a pre-gate — the run aborts unless the verifier
   prints `13/13 axes within band`.
3. `context_compat_check()` rounds the ISL tail (clamp hi `245000`) up to
   `262144` for `--max-context-length`.
4. aiperf replays the corpus at concurrency `16`; results are written flat
   (single concurrency).

**Output tree:**

```
/tmp/agentx_corpora/conformance_256k/     # CORPUS (cache), not under RESULT_DIR
  session_00000.json ... session_00199.json

RESULT_DIR/
  conformance_256k/
    aiperf_artifacts/
    benchmark_command.txt
    benchmark.log
    <result>.json                         # name from AGENTIC_RESULT_FILENAME
  suite_summary.json
```

---

## Scenario 2: multiple profiles in one run

```yaml
workloads:
  - name: conformance_256k
    source: profile
    preset: conformance_256k
  - name: conformance_512k
    source: profile
    preset: conformance_512k
```

**What happens:** the two workloads run **in sequence** against the same
endpoint. Each is generated + verified + replayed independently; both roll up
into one `suite_summary.json`.

**Output tree:**

```
RESULT_DIR/
  conformance_256k/
    aiperf_artifacts/  benchmark_command.txt  benchmark.log  <result>.json
  conformance_512k/
    aiperf_artifacts/  benchmark_command.txt  benchmark.log  <result>.json
  suite_summary.json
```

---

## Scenario 3: real HF trace (`source: hf` + `loader:`)

```yaml
workloads:
  - name: inferencex
    source: hf
    loader: semianalysis_cc_traces_weka_062126_256k
```

**What happens:**

1. No corpus is generated. `resolve_trace_loader()` maps the loader to
   `--public-dataset semianalysis_cc_traces_weka_062126_256k`; the dataset is
   downloaded at run time (3× retry/backoff) into the shared HF cache.
2. The `_256k` loader suffix sets the context-gating ISL tail to `262144`.
3. With **no** Tier 2 `filter:`, the replay uses the byte-identical
   `--public-dataset` path (no local `filter_weka_corpus.py` step).

**Output tree:** same layout as Scenario 1 under `RESULT_DIR/inferencex/`. There
is no `SUITE_CORPUS_DIR/inferencex/` dir because the unfiltered hf path streams
from the HF cache rather than a materialized `weka_trace` dir.

---

## Scenario 4: reusable HF preset with a bundled sweep (`inferencex_256k`)

```yaml
workloads:
  - name: inferencex_preset
    preset: inferencex_256k
```

**What happens:** [profiles/inferencex_256k.yaml](profiles/inferencex_256k.yaml)
bundles `source: hf`, the loader, `concurrency: [2, 4, 8]`, `duration: 900`, and
Tier 1 knobs (`num_dataset_entries: 393`, `trajectory: {min: 0.25, max: 0.75}`).
Because `concurrency` is a **list**, the workload is swept and results land in
per-concurrency subdirs. The sweep runs the concurrency values **sequentially**
(not in parallel), each into its own `conc<N>/` subdir, so total wall-time is
roughly `N x (cache-warmup + duration)`.

**Output tree:**

```
RESULT_DIR/
  inferencex_preset/
    conc2/   aiperf_artifacts/  benchmark_command.txt  benchmark.log  <result>.json
    conc4/   aiperf_artifacts/  benchmark_command.txt  benchmark.log  <result>.json
    conc8/   aiperf_artifacts/  benchmark_command.txt  benchmark.log  <result>.json
  suite_summary.json
```

---

## Scenario 5: Tier 1 knobs (`num_dataset_entries`, `trajectory`)

```yaml
workloads:
  - name: inferencex_light
    preset: inferencex_256k
    num_dataset_entries: 50            # pull fewer trace sessions
    trajectory: { min: 0.30, max: 0.80 }  # start-window ratio for captured traces
```

**What happens:** entry keys win over the preset (`_merge_preset()`). Tier 1
knobs steer the replay only: `--num-dataset-entries 50` (hf downloads only) and
`--trajectory-start-min-ratio 0.30 --trajectory-start-max-ratio 0.80`. No corpus
is filtered on disk. `trajectory` is validated as `0.0 <= min <= max <= 1.0`.

**Output tree:** as Scenario 4 (a sweep, since the preset's `[2, 4, 8]` is
inherited) under `RESULT_DIR/inferencex_light/conc<N>/`.

---

## Scenario 6: Tier 2 filter (`max_isl`, `max_turns`, `sample`)

```yaml
workloads:
  - name: inferencex_small
    preset: inferencex_256k
    concurrency: [2]              # entry overrides the preset's [2, 4, 8] sweep
    num_dataset_entries: 50       # Tier 1: pull fewer sessions
    trajectory: { min: 0.30, max: 0.80 }
    filter:                       # Tier 2: local subset/trim (download once, then filter)
      max_isl: 200000            # drop sessions with any turn over 200k input tokens
      max_turns: 40              # truncate each session to its first 40 turns
      sample: 100                # randomly keep 100 sessions (seed=42)
```

**What happens:** a `filter:` on an hf workload triggers `materialize_hf_corpus()`:
download once, then `filter_weka_corpus.py` applies **max_turns, then max_isl,
then sample** (in that order) and writes a materialized `weka_trace` dir. The
replay then uses `--custom-dataset-type weka_trace --input-file <dir>`. The
filtered corpus is cached under a content-addressed key
`hf_<loader>_<sha1(filter)[:8]>`. An empty filter result fails loudly (exit 1).

**Output tree:**

```
/tmp/agentx_corpora/hf_semianalysis_cc_traces_weka_062126_256k_<hash>/   # filtered corpus (cache)
  session_00000.json ...

RESULT_DIR/
  inferencex_small/
    aiperf_artifacts/  benchmark_command.txt  benchmark.log  <result>.json   # single conc => flat
  suite_summary.json
```

---

## Scenario 7: custom inline workload (`my_case`)

Define a profile inline (no preset) — copy the fields from
[profiles/custom.example.yaml](profiles/custom.example.yaml):

```yaml
workloads:
  - name: my_case
    source: profile
    model_tag: GLM-5.2-MXFP4
    id_prefix: my_case
    seed: 42
    n_sessions: 150
    block_size: 64
    isl_p:   [48000, 120000, 200000]
    osl_p:   [256, 2000, 9000]
    delay_p: [3, 20, 180]
    turns:
      values:  [2, 3, 4, 6, 10, 20, 45, 103]
      weights: [20, 24, 20, 12, 8, 6, 7, 3]
    cache_hit: [0.88, 0.90]
    clamps:
      isl:   [1200, 205000]
      osl:   [8, 20000]
      delay: [1, 600]
```

**What happens:** identical to Scenario 1, but the distribution targets come from
the entry itself instead of a shipped preset. With no `verify:` block, the
verifier derives the turns targets from the `turns` distribution and the cache
target from `mean(cache_hit) * 100`, so the profile still round-trips to
`13/13 axes within band`. Output tree matches Scenario 1 under
`RESULT_DIR/my_case/`.

---

## Scenario 8: context-window behavior (auto vs pin)

Case-B's ISL tail (`clamps.isl` hi `520000`) rounds up to a `524288` window.

**Auto-detect (recommended):**

```yaml
serving:
  max_model_len: 0            # auto-detect from /v1/models
workloads:
  - { name: conformance_512k, preset: conformance_512k }
```

- If the served window is `>= 524288`, `context_compat_check()` sets
  `--max-context-length 524288`, verdict `OK`.
- If the served window is smaller (say `262144 < 524288`), it **WARNs** and caps
  `--max-context-length` at the served window (late turns get truncated).
- With `AGENTIC_STRICT_CONTEXT=1`, that same case is **SKIPPED** instead
  (recorded as `SKIP(context)` in `suite_summary.json`, no replay).

**Pin explicitly** (e.g. when `/v1/models` doesn't expose the window, or to force
a cap):

```yaml
serving:
  max_model_len: 524288       # Case-B needs 524288
```

A pinned value always wins over auto-detect (and over the `0` default).

---

## Scenario 9: `DRY_RUN=1` preview

```bash
DRY_RUN=1 AGENTIC_CONFIG=agentic.yaml bash scripts/common/benchmark_agentic_suite.sh
```

For the Scenario 1 config, the driver prints the resolved plan and each assembled
command **without contacting a server** (no download, no generate, no replay):

```
[agentic][DRY_RUN] resolved suite plan
  config                 : agentic.yaml
  serving.model          : auto
  serving.max_model_len  : 0
  serving.port           : auto (AGENTIC_PORT=2322)
  serving.server_metrics : auto
  run.concurrency        : 16
  run.duration           : 900
  workloads (conformance_256k)
  RESULT_DIR             : /run_logs/0
  SUITE_CORPUS_DIR       : /tmp/agentx_corpora

[agentic][DRY_RUN] workload='conformance_256k' source='profile' conc=16 duration=900
  context verdict        : OK (--max-context-length 262144)
  trace source           : --custom-dataset-type weka_trace --input-file /tmp/agentx_corpora/conformance_256k
  result dir             : /run_logs/0/conformance_256k
  command:
aiperf profile --scenario inferencex-agentx-mvp --url http://localhost:2322 --endpoint /v1/chat/completions --endpoint-type chat --streaming --model auto --concurrency 16 --benchmark-duration 900 --random-seed 42 --failed-request-threshold 0.10 --trajectory-start-min-ratio 0.90 --trajectory-start-max-ratio 0.98 --agentic-cache-warmup-duration 60 --warmup-grace-period 1800 --use-server-token-count --tokenizer-trust-remote-code --no-gpu-telemetry --slice-duration 1.0 --max-context-length 262144 --output-artifact-dir /run_logs/0/conformance_256k/aiperf_artifacts --custom-dataset-type weka_trace --input-file /tmp/agentx_corpora/conformance_256k
```

Notes: `source: profile` replays near-complete sessions
(`--trajectory-start-*-ratio 0.90/0.98`); hf workloads default to `0.25/0.75`.
The leading `aiperf` is the isolated venv's aiperf CLI path at run time.

---

## Scenario 10: anti-pattern — circular preset

```yaml
# profiles/a.yaml   ->  preset: b
# profiles/b.yaml   ->  preset: a
workloads:
  - { name: loop, preset: a }
```

**What happens:** `_merge_preset()` tracks visited presets and raises immediately:

```
ValueError: circular preset: a
```

The config load fails (`config load failed`) before any corpus work. Break the
cycle so each preset chain terminates at a base profile.
