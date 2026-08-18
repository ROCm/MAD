# AgentX benchmarking core

AgentX runs a **list of agentic trace-replay workloads** against **one** served
endpoint. For each workload the suite either **generates** a synthetic
`weka_trace` corpus from a distribution profile or **downloads** a real captured
HuggingFace (HF) trace, **verifies** the corpus as a hard pre-gate, **replays**
it with aiperf, and writes a per-workload result dir plus a combined suite
summary. The whole run is described by one `agentic.yaml` (a `serving:` block, a
`run:` block, and a `workloads:` list); adding "N cases" just means adding N list
entries.

```mermaid
flowchart LR
  config["agentic.yaml"] --> resolve["agentx_config.py (resolve_config)"]
  resolve --> src{"workload source?"}
  src -->|profile| gen["gen_agentx_profile.py (generate corpus)"]
  src -->|hf| dl["download (+ filter_weka_corpus.py)"]
  gen --> verify["verify_agentx_profile.py (pre-gate)"]
  verify -->|"13/13 axes within band"| replay["aiperf replay"]
  dl --> replay
  replay --> results["RESULT_DIR/<name>/ results"]
```

This document covers the **agentx core**: the config schema, the profile/preset
model, the generate/verify tools, and the Tier 1 / Tier 2 knobs. The launcher /
disaggregated-serving integration and the deep env/run-path reference are out of
scope here (see [See also](#see-also)).

## What AgentX adds on top of aiperf

aiperf's `inferencex-agentx-mvp` scenario provides the base: an agentic trace
replay engine, the `weka_trace` corpus format, HuggingFace loaders, and the core
metrics (TTFT, E2E latency, throughput, cache-hit). It does **not** provide
multi-workload orchestration, corpus generation, corpus verification, declarative
config, or backend portability. AgentX is the orchestration + generation layer
that closes those gaps by **wrapping** aiperf (not forking it).

| Capability | aiperf (base) | AgentX adds |
| --- | --- | --- |
| Replay engine | Single-workload CLI | Multi-workload orchestration over one config |
| Corpus source | HF captured traces only | HF traces + seed-deterministic generated profiles |
| Corpus verification | None | 13-axis pre-gate, hard abort on drift |
| Config format | CLI flags only | Declarative YAML (serving + workloads[]) |
| Preset reuse | Copy-paste | Inheritance with per-entry override |
| Context sizing | Hardcoded 256k/1M | Auto-detect from /v1/models, any window |
| Corpus filtering | Download full corpus | Tier 2 trim (max_isl/max_turns/sample), cached |
| Backend integration | vLLM-specific paths | OpenAI-API core, thin per-backend hook |
| Multi-workload runs | N CLI calls | One config -> N result dirs + unified summary |
| Preview | None | DRY_RUN=1, no server, sub-second |

The nine additions, one line each:

1. **Suite driver** (`benchmark_agentic_suite.sh`) — loops `workloads[]`, materializes each corpus, runs aiperf, writes one `suite_summary.json`.
2. **Profile generator** (`gen_agentx_profile.py`) — distribution targets -> byte-identical corpus via fixed seed.
3. **13-axis verifier** (`verify_agentx_profile.py`) — measures ISL/OSL/Turns/Delay P50/P90/P99 + Cache-hit P50 and aborts unless all axes pass.
4. **Declarative config** (`agentx_config.py` + `agentic.yaml`) — serving / run / workloads[] blocks; add a workload in one entry.
5. **Tier 1 / Tier 2 separation** — Tier 1 (concurrency, num_dataset_entries, trajectory) steers replay; Tier 2 (max_isl, max_turns, sample) produces a cached subset corpus.
6. **Model-agnostic context gating** — auto-detect served window from `/v1/models`, cap `--max-context-length` at `min(next_pow2(ISL tail), window)`, WARN/cap or SKIP under `AGENTIC_STRICT_CONTEXT=1`.
7. **Preset inheritance** — ship a shape once in `profiles/<name>.yaml`, reference via `preset:`, override per entry; circular chains raise `ValueError`.
8. **Backend-agnostic core** (`agentic_lib.sh`, ~480 lines) — speaks only the OpenAI API; a new backend is a thin (~60-line) hook. Currently integrated: SGLang disaggregated (`scripts/sglang_disagg/benchmark_agentic.sh`, port 2322); vLLM/TRT-LLM/TGI are future hooks.
9. **DRY_RUN preview** — `DRY_RUN=1` prints the resolved plan and every assembled aiperf command with no server/download/replay.

## File map

```
scripts/common/agentx/
  agentx_config.py        # config loader: parses agentic.yaml + profiles, resolves per-workload params, emits JSON/shell
  gen_agentx_profile.py   # seed-deterministic corpus generator (profile JSON -> session_XXXXX.json files)
  verify_agentx_profile.py# corpus verifier: 13-axis conformance table + "N/N axes within band" pre-gate
  filter_weka_corpus.py   # Tier 2 filter: trim a downloaded hf corpus (max_isl / max_turns / sample)
  agentic.example.yaml    # annotated canonical config; copy to agentic.yaml and edit
  profiles/               # shipped presets + authoring guide
    README.md             # profile/preset authoring guide (see profiles/README.md)
    conformance_256k.yaml # Case-A generated conformance profile (ExplainX targets)
    conformance_512k.yaml # Case-B longer-context conformance profile (ISL tail to 500k)
    inferencex_256k.yaml  # reusable source=hf preset (loader + bundled sweep + Tier 1 knobs)
    small.yaml            # tiny generated profile (fast smoke)
    custom.example.yaml   # annotated template for a user-defined profile
```

The two bash drivers that consume this core live one level up:
`scripts/common/benchmark_agentic_suite.sh` (the suite loop) and
`scripts/common/agentic_lib.sh` (corpus materialization + replay assembly).

## Quick start

A minimal valid `agentic.yaml` is just a workloads list; `serving:` and `run:`
fall back to their defaults:

```yaml
workloads:
  - { name: quick, preset: conformance_256k }
```

With no `serving:`/`run:` blocks, `resolve_config()` supplies `serving.model:
auto`, `serving.max_model_len: 0` (auto-detect), `serving.port: auto` (`auto`
resolves to the recipe default — `2322` for the sglang router — so `auto` and
`2322` name the same port), `serving.server_metrics: auto`, `run.concurrency:
16`, and `run.duration: 900`. See [SCENARIOS.md](SCENARIOS.md) for copy-paste
examples of every workload shape.

### Run it

**Prerequisite:** an OpenAI-compatible endpoint must already be served on
`AGENTIC_PORT` (default `2322`). This suite does **not** start a server — on a
cluster the endpoint is brought up by the launcher recipe (below; see
[../../sglang_disagg/README.MD](../../sglang_disagg/README.MD)), and for a
direct run you must start or point to your own endpoint first. To wire up a new
backend, see the integration guide at
[./templates/README.md](./templates/README.md).

```bash
# Single preset, config-less (driver synthesizes a one-entry config)
AGENTIC_WORKLOAD=conformance_256k bash scripts/common/benchmark_agentic_suite.sh

# Multi-workload suite from a config
AGENTIC_CONFIG=agentic.yaml bash scripts/common/benchmark_agentic_suite.sh

# Preview first — no server needed (see "Preview / debug" below)
DRY_RUN=1 AGENTIC_CONFIG=agentic.yaml bash scripts/common/benchmark_agentic_suite.sh
```

See [Preview / debug](#preview--debug) for what `DRY_RUN=1` prints.

On a cluster you normally invoke the suite **indirectly** via `sbatch
scripts/sglang_disagg/run_xPyD_models.slurm` (see
[../../sglang_disagg/README.MD](../../sglang_disagg/README.MD)); run
`benchmark_agentic_suite.sh` directly only for local or `DRY_RUN` use, and in
that case set `RESULT_DIR` yourself since it is otherwise launcher-provided.

### What to expect / run timing

Per workload, wall-time is dominated by a **cache-warmup** phase followed by the
**measured replay window**:

- **Measured window** — `run.duration` (default `900`s = 15 min); this is the
  aiperf `--benchmark-duration`.
- **Cache warmup** — `--agentic-cache-warmup-duration` (default `60`s; `300`s
  for large model families such as DeepSeek/Kimi/GLM) runs *before* measurement,
  bounded by `--warmup-grace-period` (default `1800`s).
- **Corpus generation** — for `source: profile` workloads this is a one-time
  step cached under `SUITE_CORPUS_DIR` (fast; scales with `n_sessions`) and
  reused on later runs unless `SUITE_CORPUS_FORCE=1`. `source: hf` workloads
  download instead of generate.
- **Concurrency sweeps** — a `concurrency` **list** runs each value in sequence,
  so it multiplies wall-time (see [SCENARIOS.md](SCENARIOS.md)).

## Glossary

- **Profile** — a set of distribution targets (`isl_p`, `osl_p`, `delay_p`,
  `turns`, `cache_hit`, `clamps`) plus a `seed`/`n_sessions` that
  `gen_agentx_profile.py` turns into a reproducible corpus. See
  [profiles/README.md](profiles/README.md).
- **Preset** — a shipped profile file in `profiles/<name>.yaml` that a workload
  inherits via `preset: <name>`. A preset can carry distribution params
  (`source: profile`), an hf `loader` + Tier 1/Tier 2 knobs (`source: hf`),
  and/or run knobs (`concurrency`/`duration`).
- **Workload** — one entry in the `workloads:` list; one result subdir.
- **Corpus** — the per-session `session_XXXXX.json` files aiperf replays. Lives
  under `SUITE_CORPUS_DIR/<name>` (a reusable cache), **not** under `RESULT_DIR`.
- **ISL** — input sequence length (input tokens per request).
- **OSL** — output sequence length (output tokens per request).

**Tier 1 (replay-level knobs)** steer how the trace is replayed and are stripped
from the generator profile dict (`_CONTROL_KEYS` in `agentx_config.py`):

- `concurrency` — session-tree concurrency (scalar or list; a list sweeps).
- `num_dataset_entries` — how many hf trace sessions to pull (hf only).
- `trajectory: { min, max }` — start-window ratio for captured traces.

**Tier 2 (corpus filter)** trims a *downloaded* hf corpus before replay
(`filter_weka_corpus.py`), applied in this order:

- `max_turns` — truncate each session to its first N turns.
- `max_isl` — drop a session if any (post-truncation) turn's input exceeds N.
- `sample` — randomly keep N sessions (fixed `seed=42`).

## Config schema

The schema is defined by `resolve_config()` in `agentx_config.py`. It reads three
top-level keys.

### `serving:` — one endpoint for the whole run

| key | default | notes |
| --- | --- | --- |
| `model` | `auto` | `auto` resolves the served-model id from `/v1/models`; or set explicitly. |
| `max_model_len` | `0` | `0` auto-detects the served window; a set value always wins. See below. |
| `port` | `auto` | `auto` -> recipe default (sglang router `2322` / vLLM shim port). |
| `server_metrics` | `auto` | `auto` -> recipe host:port list; or space-separated endpoints. |

### `run:` — default replay knobs

| key | default | notes |
| --- | --- | --- |
| `concurrency` | `16` | scalar or list; a list sweeps per workload. |
| `duration` | `900` | measured window (s); the scenario minimum for a valid submission is 900. |

### `workloads:` — a list of entries

Each entry is merged over its `preset:` chain (`_merge_preset()`; entry keys win)
and resolved by `_resolve_workload_entry()`. Recognized control keys
(`_CONTROL_KEYS`, stripped from any generator profile):

| key | applies to | meaning |
| --- | --- | --- |
| `name` | all | workload name; becomes the result subdir. |
| `source` | all | `profile` (generate) or `hf` (download). Defaults to `profile`. |
| `preset` | all | inherit `profiles/<name>.yaml`. |
| `loader` | `hf` | aiperf `--public-dataset` id (sets the context-gating ISL tail). |
| `filter` | `hf` | Tier 2 map: `max_isl` / `max_turns` / `sample`. |
| `num_dataset_entries` | `hf` | Tier 1: trace sessions to pull. |
| `trajectory` | all | Tier 1: `{ min, max }` start-window ratio (`0.0 <= min <= max <= 1.0`). |
| `concurrency` | all | per-entry override of `run.concurrency`. |
| `duration` | all | per-entry override of `run.duration`. |

For `source: profile`, the entry additionally carries (or inherits) the profile
distribution fields (`model_tag`, `id_prefix`, `seed`, `n_sessions`,
`block_size`, `isl_p`, `osl_p`, `delay_p`, `turns`, `cache_hit`, `clamps`,
`verify`) documented in [profiles/README.md](profiles/README.md).

## Environment overrides

Environment variables override file values (applied in `resolve_config()`):

- `MODEL` -> `serving.model`
- `MAX_MODEL_LEN` -> `serving.max_model_len`
- `AGENTIC_PORT` -> `serving.port`
- `AGENTIC_SERVER_METRICS` -> `serving.server_metrics`
- `AGENTIC_CONC` -> `run.concurrency`
- `DURATION` -> `run.duration`
- `AGENTIC_WORKLOAD=<name>` — restrict the run to that single named entry (also
  enables the config-less shorthand; see below).

`AGENTIC_WORKLOAD` has **two distinct meanings** depending on whether
`AGENTIC_CONFIG` is set — keep the two straight:

**Config-less shorthand (true minimum, no `AGENTIC_CONFIG`).** `AGENTIC_WORKLOAD`
*names the workload to synthesize*:

```bash
AGENTIC_WORKLOAD=conformance_256k MAX_MODEL_LEN=262144 AGENTIC_CONC=4 \
  bash scripts/common/benchmark_agentic_suite.sh
```

**Filter an existing config to one entry (with `AGENTIC_CONFIG`).** Here
`AGENTIC_WORKLOAD` *selects* the single entry named `quick` from the config
(via `resolve_config()`); it does **not** use the shorthand:

```bash
AGENTIC_CONFIG=agentic.yaml AGENTIC_WORKLOAD=quick \
  bash scripts/common/benchmark_agentic_suite.sh
```

With no `--config`, `_synth_config_from_env()` synthesizes a one-entry config
from `AGENTIC_WORKLOAD`: `inferencex` maps to the shipped hf loader (`_HF_PRESETS`),
and any other name is treated as `preset: <name>` (so `conformance_256k` /
`conformance_512k` resolve to their shipped profiles).

### Minimal required

The true minimum for each entry point — everything else auto-defaults (model
`auto`, `max_model_len` auto-detect, port `2322`, concurrency `16`, duration
`900`). Prerequisite in every case: an OpenAI-compatible endpoint must already be
served on `AGENTIC_PORT` (default `2322`) — launcher-provided on a cluster, or
your own for a direct run.

- **Multi-workload suite** — point at a config, nothing else required:

```bash
AGENTIC_CONFIG=agentic.yaml bash scripts/common/benchmark_agentic_suite.sh
```

- **Single preset (config-less)** — name a preset; the driver synthesizes a
  one-entry config from that name:

```bash
AGENTIC_WORKLOAD=conformance_256k bash scripts/common/benchmark_agentic_suite.sh
```

## Environment variable reference

Every user-facing environment variable, grouped by role. Defaults shown are the
values applied when the variable is unset.

All variables are **optional** except that you must set exactly **one**
entry-point variable (`AGENTIC_CONFIG` or `AGENTIC_WORKLOAD`) and have a served
endpoint (see [Minimal required](#minimal-required)).

### Serving / selection

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODEL` | `auto` | served model id (maps to `serving.model`; auto-discovers from `/v1/models`) |
| `MAX_MODEL_LEN` | `0` (auto) | pin served context window; `0` = auto-detect (`serving.max_model_len`) |
| `AGENTIC_PORT` | `2322` | endpoint/router port (`serving.port`) |
| `AGENTIC_SERVER_METRICS` | `auto` | aiperf `--server-metrics` endpoints; on disaggregated serving its first `host:port` is also probed for the served `max_model_len` |

### Run / replay

| Variable | Default | Meaning |
| --- | --- | --- |
| `AGENTIC_CONC` | `16` | replay concurrency (`run.concurrency`) |
| `DURATION` | `900` | measured window in seconds (`run.duration`) |
| `AGENTIC_NUM_DATASET_ENTRIES` | `393` | default `--num-dataset-entries` for hf workloads (global default; per-workload YAML `num_dataset_entries` overrides) |

### Corpus & HuggingFace

| Variable | Default | Meaning |
| --- | --- | --- |
| `SUITE_CORPUS_DIR` | `${TMPDIR:-/tmp}/agentx_corpora` | corpus cache root |
| `SUITE_CORPUS_FORCE` | `0` | set `1` to regenerate a cached corpus after editing a profile |
| `WEKA_LOADER_OVERRIDE` | recipe default | override the hf trace loader id outside the YAML `loader:` path |
| `AGENTIC_HF_ISL_TAIL` | derived from loader | override the loader-derived ISL tail used for context gating |
| `AGENTIC_TRACE_DL_ATTEMPTS` | `3` | hf trace download retry count |
| `AGENTIC_TRACE_DL_TIMEOUT` | `900` | per-attempt hf download timeout (seconds) |

### Context gating

| Variable | Default | Meaning |
| --- | --- | --- |
| `AGENTIC_MAX_CONTEXT_LENGTH` | derived (falls back to `MAX_MODEL_LEN`) | force the `--max-context-length` cap |
| `AGENTIC_STRICT_CONTEXT` | `0` | set `1` to SKIP (instead of WARN/cap) a workload whose ISL tail exceeds the served window |

### Timing / warmup

| Variable | Default | Meaning |
| --- | --- | --- |
| `AGENTIC_CACHE_WARMUP_DURATION` | `60` (`300` for DeepSeek/Kimi/GLM families) | `--agentic-cache-warmup-duration` |
| `AGENTIC_WARMUP_GRACE_PERIOD` | `1800` | `--warmup-grace-period` |
| `AGENTIC_ROUTER_READY_TIMEOUT` | `600` | seconds to wait for the router/endpoint to become ready before failing |

### Entry points / preview

| Variable | Default | Meaning |
| --- | --- | --- |
| `AGENTIC_CONFIG` | (unset) | Required (choose one of CONFIG/WORKLOAD): path to an `agentic.yaml`; runs the multi-workload suite driver |
| `AGENTIC_WORKLOAD` | (unset) | Required (choose one of CONFIG/WORKLOAD): run one workload by name: config-less preset shorthand, or select a single entry from `AGENTIC_CONFIG` |
| `DRY_RUN` | `0` | set `1` to print the resolved plan and each assembled aiperf command without contacting a server |
| `AGENTIC_RESOLVE_ONLY` | `0` | set `1` to resolve the served `max_model_len` and exit without running; diagnostic |

### Output / labeling

| Variable | Default | Meaning |
| --- | --- | --- |
| `RESULT_DIR` | `/run_logs/${SLURM_JOB_ID:-0}` | root for per-workload result dirs and `suite_summary.json`; the hooks and suite driver default it to `/run_logs/${SLURM_JOB_ID:-0}` (so a launcher supplies `SLURM_JOB_ID`). For a direct run without the launcher, set `RESULT_DIR` yourself (else results land in `/run_logs/0`). |
| `AGENTIC_RESULT_FILENAME` | `agentic_${SLURM_JOB_ID}_xP..._yD..._${MODEL_NAME}` | aggregate result JSON basename |
| `KV_OFFLOADING` | `none` | label threaded into result aggregation |

Advanced / maintainer overrides default to sane values and normally need no
change: `AIPERF_PIN`, `AGENTIC_UTILS_PIN`, `INFERENCEX_REPO`,
`AGENTIC_RUNTIME_DIR`, `INFMAX_WS`, `AIPERF_VENV`,
`AIPERF_FAILED_REQUEST_THRESHOLD`, `AIPERF_UNSAFE_OVERRIDE`,
`AGENTX_YAML_FALLBACK`, `AGENTIC_OUTPUT_DIR`, `AGENTIC_LIB`.

## `max_model_len` guidance

- **Prefer `0` / auto** (the default). The suite auto-detects the served window
  from `/v1/models` (`resolve_served_max_model_len` in `agentic_lib.sh`) and
  never over-estimates it.
- **Pin a value** only when either (a) auto-detect returns `0` because the
  endpoint doesn't expose it (e.g. the vLLM disagg `/v1/models` shim), or
  (b) you want to force a smaller cap than the model actually supports.
- **ISL-tail interaction:** if a workload's ISL tail exceeds the served window,
  `context_compat_check()` WARNs and caps `--max-context-length` at the window;
  set `AGENTIC_STRICT_CONTEXT=1` to SKIP that workload instead.

## Preview / debug

Two distinct mechanisms:

**Config-level (Python, no server).** Inspect what the loader resolves:

```bash
# resolve a YAML profile to JSON (what gen/verify consume)
python3 agentx_config.py --profile profiles/conformance_256k.yaml --emit-json

# dump the fully-resolved config (serving/run/workloads)
python3 agentx_config.py --config agentic.yaml --dump-json

# emit SUITE_* globals the bash driver eval's
python3 agentx_config.py --config agentic.yaml --emit-config-shell

# emit WL_* for one workload (writes the resolved profile JSON to P)
python3 agentx_config.py --config agentic.yaml --workload conformance_256k \
  --profile-out /tmp/p.json --emit-workload-shell
```

**Runtime (bash, no server).** `DRY_RUN=1` prints the resolved N-workload plan
and each assembled aiperf command without contacting an endpoint:

```bash
DRY_RUN=1 AGENTIC_CONFIG=agentic.yaml bash scripts/common/benchmark_agentic_suite.sh
```

See [SCENARIOS.md](SCENARIOS.md#scenario-9-dry_run1-preview) for the exact
`DRY_RUN` output shape.

## Troubleshooting

- **Verify pre-gate fails (`not N/N`).** `materialize_corpus()` aborts the run
  when the corpus doesn't match the profile's own `verify:` targets. Check the
  profile's `verify.turns_p` / `cache_target` / `band_overrides`, and if you just
  edited the profile, regenerate with `SUITE_CORPUS_FORCE=1` (see below).
- **Stale cached corpus.** Corpora are cached at `SUITE_CORPUS_DIR/<name>`
  (default `/tmp/agentx_corpora`). Editing a profile does **not** invalidate the
  cache; set `SUITE_CORPUS_FORCE=1` to regenerate.
- **Unknown preset name.** `preset: <name>` loads `profiles/<name>.yaml`; a
  missing/misspelled name yields an empty base merge (or a `FileNotFoundError`).
  Confirm the file exists under `profiles/`.

## See also

- How/why the replay mechanism works + accuracy: [HOW_IT_WORKS.md](HOW_IT_WORKS.md).

Intentionally out of scope here (pointers only):

- Launcher / disaggregated-serving integration.
- Full env / run-path reference (`agentic_lib.sh` install + endpoint helpers).
- Result-JSON interpretation (`suite_summary.json`, aggregate metrics).
- Post-run health check: `scripts/common/validate_agentic_result.sh`.
