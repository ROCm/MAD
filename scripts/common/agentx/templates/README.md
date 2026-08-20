# AgentX backend integration guide

The AgentX core ([../README.md](../README.md)) speaks only the OpenAI API, so
adding a new serving backend is a thin per-backend **hook** — a ~60-line bash
script that discovers the shared library, sets a serve port, and calls six
library functions in order. This directory ships a copy-paste template for that
hook.

## Copy-paste workflow

1. Copy [`benchmark_agentic.template.sh`](benchmark_agentic.template.sh) into
   your launcher's script dir as `benchmark_agentic.sh`.
2. Fill the three `# CHANGE:` fields (see below).
3. Wire it into the launcher: it is selected via
   `export BENCHMARK_SCRIPT_FILE=benchmark_agentic.sh` (typically exposed to users
   as `BENCHMARK_SCRIPT=agentic`).
4. Preview with `DRY_RUN=1` (no server needed), then run against a live endpoint.

Two working, integrated hooks to copy from:

- SGLang disaggregated P/D: [../../../sglang_disagg/benchmark_agentic.sh](../../../sglang_disagg/benchmark_agentic.sh)
- vLLM disaggregated P/D: [../../../vllm_dissag/benchmark_agentic.sh](../../../vllm_dissag/benchmark_agentic.sh)

## The three fields to fill

1. **Backend name** — the `<YOUR BACKEND>` header comment (cosmetic).
2. **`AGENTIC_PORT`** — your router/proxy serve port (the template samples
   `8000`; sglang uses `2322`, vLLM chains `BENCHMARK_PORT`/`PROXY_PORT`).
3. **Lib-discovery repo-dir candidate** (optional) — if your launcher mounts the
   repo somewhere whose sibling `common/` is not reachable via `../common`, add
   your launcher's repo-dir env var to the discovery loop. Otherwise leave the
   sample candidate as-is (it is harmless) or set `AGENTIC_LIB` at runtime.

## Backend contract

The core talks to your endpoint over two OpenAI-compatible routes:

- `POST /v1/chat/completions` — streaming chat, used for the trace replay.
- `GET /v1/models` — model discovery and readiness gating
  (`wait_for_router_ready` and `resolve_served_model_name` both poll this;
  `/v1/models` `data[0].max_model_len` also drives context auto-detection).

If your framework does not serve `/v1/models` (or returns 503 while workers
register), front it with a tiny shim that answers `/v1/models` once the upstream
is healthy and proxies everything else. See the working example
[../../../vllm_dissag/agentic_models_shim.py](../../../vllm_dissag/agentic_models_shim.py)
and point `AGENTIC_PORT` at the shim.

## The six library functions

The hook calls these in order (defined in
[../../agentic_lib.sh](../../agentic_lib.sh)):

1. `install_agentic_deps` — build the isolated, pinned aiperf uv venv.
2. `resolve_trace_source` — pick the trace loader/dataset and download it.
3. `wait_for_router_ready` — block until `GET /v1/models` answers on `AGENTIC_PORT`.
4. `resolve_served_model_name` — read the served model id from `/v1/models`.
5. `build_replay_cmd` — assemble the aiperf `inferencex-agentx-mvp` command.
6. `run_agentic_replay_and_write_outputs` — run the replay, aggregate JSON, plots.

## Three integration shapes

1. **Single-node monolith** — one process serving `/v1/*` on one port. Set
   `AGENTIC_PORT` to that port and `RESULT_DIR` to a real dir, then run the hook
   directly (or with `DRY_RUN=1`). No `sbatch`.
2. **Disaggregated P/D** — mirror the shipped hooks: point `AGENTIC_PORT` at the
   router/proxy and let the launcher provide `RESULT_DIR`
   ([sglang](../../../sglang_disagg/benchmark_agentic.sh),
   [vllm](../../../vllm_dissag/benchmark_agentic.sh)).
3. **Other frameworks (TRT-LLM / TGI / Triton)** — change `AGENTIC_PORT` to the
   framework's OpenAI port and confirm `/v1/chat/completions` + `/v1/models`
   (add the shim if `/v1/models` is missing).

## Disaggregated backends

Disaggregated P/D front-ends (a router, proxy, or `/v1/models` shim) often do
**not** advertise `max_model_len`, so the library's front-end auto-detect returns
`0`. The template ships a `# === agentx:BEGIN resolve served context window
(disagg) ===` block that instead probes the prefill **WORKER** for the served
window. Wiring it up:

1. **Point the probe at your worker.** The block reads the first `host:port` from
   `AGENTIC_SERVER_METRICS` (`<prefill-host:port> [<decode-host:port> ...]`) — so
   the first entry must be the real worker whose OpenAI server exposes the window.
   On the shipped sglang/vLLM launchers the recipe auto-derives
   `AGENTIC_SERVER_METRICS` in-container, so users normally never set it; a fresh
   backend integrator must ensure their worker's `host:port` lands there.
2. **Fill the `# CHANGE:` probe** for your framework: the generic default parses
   `data[0].max_model_len` from the worker's `/v1/models`. If your framework
   serves the window on a different or second endpoint, edit the endpoint/JSON
   (a commented sglang-style `/get_server_info` fallback is included to enable).
3. **Single-node / monolith backends can delete the whole block** — the library's
   front-end `/v1/models` auto-detect already covers them.

`AGENTIC_RESOLVE_ONLY=1` resolves the served `max_model_len`, prints it, and
exits without running — a diagnostic for checking the probe. It is **not**
forwarded through the launchers, so use it in a direct/local run of the hook.

**Intentional probe divergence:** the sglang hook tries `/v1/models` **and**
`/get_server_info` (older builds only expose it there), while the vLLM hook tries
only `/v1/models`. This is deliberate — don't "unify" them or older sglang breaks.

## Required vs optional env

Set exactly **one** entry-point variable — `AGENTIC_CONFIG` (a config path) or
`AGENTIC_WORKLOAD` (a single-workload name) — and have a live endpoint on
`AGENTIC_PORT`. Everything else is optional and auto-defaults. See the core
[env reference](../README.md#environment-variable-reference) and
[Minimal required](../README.md#minimal-required), and
[../SCENARIOS.md](../SCENARIOS.md) for copy-paste configs.
