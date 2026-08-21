# AgentX backend integration guide

The AgentX core ([../README.md](../README.md)) speaks only the OpenAI API, so
all backend hook logic now lives in ONE shared, `--backend`-parameterized script
([../../benchmark_agentic.sh](../../benchmark_agentic.sh)). Adding a backend is
two small edits — a `case` arm plus a thin shim — instead of copying and drifting
a full template.

## Adding a backend

1. **Add a `case "$backend"` arm** in
   [`scripts/common/benchmark_agentic.sh`](../../benchmark_agentic.sh) that sets
   the backend's `AGENTIC_PORT` default and its ctx-window resolver endpoint list
   (`ctx_endpoints=("path|kind" ...)`, where `kind` is `models` to read
   `data[0].max_model_len` from an OpenAI `/v1/models` ModelCard, or anything
   else to read `max_model_len`/`context_length`/`server_args.*` — e.g. sglang's
   `/get_server_info`). Single-endpoint backends list one entry; add more only if
   your framework serves the window elsewhere.
2. **Add the `$backend` value** to the validation and dispatch `case`s (mirror
   the existing `sglang|vllm` arms).
3. **Add a ~7-line shim** at `scripts/<backend>_disagg/benchmark_agentic.sh` that
   locates the shared script (its `../common/benchmark_agentic.sh` sibling, plus
   any launcher repo-dir env var if the in-container mount hides `../common`, plus
   an `AGENTIC_LIB` override) and `exec bash "$_cand" --backend <backend> "$@"`.
   Copy an existing shim
   ([sglang](../../../sglang_disagg/benchmark_agentic.sh),
   [vllm](../../../vllm_dissag/benchmark_agentic.sh)).
4. **Wire it into the launcher:** it is selected via
   `export BENCHMARK_SCRIPT_FILE=benchmark_agentic.sh` (typically exposed to users
   as `BENCHMARK_SCRIPT=agentic`).
5. Preview with `DRY_RUN=1` (no server needed), then run against a live endpoint.

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

## Disaggregated context-window resolution

Disaggregated P/D front-ends (a router, proxy, or `/v1/models` shim) often do
**not** advertise `max_model_len`, so the library's front-end auto-detect returns
`0`. The shared script ships a `# === agentx:BEGIN resolve served context window
(disagg) ===` block that instead probes the prefill **WORKER** — the first
`host:port` in `AGENTIC_SERVER_METRICS` — over each entry in the backend's
`ctx_endpoints`. On the shipped launchers `AGENTIC_SERVER_METRICS` is
auto-derived in-container, so users normally never set it. Single-node / monolith
backends need no worker probe — an empty `AGENTIC_SERVER_METRICS` falls through
to the library's front-end `/v1/models` auto-detect.

`AGENTIC_RESOLVE_ONLY=1` resolves the served `max_model_len`, prints it, and
exits without running — a diagnostic for checking the probe. It is **not**
forwarded through the launchers, so use it in a direct/local run.

**Intentional probe divergence:** sglang probes `/v1/models` **and**
`/get_server_info` (older builds only expose it there), while vLLM probes only
`/v1/models`. This lives in each backend's `ctx_endpoints` — keep per-backend
lists rather than forcing one shared list.
