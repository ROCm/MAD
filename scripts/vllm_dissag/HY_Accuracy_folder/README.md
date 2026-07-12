# HY_Accuracy_folder — Hy3 WideEP accuracy test suite

Self-contained accuracy tooling for the MoRI-EP disaggregated stack. Validates that a
disagg config produces *correct* output, not just HTTP-200 throughput — critical because
some configs (e.g. EP32) emit silent KV-corruption garbage while perf benchmarks still
report "successful". Used to establish EP16 = 100% vs EP32 = 0% on Hy3-preview.

## Files
| File | Purpose |
|---|---|
| `benchmark_accuracy_hy3.sh` | Orchestrator. Runs the tiers below + prints a combined PASS/FAIL verdict. Invoked by the launcher via `BENCHMARK_SCRIPT=accuracy`. |
| `accuracy_eval.py` | Tier 1 — known-answer set (~20 factual/math/code prompts), scored vs ground truth. (Optional GSM8K mode behind `--gsm8k N`.) |
| `niah_probe.py` | Tier 2 — needle-in-haystack long-context retrieval at 54k/128k/256k. |
| `accuracy_probe.py` | Tier 3 — greedy exact-match equivalence vs a golden config (capture/compare). |
| `benchmark_hold.sh` | Debug helper — `BENCHMARK_SCRIPT=hold` keeps the server up N hours for live poking (does not score). |

## How to run
Via the launcher (in-container, recommended):
```bash
export BENCHMARK_SCRIPT=accuracy        # selector wired in run_xPyD_models.slurm
# ... normal MODEL_NAME=Hy3-preview / RUN_MORI=1 / xP / yD submit ...
```
Or manually against a serving disagg server (run INSIDE the container):
```bash
bash HY_Accuracy_folder/benchmark_accuracy_hy3.sh http://127.0.0.1:30000 <tag> [golden.json]
```

## CRITICAL — disagg request protocol
The MoRIIO router only injects KV-routing (prefill/decode addressing) for requests that
match the `vllm bench serve` client shape. Raw HTTP probes crash the decode engine with
`KeyError: 'remote_host'`. The probes therefore send:
- `"stream": true` (parse SSE chunks), and
- an explicit `x-request-id` header.
Run them from **inside the serving container** (`docker exec`) against `127.0.0.1:30000`,
not from the host.

## Scoring
Greedy (temperature 0), deterministic. Known-answer scoring matches the **first answer
line** (stop at `\n`) with **word boundaries**, so a short answer like `3` must appear as
the answer token — not buried in over-generated continuation or coincidental garbage.

## Status
Known-answer tier validated (EP16 100% / EP32 0%). NIAH + equivalence tiers included;
NIAH not yet run end-to-end at the time of writing.
