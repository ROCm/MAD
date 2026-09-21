# AgentX Case-B conformance trace — engine-agnostic serving benchmark (GLM-5.2-class, MI355X)

A **deterministic, spec-constructed agentic replay trace** that matches the customer **Case-B**
workload on all six axes, plus a generic replay driver. Freeze once, replay identically against
**any** serving setup — 1-node aggregated, 2-node, disaggregated 1P1D, any TP/EP — to benchmark
tok/s, TTFT, ITL, E2E, and prefix-cache behaviour. Only the endpoint URL changes.

Case-B is the **large-context** sibling of Case-A: bigger inputs (up to 500K tokens), a heavier
multi-turn distribution, and modest outputs.

## What Case-B is, and how this trace matches it (13/13 axes, verified)

`verify_caseB.py`, 300 sessions, seed 42:

| Axis | Measured | Case-B target | Verdict |
|---|---:|---:|:--|
| Input ISL P50 / P90 / P99 | 61,490 / 204,765 / 520,000 | 62K / 220K / 500K | PASS |
| Output OSL P50 / P90 / P99 | 176 / 1,278 / 7,151 | 180 / 1.4K / 7K | PASS |
| Turns/session P50 / P90 / P99 | 5 / 82 / 144 | 5 / 82 / 144 | PASS |
| Inter-turn delay P50 / P90 / P99 | 3 / 30 / 152 s | 3.6 / 23 / 240 s | PASS |
| Prefix cache hit | 88% | 88% | PASS |

Spec-decode target: **3 draft tokens, 44% acceptance** (engine-side serve setting, applied at
serve time; recorded as model metadata, not a replay axis).

## Why a constructed trace (not an engine capture)

- **Exact & repeatable.** The parameters + seed *are* the definition. Seed 42 → byte-identical
  trace every run, so results are reproducible and the trace is documentable for later playbacks.
- **Engine-captured (emergent) traces can't guarantee the distribution** — percentiles land
  wherever the model happens to, vary per run, and generation is GPU-bound.
- **Valid for serving benchmarks:** engine cost depends on token **counts** and cache
  **structure**, not token content. Replaying a request of ISL=N, OSL=M with a given prefix-reuse
  pattern drives exactly the same prefill + decode + KV work as a "real" one of the same shape.

## Replays to any topology

Open-loop: the trace records demand (per-request in/out tokens, turn structure, think-times, KV
prefix reuse), nothing engine- or topology-specific. To compare configs, point `URL` at each:

```bash
URL=http://<agg_node>:8801   ./replay_caseB.sh   # single-node aggregated
URL=http://<router_ip>:30000 ./replay_caseB.sh   # 2-node / disaggregated 1P1D router
```

## ⚠ Serving requirement — 500K context

Case-B's input **P99 = 500K tokens** (Case-A was 235K). The endpoint under test **must be
launched with `--max-model-len 524288`** (512K window). KV memory per request is ~2× Case-A, so
expect a lower `--max-num-seqs` and heavier prefill pressure. GLM-5.2 supports up to a 1M window,
so this is within envelope but memory-tight.

## Prerequisites

- **aiperf** with the AgentX scenario (`inferencex-agentx-mvp`, `--custom-dataset-type
  weka_trace`) — SemiAnalysis fork:
  `pip install "git+https://github.com/SemiAnalysisAI/aiperf.git@cquil11/aiperf-agentx-v1.0"`.
- An **OpenAI-compatible endpoint** serving the model at `--max-model-len 524288`.
- The model **tokenizer** (local path or HF id).

## Usage

```bash
# 1. Materialize the trace (regenerate deterministically, or untar the frozen copy)
python3 gen_caseB_conformance.py corpus 300 42        # <out_dir> <n_sessions> <seed>
#   or:  tar xzf caseB_conformance_corpus.tar.gz       # unpacks corpus/

# 2. Confirm conformance (prints the 13/13 table above)
python3 verify_caseB.py corpus

# 3. Replay against your endpoint (sweeps concurrency, writes results/summary.csv)
URL=http://localhost:8801 SERVED=GLM-5.2-MXFP4 TOK=/models/GLM-5.2-MXFP4 ./replay_caseB.sh
```

`replay_caseB.sh` auto-unpacks the frozen trace if `corpus/` is absent, so step 1 is optional.

## Overridable env (`replay_caseB.sh`)

| Var | Default | Meaning |
|-----|---------|---------|
| `URL` | *(required)* | endpoint under test |
| `SERVED` | `GLM-5.2-MXFP4` | served-model-name |
| `TOK` | `/models/GLM-5.2-MXFP4` | tokenizer path or HF id |
| `CONCS` | `1 2 4 8 16` | concurrency points |
| `DUR` | `300` | seconds per point (≥900 to let long requests finish) |
| `CORP` | `./corpus` | trace directory (auto-unpacked if empty) |
| `OUT` | `./results` | output dir |
| `IMG` | `rocm/atom-dev:latest` | container to run aiperf in (`IMG=""` for bare-metal) |

## Files

| File | Role |
|------|------|
| `gen_caseB_conformance.py` | Seeded synthesizer — Case-B params → WekaTrace (deterministic). |
| `verify_caseB.py` | Conformance checker — measured-vs-target table (13/13). |
| `replay_caseB.sh` | Generic aiperf replay/sweep driver against any `URL`. |
| `caseB_conformance_corpus.tar.gz` | The frozen 300-session trace. |
| `MANIFEST.md` | Conformance proof + provenance. |

## Notes

- **conc=1** is not supported by the aiperf agentic-replay scenario (its warmup builds one
  trajectory lane per concurrency; a single lane floors the warmup-credit count to 0). Sweep from
  conc≥2, or use a non-agentic `--fixed-schedule` replay for a single-stream number.
- Reported cache-hit is the endpoint's realized server-side prefix hit; the trace's constructed
  reuse structure is ~88% (see MANIFEST).
