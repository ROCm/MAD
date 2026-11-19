# AgentX Case-A conformance trace — engine-agnostic serving benchmark (GLM-5.2-class, MI355X)

A **deterministic, spec-constructed agentic replay trace** that matches the ExplainX **Case-A**
workload on all six axes, plus a generic replay driver. Freeze once, replay identically against
**any** serving setup — 1-node aggregated, 2-node, disaggregated 1P1D, any TP — to benchmark
tok/s, TTFT, ITL, E2E, and prefix-cache behaviour. Only the endpoint URL changes.

## What Case-A is, and how this trace matches it

Case-A is a long-context agentic profile: large inputs, a long-tailed output/turn distribution,
realistic inter-turn delays, and heavy prefix-cache reuse. This trace is **constructed to those
parameters** and verified 13/13 axes (`verify_caseA.py`, 200 sessions, seed 42):

| Axis | Measured | Case-A target | Verdict |
|---|---:|---:|:--|
| Input ISL P50 / P90 / P99 | 74,140 / 146,111 / 245,000 | 74K / 155K / 235K | PASS |
| Output OSL P50 / P90 / P99 | 306 / 2,980 / 16,193 | 320 / 3.3K / 17K | PASS |
| Turns/session P50 / P90 / P99 | 3 / 20 / 103 | 3 / 20 / 103 | PASS |
| Inter-turn delay P50 / P90 / P99 | 3 / 33 / 229 s | 4 / 31 / 240 s | PASS |
| Prefix cache hit | 88% | 88-90% | PASS |

## Why a constructed trace (not an engine capture)

- **Exact & repeatable.** The parameters + seed *are* the definition. Seed 42 → byte-identical
  trace every run, so results are reproducible and the trace is documentable for later playbacks.
- **Engine-captured (emergent) traces can't guarantee the distribution** — percentiles land
  wherever the model happens to, vary per run, and generation is GPU-bound (hours for the 17K
  output tail).
- **Valid for serving benchmarks:** engine cost depends on token **counts** and cache
  **structure**, not token content. Replaying a request of ISL=N, OSL=M with a given prefix-reuse
  pattern drives exactly the same prefill + decode + KV work as a "real" one of the same shape.
  (ExplainX's own Case-A figures are likewise a parameterized profile.)

## Replays to any topology

The trace is **open-loop**: it records demand (per-request in/out tokens, turn structure,
think-times, KV prefix reuse), nothing engine- or topology-specific. To compare configs, point
`URL` at each and replay the same trace:

```bash
URL=http://<agg_node>:8801   ./replay_caseA.sh   # single-node aggregated
URL=http://<router_ip>:30000 ./replay_caseA.sh   # 2-node / disaggregated 1P1D router
```

## Prerequisites

- **aiperf** with the AgentX scenario (`inferencex-agentx-mvp`, `--custom-dataset-type
  weka_trace`) — SemiAnalysis fork:
  `pip install "git+https://github.com/SemiAnalysisAI/aiperf.git@cquil11/aiperf-agentx-v1.0"`.
- An **OpenAI-compatible endpoint** serving the model under test (`/v1/chat/completions`).
- The model **tokenizer** (local path or HF id).
- Optional: a container image with aiperf installed (the driver runs aiperf in a container by
  default — see the note in `replay_caseA.sh`).

## Usage

```bash
# 1. Materialize the trace (either regenerate deterministically, or untar the frozen copy)
python3 gen_caseA_conformance.py corpus 200 42        # <out_dir> <n_sessions> <seed>
#   or:  mkdir corpus && tar xzf caseA_conformance_corpus.tar.gz -C corpus --strip-components=1

# 2. Confirm conformance (prints the 13/13 table above)
python3 verify_caseA.py corpus

# 3. Replay against your endpoint (sweeps concurrency, writes results/summary.csv)
URL=http://localhost:8801 SERVED=GLM-5.2-MXFP4 TOK=/models/GLM-5.2-MXFP4 ./replay_caseA.sh
```

`replay_caseA.sh` auto-unpacks the frozen trace if `corpus/` is absent, so step 1 is optional.

## Overridable env (`replay_caseA.sh`)

| Var | Default | Meaning |
|-----|---------|---------|
| `URL` | *(required)* | endpoint under test, e.g. `http://host:port` |
| `SERVED` | `GLM-5.2-MXFP4` | served-model-name the endpoint answers to |
| `TOK` | `/models/GLM-5.2-MXFP4` | tokenizer path or HF id |
| `CONCS` | `1 2 4 8 16` | concurrency points to sweep |
| `DUR` | `300` | seconds per point (use ≥900 to let 17K-output requests complete) |
| `CORP` | `./corpus` | trace directory (auto-unpacked if empty) |
| `OUT` | `./results` | output dir (`summary.csv` + per-conc artifacts) |
| `IMG` | `rocm/atom-dev:latest` | container to run aiperf in (set `IMG=""` to run bare-metal) |
| `AIPERF` | `aiperf` | aiperf binary path |

## Files

| File | Role |
|------|------|
| `gen_caseA_conformance.py` | Seeded synthesizer — Case-A params → WekaTrace (deterministic). |
| `verify_caseA.py` | Conformance checker — measured-vs-target table (13/13). |
| `replay_caseA.sh` | Generic aiperf replay/sweep driver against any `URL`. |
| `caseA_conformance_corpus.tar.gz` | The frozen 200-session trace (replay without regenerating). |
| `MANIFEST.md` | Conformance proof + provenance. |

## Notes

- **conc=1** is not supported by the aiperf agentic-replay scenario (its warmup builds
  `concurrency` trajectory lanes and the warmup-credit count floors to 0 at a single lane). Sweep
  from conc≥2, or use a non-agentic `--fixed-schedule` replay for a single-stream number.
- Reported cache-hit is the endpoint's realized server-side prefix hit; the trace's *constructed*
  reuse structure is ~88% (see MANIFEST).
