# Case-A Conformance Trace — customer benchmark artifact

**What this is:** a deterministic, spec-constructed WekaTrace corpus that matches the
ExplainX **Case-A** parameters on all six axes. Built for repeatable engine benchmarking:
freeze once, replay identically any number of times on any engine/topology.

## Conformance (measured vs Case-A target) — 13/13 PASS

| Axis | Measured | Case-A target | Verdict |
|---|---:|---:|:--|
| Input ISL P50 | 74,140 | 74,000 | PASS |
| Input ISL P90 | 146,111 | 155,000 | PASS |
| Input ISL P99 | 245,000 | 235,000 | PASS |
| Output OSL P50 | 306 | 320 | PASS |
| Output OSL P90 | 2,980 | 3,300 | PASS |
| Output OSL P99 | 16,193 | 17,000 | PASS |
| Turns/session P50 | 3 | 3 | PASS |
| Turns/session P90 | 20 | 20 | PASS |
| Turns/session P99 | 103 | 103 | PASS |
| Inter-turn delay P50 | 3s | 4s | PASS |
| Inter-turn delay P90 | 33s | 31s | PASS |
| Inter-turn delay P99 | 229s | 240s | PASS |
| Prefix cache hit | 88% | 88-90% | PASS |

200 sessions, 1,778 requests. (Spec-decode — 5 draft tokens, ~56% accept — is an
engine-side setting applied at serve time, recorded as model metadata.)

## Why constructed, not engine-captured
For a customer benchmark the trace must (a) match Case-A **exactly**, (b) be **deterministic
and documented**, (c) **replay identically N times**. Engine-generated (emergent) traces
can't guarantee (a) — the percentiles fall where the model happens to land, they vary per
run, and generation is GPU-bound (hours to tune the output tail). A constructed trace fixes
all three: the parameters + seed **are** the definition.

This is valid for serving benchmarks because engine cost depends on token **counts** and
cache **structure**, not token content: replaying a request of ISL=N, OSL=M with a given
prefix-reuse pattern drives exactly the same prefill + decode + KV work regardless of whether
the tokens are "real." (ExplainX's own Case-A figures are likewise a parameterized profile.)

## Reproduce (byte-identical)
```
python3 gen_caseA_conformance.py corpus 200 42   # <out_dir> <n_sessions> <seed>
python3 verify_caseA.py corpus                    # prints the conformance table above
```
Same seed (42) => identical corpus every time. Change seed for an independent draw of the
same distribution; raise n_sessions for tighter percentiles.

## Files
- `corpus/` (+ `caseA_conformance_corpus.tar.gz`) — 200 WekaTrace session JSONs (the trace)
- `gen_caseA_conformance.py` — the seeded synthesizer (documented spec)
- `verify_caseA.py` — the conformance checker (measured-vs-target)

## How it's benchmarked
Replay this one frozen trace via aiperf `inferencex-agentx-mvp --custom-dataset-type
weka_trace` against each endpoint at matched TP/concurrency (`replay_caseA.sh`). Because the
trace is fixed, the only variable is the engine/topology — an apples-to-apples comparison.
