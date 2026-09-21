# Case-B Conformance Trace — customer benchmark artifact

Deterministic, spec-constructed WekaTrace matching the customer **Case-B** parameters on all
six axes. Same methodology as Case-A (see `../AgentX_CaseA/`), new percentile targets.

## Conformance (measured vs Case-B target) — 13/13 PASS

| Axis | Measured | Case-B target | Verdict |
|---|---:|---:|:--|
| Input ISL P50 / P90 / P99 | 61,490 / 204,765 / 520,000 | 62K / 220K / 500K | PASS |
| Output OSL P50 / P90 / P99 | 176 / 1,278 / 7,151 | 180 / 1.4K / 7K | PASS |
| Turns/session P50 / P90 / P99 | 5 / 82 / 144 | 5 / 82 / 144 | PASS |
| Inter-turn delay P50 / P90 / P99 | 3 / 30 / 152 s | 3.6 / 23 / 240 s | PASS |
| Prefix cache hit | 88% | 88% | PASS |

300 sessions, ~6,900 requests, seed 42 (byte-identical on regeneration).
Spec-decode target: **3 draft tokens, 44% acceptance** (engine-side setting, applied at serve time).

## Key difference vs Case-A (serving impact)
- **Input P99 = 500K tokens** (Case-A was 235K). Requires serving at **`--max-model-len 524288`**
  (512K window). KV memory per request ~2× Case-A → expect lower `--max-num-seqs` and more
  prefill pressure.
- **Much more multi-turn**: turns P90 82 (Case-A: 20), P99 144. Heavier prefix-cache dependence.

## Reproduce
```
python3 gen_caseB_conformance.py corpus 300 42   # <out_dir> <n_sessions> <seed>
python3 verify_caseB.py corpus                    # prints the table above
```

## Files
- `corpus/` (+ `caseB_conformance_corpus.tar.gz`) — 300 WekaTrace session JSONs
- `gen_caseB_conformance.py` — seeded synthesizer (Case-B params)
- `verify_caseB.py` — conformance checker
