# Authoring AgentX profiles & presets

A **profile** is ~6 distribution targets plus a `seed` and a session count.
`gen_agentx_profile.py` turns it into a reproducible `weka_trace` corpus, and
`verify_agentx_profile.py` gates that corpus against the same targets. There is
**no per-case code** — you add a workload by adding a profile file here.

- New to the core concepts? Start at [../README.md](../README.md).
- Want worked, copy-paste examples? See [../SCENARIOS.md](../SCENARIOS.md).

## Anatomy of a profile

Fields below are shown against [conformance_256k.yaml](conformance_256k.yaml) (the
generalized ROCm/MAD #173 Case-A profile). They are consumed by
`generate_corpus()` in `gen_agentx_profile.py` and `verify()` in
`verify_agentx_profile.py`.

```yaml
name: conformance_256k        # informational workload name
model_tag: GLM-5.2-MXFP4      # written into requests[].model + models[]; retag per served model
id_prefix: caseA              # session-id salt (see note below)
seed: 42                      # fixed seed => byte-identical corpus every run
n_sessions: 200               # sessions to generate (more => tighter percentiles, bigger corpus)
block_size: 64                # KV block size (tokens/block)

# Distribution targets, each a P50 / P90 / P99 triple:
isl_p:   [74000, 155000, 235000]   # input tokens per request
osl_p:   [320, 3300, 17000]        # output tokens per request
delay_p: [4, 31, 240]              # inter-turn think delay (seconds)

# Turns-per-session discrete distribution (parallel lists):
turns:
  values:  [1, 2, 3, 4, 6, 10, 20, 45, 103]
  weights: [22, 24, 20, 12, 6, 5, 6, 3, 2]

cache_hit: [0.88, 0.90]       # per-turn prefix-reuse band [lo, hi]

clamps:                       # post-draw sampling clamps [lo, hi]
  isl:   [1200, 245000]
  osl:   [8, 20000]
  delay: [1, 600]

# Optional verifier block:
verify:
  turns_p: [3, 20, 103]       # Turns P50/P90/P99 targets
  cache_target: 89            # Cache-hit P50 % target
  # band_overrides:           # widen a per-axis tolerance band, e.g.:
  #   "Input ISL P99": [0.75, 1.25]
```

### What the verifier checks (13 axes)

`verify_agentx_profile.py` measures the corpus and prints a per-axis table whose
verdict tokens are `PASS` / `off`, then a `N/N axes within band` summary. The 13
axes are: **ISL** P50/P90/P99, **OSL** P50/P90/P99, **Turns** P50/P90/P99,
**Delay** P50/P90/P99, and **Cache hit P50 %**.

Default tolerance bands (from `DEFAULT_BANDS`):

| group | band (lo–hi multipliers) |
| --- | --- |
| isl | 0.80 – 1.20 |
| osl | 0.70 – 1.40 |
| turns | 0.60 – 1.60 |
| delay | 0.50 – 2.00 |
| cache | 0.97 – 1.03 |

Override a single axis with `verify.band_overrides` keyed by the exact axis label
(e.g. Case-B in [conformance_512k.yaml](conformance_512k.yaml) widens
`"Input ISL P99"` to `[0.75, 1.25]`). Targets for turns/cache come from
`verify.turns_p` / `verify.cache_target` when present; otherwise turns targets are
derived from the `turns` distribution and the cache target from `mean(cache_hit) * 100`.

## Field constraints

- **Percentiles monotonic:** each `*_p` triple should satisfy `P50 <= P90 <= P99`
  (the lognormal fit in `lognorm_from_p()` assumes an increasing triple).
- **Equal-length turns arrays:** `turns.values` and `turns.weights` must be the
  same length (they are zipped in `rng.choices(...)` and the weighted-percentile
  derivation).
- **`cache_hit` is a `[lo, hi]` band** with `0 <= lo <= hi <= 1` (used as
  `rng.uniform(cache_lo, cache_hi)` per turn).
- **`clamps` are `[lo, hi]` pairs** for `isl` / `osl` / `delay`; each draw is
  clamped into `[lo, hi]`.

## Create a profile from scratch

Use the **flag-based** CLIs (not any positional form). Round-trip needs no GPU:

```bash
# 1. Start from the annotated template.
cp profiles/custom.example.yaml profiles/my_case.yaml
#    edit name/id_prefix/targets to taste

# 2. Resolve YAML -> JSON (what gen/verify consume).
python3 agentx_config.py --profile profiles/my_case.yaml --emit-json > /tmp/my.json

# 3. Generate a corpus (overrides optional).
python3 gen_agentx_profile.py --profile /tmp/my.json --out-dir /tmp/my_corpus \
  [--n-sessions N --seed S --model-tag TAG --id-prefix P --block-size B]

# 4. Verify until it passes.
python3 verify_agentx_profile.py --profile /tmp/my.json --corpus /tmp/my_corpus
#    -> ends with "13/13 axes within band" (exit 0) when all axes PASS
```

Then reference it from `agentic.yaml`:

```yaml
workloads:
  - { name: my_case, preset: my_case }
```

If the verifier reports an `off` axis, nudge the offending `*_p` target (or widen
that axis via `verify.band_overrides`) and re-run steps 3–4.

## Preset inheritance

`_merge_preset()` (in `agentx_config.py`) merges an entry over its `preset:`
chain, **entry keys win** over inherited ones. A preset may bundle distribution
params (`source: profile`), an hf `loader` + Tier 1/Tier 2 knobs
(`source: hf`, see [inferencex_256k.yaml](inferencex_256k.yaml)), and/or run
knobs (`concurrency` / `duration`). A **circular** preset chain raises
`ValueError: circular preset: <name>`.

```yaml
# preset carries the sweep; the entry overrides just concurrency
- name: inferencex_small
  preset: inferencex_256k
  concurrency: [2]        # wins over the preset's [2, 4, 8]
```

## The verify pre-gate

At run time `materialize_corpus()` (in `../agentic_lib.sh`) generates the corpus
then runs `verify_agentx_profile.py`, and **aborts the run** unless the corpus is
`N/N axes within band` against the profile's own `verify:` targets. Corpora are
cached at `SUITE_CORPUS_DIR/<name>` (default `/tmp/agentx_corpora`); editing a
profile does not invalidate the cache, so regenerate with `SUITE_CORPUS_FORCE=1`.

## Placement rule

Drop the file at `profiles/<name>.yaml` so that `preset: <name>` (and the
config-less `AGENTIC_WORKLOAD=<name>` shorthand) resolves it.

## Common mistakes

- **Circular preset chain** — `A` presets `B` presets `A` -> `ValueError:
  circular preset`.
- **Mismatched `turns` lengths** — `values` and `weights` must be equal length.
- **Non-monotonic percentiles** — a `*_p` triple that isn't increasing skews the
  lognormal fit and fails verification.
- **Editing a profile without `SUITE_CORPUS_FORCE=1`** — the stale cached corpus
  is reused and your edits appear to have no effect.

## Note on `id_prefix`

The conformance presets pin `id_prefix: caseA` (the literal salt #173 used for
**both** Case-A and Case-B) so regeneration is byte-identical to the committed
ROCm/MAD #173 corpora. For a new workload, use a distinct `id_prefix` — it is
just a session-id salt that changes corpus identity. Start from
[custom.example.yaml](custom.example.yaml).
