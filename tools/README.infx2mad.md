# infx2mad — run InferenceX e2e scenarios through MAD

[InferenceX](https://github.com/SemiAnalysisAI/InferenceX) publishes the AMD
serving scenarios SemiAnalysis measures. `tools/infx2mad.py` turns one of those
scenarios into the artifacts madengine already runs, so we reproduce their
numbers with our harness instead of growing a second one.

## Why a translator and not a new harness

49 of the 52 AMD scenarios map onto a MAD runner that already exists:

| InferenceX framework | MAD runner |
| --- | --- |
| `sglang` | `scripts/sglang` |
| `vllm` | `scripts/vllm` |
| `atom` | `scripts/atom` |
| `sglang-disagg` | `scripts/sglang_disagg` |

So this is a **router**: read the scenario, pick the runner that serves its
framework, emit that runner's config. Adding a scenario is config, not code.
`scripts/sglang`, `scripts/vllm` and `scripts/atom` share one config schema, so
one emitter serves all three.

## Usage

```bash
git clone --depth 1 https://github.com/SemiAnalysisAI/InferenceX.git
INFX=InferenceX/inferencex-e2e

# one scenario
python3 tools/infx2mad.py --infx "$INFX" \
    --scenario dsr1-fp8-mi355x-sglang \
    --weights-dir /path/to/DeepSeek-R1-0528 \
    --out-dir out

# everything that maps to a runner on this cluster, with the gaps named
python3 tools/infx2mad.py --infx "$INFX" --all --out-dir out

# check every generated block against the target runner's CSV schema
python3 tools/validate_configs.py . out
```

Each scenario yields three artifacts: the runner's config YAML, a madengine
manifest, and a `models.json` entry for a plain `madengine run --tags`.

```bash
madengine run -m out/dsr1-fp8-mi355x-sglang.manifest.json -o perf.csv --live-output
```

## Clusters are data

`--cluster-file <yaml>` merges entries over the built-ins. See
`clusters.example.yaml`, where every field carries the failure it prevents, and
`clusters.dccs.yaml` for a filled-in gfx942 entry. Cluster facts (partition,
GRES, interface, node exclusions, fabric env, run-as-user) live there so a
scenario stays portable.

```bash
python3 tools/infx2mad.py --infx "$INFX" --all \
    --cluster dccs-4081 --cluster-file tools/clusters.dccs.yaml --out-dir out942
```

## Fidelity is explicit, never implied

Every deviation from the recipe is recorded in the manifest's
`_infx_provenance.fidelity_gaps` and in the config header, together with the
InferenceX commit the artifacts were generated from. A translated row that
silently differs from the scenario is worse than no row. The `--smoke-model` and
`--smoke-first-point` options exist so a stand-in run can be made cheaply, and
they are loud for the same reason.

Four invariants the emitters hold, each forced by something concrete:

1. **One config block per (tp, concurrency) point.** The runners expand list
   args with `itertools.product`, a cartesian product, while InferenceX zips tp
   against its own concurrency range. Collapsing them invents points that were
   never measured.
2. **Per-point server args.** A recipe override carries args belonging to one
   concurrency (`cuda-graph-max-bs`, `max-running-requests`, the hicache group),
   which a cartesian expander cannot represent at all.
3. **Drop false-valued boolean args.** The runners render any bool as a bare
   flag, so a recipe's `enable-metrics: false` would be emitted as the flag and
   turn the option ON.
4. **A concurrency value is not a unique key.** dsv4's agentic recipe has two
   overrides at conc=128 differing only in `ep`; the scenario's search-space
   point disambiguates through `ep`, `dp-attn` and `kv-offloading`.

## Known limits

- **Asymmetric TP in disagg.** `scripts/sglang_disagg/run.sh` resolves a single
  `SGLANG_TP_SIZE` for both roles, so a point with prefill tp=4 and decode tp=8
  has no faithful mapping. Such points are dropped with the reason attached.
- **`xP`/`yD` are node counts** while a scenario states workers; they coincide
  only while tp equals the node's GPU count.
- **`atom-disagg` and `tilert`** have no MAD runner.
- **Agentic measurement window.** The recipes state none and `agentic_lib.sh`
  defaults to 120 s against a 900 s scenario minimum, so pass
  `--agentic-duration 900` for a submission-valid row.

## Helper

`tools/probe_sudo.sh <node>...` reports which nodes can actually run docker.
madengine reaches docker through passwordless sudo, and two independent things
go wrong per node: sssd/PAM breaks so sudo refuses, or dockerd is not running.
Both fail long before the model loads and look like a harness bug. Feed the
result into the cluster entry's `exclude_nodes`.
