# infx2mad — run InferenceX e2e scenarios through MAD

[InferenceX](https://github.com/SemiAnalysisAI/InferenceX) publishes the AMD
serving scenarios SemiAnalysis measures. `tools/infx2mad.py` turns one of those
scenarios into the artifacts madengine already runs, so we reproduce their
numbers with our harness instead of growing a second one.

## Why a translator and not a new harness

47 of the 52 AMD scenarios map onto a MAD runner that already exists:

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
# Pinned, not `--depth 1` off the default branch: upstream changes its
# scenario schema without notice (it has already moved recipes to
# srt-recipe-dir and dropped base.model). This is the commit the schema
# handling was last verified against; the older f85e1b9 layout also works.
git clone https://github.com/SemiAnalysisAI/InferenceX.git
git -C InferenceX checkout 47cdc9d02d2f50018efbbc207c0c3ce9e425dae4
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
manifest, and a `models.json` entry.

The manifest is the self-contained launch path: it carries the image tag, the
`BASE_DOCKER` build argument the scenario pins, the mounts and the SLURM
settings.

```bash
madengine run -m out/dsr1-fp8-mi355x-sglang.manifest.json -o perf.csv --live-output
```

The `models.json` entry is not self-contained. It has no build arguments --
`docker/pyt_infx.ubuntu.amd.Dockerfile` deliberately has no default
`BASE_DOCKER` -- and no image reference, so a plain `madengine run --tags`
needs an image built from the manifest beforehand and selected explicitly.

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
silently differs from the scenario is worse than no row. The `--smoke-model`,
`--smoke-first-point` and `--smoke-point` options exist so a stand-in run can be
made cheaply, and they are loud for the same reason.

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

## Routing TP AllReduce through RCCL (opt-in)

SGLang and vLLM serve tensor-parallel AllReduce from their own custom
all-reduce, so a TP serving point as InferenceX runs it puts almost no traffic
on RCCL. `--rccl-allreduce` adds `--disable-custom-all-reduce` to every point so
that AllReduce goes through RCCL instead. It is off by default, is recorded as a
fidelity gap (the row no longer measures what the scenario names). ATOM has
no such switch, so there the gap says the mode was not applied; disaggregated
scenarios take their server args from the launcher and refuse the option.

## GPU product

A scenario is refused on a different arch, but parts sharing an arch (MI350X
and MI355X on gfx950, MI300X and MI325X on gfx942) differ in power and clocks.
The cluster entry's `gpu_product` is compared with the scenario's part, and a
difference -- or an entry that states none -- is a fidelity gap: the numbers do
not compare with the published ones.

## Checkpoint precision and revision

The engine takes its quantization from the checkpoint, not from the scenario's
label, so the translator reads the mounted checkpoint and reports a gap when its
revision differs from the recipe's pin, when its precision differs from the
scenario's (judged by the MoE experts where present; an unquantized checkpoint
matches only if the recipe quantizes at load time), and when there is no
checkpoint on the cluster to read, in which case the precision is unverified.
A recipe `tokenizer-path` that points into InferenceX's own runner cache is
rebased onto the mounted checkpoint (or the repository id), with the swap
recorded; any other absolute tokenizer path drops the point.

## Node fit and offline draft models

The aggregated route is one node: a point spanning more GPUs than the cluster
entry's `gpus_per_node` (tp, times data-parallel-size where the engine spells
DP that way) is dropped with the reason. A recipe that runs offline
(`HF_HUB_OFFLINE=1`) and names a speculative draft checkpoint by hub id relies
on it being in InferenceX's runner cache; here the draft is looked up in the
cluster's model stores, mounted, and the argument pointed at the mount (the
swap is recorded, its revision is not checked). A point whose draft is in no
store is dropped rather than started to fail.

## Known limits

- **Asymmetric TP in disagg.** `scripts/sglang_disagg/run.sh` resolves a single
  `SGLANG_TP_SIZE` for both roles, so a point with prefill tp=4 and decode tp=8
  has no faithful mapping. Such points are dropped with the reason attached.
- **`xP`/`yD` are node counts** while a scenario states workers; they coincide
  only while tp equals the node's GPU count.
- **`atom-disagg` and `tilert`** have no MAD runner.
- **Recipe `setup_script`.** A recipe may prepare its container before the
  engine starts. `pip-runtime-deps.sh` is reproducible — it installs
  `SETUP_PIP_PACKAGES` and nothing else — so it is folded into the image build
  and the packages become part of the image tag's fingerprint. Any other
  setup_script patches the container in ways the wrapper cannot restate, so the
  scenario (or the one override that declares it) is dropped with the reason
  attached rather than run unprepared.
- **Pinned engine images expire.** A scenario pins the image its numbers were
  measured against, and the vLLM nightly tags are garbage-collected upstream, so
  an older scenario can become unbuildable. `--base-image` substitutes one and
  records the substitution as a fidelity gap; the tag fingerprint changes with
  it, so the run cannot reuse an image built from the pinned one.
- **Agentic measurement window.** The recipes state none and `agentic_lib.sh`
  defaults to 120 s against a 900 s scenario minimum, so pass
  `--agentic-duration 900` for a submission-valid row.

## Helper

`tools/probe_sudo.sh <node>...` reports which nodes can actually run docker.
madengine reaches docker through passwordless sudo, and two independent things
go wrong per node: sssd/PAM breaks so sudo refuses, or dockerd is not running.
Both fail long before the model loads and look like a harness bug. Feed the
result into the cluster entry's `exclude_nodes`.
