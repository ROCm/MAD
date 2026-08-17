# Manifest overlays — turning a working run into a measurable one

A profiled run is the mad-slurm-multinode manifest plus the additions in this directory. They are
kept as overlays rather than as whole manifests on purpose: the cluster-specific half of a manifest
(HCA list, GID index, interfaces, partition, image) belongs to that skill and changes per cluster,
while the profiling half is the same everywhere and is what this skill owns.

| file | for |
|---|---|
| `primus-megatron.overlay.json` | Primus/Megatron-LM pretraining |
| `primus-exp-yaml.profiler.yaml` | the Primus experiment YAML, where the torch profiler is switched on |
| `sglang-disagg.overlay.json` | sglang PD-disaggregated serving |

## Applying one

```bash
# base.json is the filled manifest from mad-slurm-multinode
jq -s '.[0] * .[1]' base.json primus-megatron.overlay.json > run_manifest_prof.json
grep -n 'FILL_' run_manifest_prof.json          # must print nothing
bash "$MULTINODE_SKILL_DIR/scripts/validate_manifest.sh" run_manifest_prof.json
```

`jq '.[0] * .[1]'` merges objects recursively, so the env blocks gain keys rather than replacing
them, and arrays are taken from the overlay, which is what `tools` wants.

The overlays keep their prose in `_comment_*` keys placed *beside* a block, never inside one. Inside
is not a comment but data: madengine renders every key of `docker_mounts` as a `-v` argument and
every key of an env block as a variable, so a `_comment` in there becomes a bogus mount. Leaving the
`_comment_*` keys in the merged manifest is fine — the manifests that produced the reference reports
carry them.

**One edit a merge cannot make.** For sglang the traces and per-role logs land under `/run_logs`,
which needs a shared-FS bind appended to a string the base manifest already has, in
`built_models.<model>.additional_docker_run_options`:

```
-v ${SLURM_SUBMIT_DIR}/slurm_output/run_logs:/run_logs -e SLURM_JOB_ID
```

It goes there rather than in `docker_mounts` because madengine passes `docker_mounts` values through
`shlex.quote()`, which blocks `${SLURM_SUBMIT_DIR}` from ever expanding.

## What each addition buys, and what is lost without it

| addition | report gains | absent |
|---|---|---|
| `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS` incl. `COLL` | every collective of the whole run: sizes, counts, datatypes, per-rank and per-node volume, the rank matrix | there is no report at all |
| `context.tools[rocprofv3_communication]` | host-side enqueue time per collective, device kernel time, compute-vs-communication split | report is volume-only and says so |
| torch profiler (`primus-exp-yaml.profiler.yaml`, or `SGLANG_TORCH_PROFILER_DIR` + `PROFILE_*`) | per-collective message size, dtype and process group | the size/process-group section |
| `gpu_info_power_profiler`, `gpu_info_vram_profiler` | power and VRAM context for the run, outside the collective report | nothing in the report; cheap, so kept on |

Two of the three data channels missing still makes a usable report; the RCCL log missing does not.
Details of what each channel can and cannot support: [../../references/interpretation.md](../../references/interpretation.md).

## The three traps these overlays exist to avoid

**Both env blocks or neither.** `context.docker_env_vars` reaches the container;
`deployment_config.env_vars` stops at the SLURM launcher. Every profiling variable is in both blocks
of both overlays for that reason. mad-slurm-multinode's `scripts/validate_manifest.sh` checks the
asymmetry.

**The base templates log the wrong subsystem.** They ship `NCCL_DEBUG_SUBSYS=INIT,NET` (sglang:
`NCCL_DEBUG=WARN`), which records communicator setup and no collectives. The run succeeds, the
report comes out nearly empty, and nothing fails.

**Space.** `NCCL_DEBUG=INFO` writes a line per collective: about 2 GB per decode node per run, and a
four-node two-role job fills 4 GB in minutes. This has hit 100% on a shared home directory, which
truncated a file mid-write and marked a finished job as failed at teardown. Plan the space, then
gzip the logs — they compress about 24x and every parser here reads `.gz` directly.

## The second run

Everything here costs performance, and two of the sglang flags change the workload outright
(`--disable-custom-all-reduce`, `--disable-cuda-graph`). Volumes come from the profiled run;
throughput comes from a second run with the overlay removed. A report quoting both from one job is
quoting a slowed-down configuration as if it were the tuned one — which is why the engine's scope
note says so in every report it prints.
