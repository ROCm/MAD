# vLLM expert-parallel serving over DeepEP V2 / MoRI

Single node, 8x MI350X (gfx950), `TP=1 / DP=8 / EP=8`. One script drives both
all-to-all backends so a comparison between them changes exactly one thing.

## Run

```bash
madengine run --tags vllm_deepep --live-output
```

or one arm at a time:

```bash
madengine run --tags pyt_vllm_deepep_v2_deepseek-v2-lite --live-output   # DeepEP V2
madengine run --tags pyt_vllm_mori_deepseek-v2-lite      --live-output   # MoRI
```

Not `--tags deepep` / `--tags mori_ep`: madengine selects every model card
carrying *any* requested tag (`tools/run_models.py:544`), and `deepep` is also
on the R1 entry -- so that tag alone would launch two unrelated jobs instead
of one comparison arm. Model-card names are unique; tags are not.

The image is not built here — supply it with `DOCKER_IMAGE_NAME`. Build it from
`docker/vllm_deepep_inference.ubuntu.amd.Dockerfile`; that file documents the
required build args (`BASE_IMAGE`, `DEEPEP_REPO`, `DEEPEP_COMMIT`,
`VLLM_WHEEL`).

`BASE_IMAGE` has no default — deliberately, see the Dockerfile header. It
needs a ROCm/Torch/HIP toolchain new enough to build RCCL's device-side
symmetric-memory API and DeepEP's own HIP code (gfx950/MI350X); the public
`rocm/vllm-dev` tag the sibling disagg image uses is not new enough. Pass it
the same way as the other build args, e.g. via `madengine build --additional-
context '{"docker_build_arg": {"BASE_IMAGE": "..."}}'` or by setting
`docker_build_arg` directly on the model card.

## Why both arms run AITER fused MoE

vLLM refuses MoRI without AITER (`Mori needs to be used with aiter fused_moe
for now`). Running the DeepEP arm on a different MoE implementation would make
the two rows incomparable — the difference would be the MoE kernel, not the
transport. So `VLLM_ROCM_USE_AITER_MOE=1` is set for both, unconditionally, in
the script rather than per-entry.

## Why the warm-up runs are discarded, and reported

AITER tunes GEMM shapes at run time behind a global lock, so an unwarmed run
measures the tuner and not the server: P99 TTFT is roughly 20 s on the first
pass against a few hundred ms once warm, and about 270 s on DeepSeek-R1.

`WARMUP_RUNS` (default 2) are executed and thrown away; `MEASURED_RUNS`
(default 3) are recorded, each as its own CSV row rather than pre-averaged, so
a run that has not converged is visible instead of hidden in a mean. The
discarded count is written into every row — a backend comparison that does not
state it cannot be interpreted.

## Knobs

These are read by the script from its environment. madengine's `env_vars`
model-card field is consumed **host-side** — it feeds image selection and, on
the self-managed-launcher path, the host environment — but it is not forwarded
into a container that madengine runs itself: the local path constructs
`Docker(...)` without `envVars`, so the only env that reaches the container is
what appears as an explicit `-e` on the `docker run` line. Hence every runtime
knob below lives in `additional_docker_run_options`, and `env_vars` carries
only `DOCKER_IMAGE_NAME`. Override by editing that string, or by exporting the
variable and running the script directly inside the container.

| variable | default | note |
| --- | --- | --- |
| `ALL2ALL_BACKEND` | `deepep_v2` | or `mori_high_throughput` |
| `MODEL_REPO` | `deepseek-ai/DeepSeek-V2-Lite` | `MODEL_PATH` overrides with a local path |
| `MODEL_REVISION` | pinned per card | see below; empty means the moving `main` |
| `TRUST_REMOTE_CODE` | `0` | see below |
| `WARMUP_RUNS` / `MEASURED_RUNS` | 2 / 3 | see above |
| `CONCURRENCY` / `ISL` / `OSL` | 16 / 256 / 128 | `ISL + OSL` must stay under `MAX_MODEL_LEN`, or every request fails and the results are a row of zeros |
| `GPU_MEMORY_UTILIZATION` | 0.70 | 0.80 for R1 |
| `EP_NIC_NAME` / `NCCL_SOCKET_IFNAME` | `bnxt_re0` / `fenic0` | cluster-specific |

`NCCL_CUMEM_ENABLE` defaults to 1 for both backends and can be overridden.
DeepEP V2 needs it — RCCL rejects symmetric memory, and `ncclDevCommCreate`,
without VMM. MoRI carries its own transport and was never shown to need it, and
on a stack where cuMem itself is broken (ROCm 7.14/7.15 userspace over a 7.0
host KFD, reported) setting it unconditionally takes the control arm down with
the subject, leaving a comparison with no baseline. Set `NCCL_CUMEM_ENABLE=0`
for the MoRI arm there.

## Reproducibility: revision pinning and remote code

All three shipped cards set `MODEL_REVISION` to the commit each was validated
against, not to a branch name, so re-running a card later serves the exact
weights and config this comparison was measured on rather than whatever
`main` has moved to since. Passing `MODEL_REVISION` on its own is not
sufficient, which is why the script does more than add `--revision` to `vllm
serve`: `vllm bench serve` loads its own tokenizer independently and has no
`--revision` flag at all, so a moved `main` would tokenize the benchmark's
prompts differently from what the pinned server understands, silently, with
each side considering its own tokenizer authoritative. The script instead
resolves `MODEL_REPO`@`MODEL_REVISION` to one local snapshot via
`huggingface_hub.snapshot_download` and points both commands at that path, so
there is only one revision instead of two that need to agree. `MODEL_PATH`
bypasses this (it is already a fixed local path); leaving `MODEL_REVISION`
empty falls back to the moving revision with a warning, for ad hoc runs where
reproducibility does not matter.

Neither DeepSeek-V2-Lite nor DeepSeek-R1 needs `TRUST_REMOTE_CODE=1` -- vLLM
has native architectures for both, and current `transformers` ships its own
config classes, so neither model's `auto_map` is ever consulted. It defaults
off, since executing a mutable HF repo's Python as root in this container
(host networking, elevated capabilities, device access) is not something to
enable unconditionally. When set, it is forwarded to both `vllm serve` and
`vllm bench serve` -- forwarding it to only the server would let a model that
genuinely needs it start successfully and then fail once the benchmark's
independent tokenizer load hits the same requirement.

## JIT cache mounts

Each card mounts its AITER JIT build cache, AITER's tuned-config directory,
and DeepEP's JIT cache under `/var/cache/mad/vllm_deepep/<card-name>/`,
matching the paths the Dockerfile documents
(`/opt/aiter/aiter/jit/build`, `/tmp/aiter_configs`, `/root/.deep_ep`).
Without these, every fresh container repeats AITER's globally-locked GEMM
tuning from cold -- the ~270 s R1 first pass this README's warm-up section
describes.

The mount path is keyed by card name, not by `DOCKER_IMAGE_NAME`: it separates
the DeepEP and MoRI arms (different AITER tuning shapes) but not two different
images pointed at the same card. If you swap in an incompatible image or
AITER build, clear that card's cache directory first -- a build cache from a
different AITER revision is a correctness risk, not just a wasted rebuild.

**DeepEP arm** keeps GIN on (`NCCL_GIN_TYPE=2`, `EP_GIN_QUEUE_DEPTH=0`). GIN
carries no payload inside one XGMI domain, but it is the path that matters for
scale-out, so the single-node run exercises it rather than a mode nobody
deploys. The buffer constructor still reserves queue pairs, so the container
needs the RDMA device: madengine passes `--device=/dev/kfd` and the render
nodes but nothing else, so `--device=/dev/infiniband` is supplied through
`additional_docker_run_options` on every entry, together with
`--ulimit memlock=-1` (registration pins memory) and `--shm-size 128g` (eight
data-parallel workers). Without the device the DeepEP arm fails at buffer
construction, not at benchmark time.

**MoRI arm** sets `MORI_DISABLE_TOPO=1`, which works around a null dereference
in `CollectAndSortCandidates`. It skips GPU/NIC affinity matching; for a
single-node run traffic stays on XGMI so the effect should be small, but that
is an expectation rather than a measurement, so the flag is recorded in the
CSV.

## Known limitations

On a `therock-compiler` ROCm 7.14/7.15 base, nothing starts at all until the `amd_smi` path is
taken out: torch on ROCm enumerates devices through `amdsmi`, that build's `amdsmi` returns an
empty processor-handle list, and so `torch.cuda.device_count()` is 0 and vLLM reports `Failed to
infer device type`. With the path absent, torch falls back to HIP and sees all 8. The path is
added twice — `run_vllm_deepep.sh` exports it, and the Dockerfile puts it in `ENV` — so clearing
it needs `-e PYTHONPATH=` on the run as well as an edit to the script. Measured on gfx950 only;
whether an older, working ROCm still needs the path is untested, which is why neither line has
been changed here. Both predate this workload's profiling support.

MoRI does not currently start on DeepSeek-R1 FP8 —
`HIP error 709: hipModuleLaunchKernel(EpDispatchIntraNodeKernel_fp8_ocp)`,
reproduced across sessions. Hence there is a `deepep_v2` entry for R1 and no
MoRI counterpart; adding one before that is fixed would only produce a failing
job.

## Output

Two files. `perf_vllm_deepep.csv` is the one registered as `multiple_results`.
MAD requires it to have at least three columns and to include `model`,
`performance` and `metric` (`tools/utils.py:1224-1236`); extra columns are
permitted but ignored, since `:1247-1251` reads only those three. Long-form,
three rows per measured run:

```
model,performance,metric
run1_throughput,1234.56,total_token_throughput_tok_s
run1_tpot,12.34,median_tpot_ms
run1_ttft,345.67,p99_ttft_ms
```

MAD prefixes each label with the model-card name, so the backend and the model
arrive in the ingested row without being repeated here.

Extra columns would therefore survive the check but never reach `perf.csv`,
which is the same as losing them. So the diagnostic columns go to
`perf_vllm_deepep_detail.csv`, which is deliberately *not* registered and so is
never parsed:

```
backend,model,model_revision,concurrency,isl,osl,num_prompts,run,total_tok_s,median_tpot_ms,p99_ttft_ms,warmup_discarded,mori_disable_topo
```

Its `model` field is the resolved model — the local path when `MODEL_PATH` is
set, not the repo id — since that is what was actually served.

**Leave `LOG_DIR` at its default.** It is tempting to point it at a bound directory so the server
log survives the run, and that silently loses the result: `perf_vllm_deepep.csv` is written under
`LOG_DIR`, and madengine collects it from a fixed path in the run directory. Measured as a
controlled pair, differing in nothing else — with `-e LOG_DIR=/run_logs` the CSV is absent and
madengine reports `Successful: 0`; without it the CSV is there and the job is `SUCCESS`. The run
looks healthy from inside either way, which is what makes it worth stating. Keeping the server log
needs a separate variable for that log alone, which this workload does not have yet.

## Profiling

Off by default. Both switches are environment variables:

| variable | default | what it turns on |
|---|---|---|
| `PROFILE_ENABLE` | `0` | RCCL debug logs (one file per process) and torch traces |
| `RDMA_COUNTERS` | follows `PROFILE_ENABLE` | adapter counters sampled from sysfs |
| `PROFILE_DIR` | `/run_logs/vllm_deepep` | the **root**; each invocation gets its own
  subdirectory `<backend>_<timestamp>_<host>_<pid>` beneath it |

**Bind a host directory onto `/run_logs` before turning either on.** The staged model directory is
deleted after the run unless `--keep-model-dir` is given, and an unbound path lives only in the
container's writable layer, so artifacts written to either are gone before anyone reads them. The
script says so at startup rather than letting the run report a profile it cannot produce.

The model cards bind `/sys/devices` read-only. `--device=/dev/infiniband` is not enough for the
counter channel: the entries under `/sys/class/infiniband` are symlinks into `/sys/devices`, so
without it the sampler reads dangling links and finds nothing.

`RDMA_COUNTERS=1` on its own is worth knowing about: reading sysfs perturbs nothing, so such a
run's throughput is still comparable with an unprofiled one. `PROFILE_ENABLE=1` is not — RCCL's
INFO logging and the torch profiler both cost throughput, so the numbers in
`perf_vllm_deepep.csv` have to come from a run without it.

More than one channel is collected because a backend that carries its expert all-to-all outside
RCCL — which is what both `deepep_v2` and `mori` do — is invisible to the RCCL log. The traces
are the only place its kernels are named and the adapter counters the only place its bytes are
counted, so either channel alone would miss exactly the thing being compared.

The sampler is `../sglang_disagg/rdma_counters.sh`, shared rather than copied: it reads sysfs and
knows nothing about either engine, so a second copy would only be a second place for the counter
schema to drift.

### Not done yet: the parsing side

Collection is wired; the `EngineSpec` that turns these artifacts into a report is **not written**.
Adding one is a single module under `.claude/skills/mad-profiling/scripts/collprof/engines/` plus
one line in its registry — nothing in `core/` changes — and the checklist is in
`references/engines.md`.

It is deliberately not written yet. That checklist opens with "read one artifact of the engine by
hand first", and this workload targets gfx950; there is no MI350X in the cluster this was prepared
on, so no vLLM run exists here to read. A spec whose regexes were guessed from memory would
produce a report that parses cleanly and states things that are not true, which is worse than no
report. Write it against the artifacts of the first real run.
