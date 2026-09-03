# vLLM expert-parallel serving over DeepEP V2 / MoRI

Single node, 8x MI350X (gfx950), `TP=1 / DP=8 / EP=8`. One script drives both
all-to-all backends so a comparison between them changes exactly one thing.

## Run

```bash
madengine run --tags vllm_deepep --live-output
```

or one arm at a time:

```bash
madengine run --tags deepep      --live-output   # DeepEP V2
madengine run --tags mori_ep     --live-output   # MoRI
```

The image is not built here — supply it with `DOCKER_IMAGE_NAME`. Build it from
`docker/vllm_deepep_inference.ubuntu.amd.Dockerfile`; that file documents the
required build args (`DEEPEP_REPO`, `DEEPEP_COMMIT`, `VLLM_WHEEL`).

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
| `WARMUP_RUNS` / `MEASURED_RUNS` | 2 / 3 | see above |
| `CONCURRENCY` / `ISL` / `OSL` | 16 / 256 / 128 | `ISL + OSL` must stay under `MAX_MODEL_LEN`, or every request fails and the results are a row of zeros |
| `GPU_MEMORY_UTILIZATION` | 0.70 | 0.80 for R1 |
| `EP_NIC_NAME` / `NCCL_SOCKET_IFNAME` | `bnxt_re0` / `fenic0` | cluster-specific |

`NCCL_CUMEM_ENABLE=1` is set for both backends and is not optional: RCCL
rejects symmetric memory without VMM.

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

## Known limitation

MoRI does not currently start on DeepSeek-R1 FP8 —
`HIP error 709: hipModuleLaunchKernel(EpDispatchIntraNodeKernel_fp8_ocp)`,
reproduced across sessions. Hence there is a `deepep_v2` entry for R1 and no
MoRI counterpart; adding one before that is fixed would only produce a failing
job.

## Output

Two files. `perf_vllm_deepep.csv` is the one registered as `multiple_results`,
and MAD dictates its shape: `tools/utils.py:1231` raises unless the columns are
exactly `model` / `performance` / `metric`, and `:1247-1250` reads only those
three and discards the rest. So it is long-form, three rows per measured run:

```
model,performance,metric
run1_throughput,1234.56,total_token_throughput_tok_s
run1_tpot,12.34,median_tpot_ms
run1_ttft,345.67,p99_ttft_ms
```

MAD prefixes each label with the model-card name, so the backend and the model
arrive in the ingested row without being repeated here.

Because the ingested file cannot carry context, the diagnostic columns go to
`perf_vllm_deepep_detail.csv`, which is deliberately *not* registered and so is
never parsed:

```
backend,model,concurrency,isl,osl,run,total_tok_s,median_tpot_ms,p99_ttft_ms,warmup_discarded,mori_disable_topo
```

Its `model` field is the resolved model — the local path when `MODEL_PATH` is
set, not the repo id — since that is what was actually served.
