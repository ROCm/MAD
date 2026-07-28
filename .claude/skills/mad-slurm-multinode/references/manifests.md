# Manifest anatomy and per-run adaptation

Keywords: run_manifest json, built_images local_image, docker_env_vars,
deployment_config slurm partition account qos reservation nodelist exclude,
distributed nnodes nproc_per_node launcher torchrun, multiple_results,
docker_mounts, ${VAR} expansion, MAD_SECRETS_HFTOKEN

A run manifest is a build+run description madengine executes with
`madengine run --manifest-file <file>`. The templates in `assets/manifests/`
are sanitized — fill them per run and write the result into `<WORKDIR>/rundir/`.
A filled, cluster-private manifest stays in the rundir, not back in the skill.

## Top-level keys

- `built_images` — the image(s) to run. `docker_image` is the local tag.
  `local_image: true` means the image is treated as locally provided: at run
  time madengine ensures it per node (load from `MAD_DOCKER_BUILDS` tar, else
  build from `dockerfile`, else `docker pull`) — see launch-and-results.md
  ("Where the image comes from"). `dockerfile` is the path used for that build.
- `built_models` — model/run metadata: `scripts` (run.sh path, resolved under
  `MODEL_DIR`), `args` (`--model_repo ...`), `multiple_results` (the per-model
  perf CSV name to look for), `additional_docker_run_options` (privileged,
  shm-size, device mounts, ulimits).
- `context` — `docker_env_vars` (env inside the container),
  `docker_mounts` (bind mounts, keyed `{container_path: host_path}`; madengine
  renders each entry as `-v <host_path>:<container_path>`), `docker_build_arg`,
  `docker_gpus`.
- `deployment_config` — `slurm` block, `distributed` block, `env_vars`
  (env on the SLURM job / host side).

## Verify assets resolve under MODEL_DIR

The `built_images.*.dockerfile` and `built_models.*.scripts` (the `run.sh`)
paths are relative to `$MODEL_DIR` — the cloned `MAD` checkout. After
cloning or switching its branch, both resolve there
(`[ -f "$MODEL_DIR/<dockerfile>" ] && [ -f "$MODEL_DIR/<scripts>" ]`). A missing
path stops the run with a report (a wrong branch, uninitialized submodules, or a
different layout) rather than a silent search; the next step is to ask the user
for the correct branch.

## Secrets (HF token)

The HF token comes from `mad.env`, not the manifest. `mad.env` exports
`MAD_SECRETS_HFTOKEN=$(cat ~/.huggingface/token)`, and madengine forwards every
`MAD_SECRETS_*` from the sourced host env into the container's env
(`core/context.py` reads `MAD_SECRETS*` from `os.environ`; host-env values take
priority over manifest entries). A manifest that also sets
`"MAD_SECRETS_HFTOKEN": "${MAD_SECRETS_HFTOKEN}"` triggers an HF 401: madengine
renders that entry single-quoted in `docker run`, so the container receives the
literal string instead of the token. The templates carry no token key, and
keeping it that way is the fix.

## Fill checklist (every run)

Cluster-private — request from the user, keep out of the skill:
- `deployment_config.slurm.partition`
- `deployment_config.slurm.account` (omit the key if the cluster has no accounts)
- `deployment_config.slurm.qos`
- `deployment_config.slurm.reservation` (AINIC reserved campaigns) — **NOTE**: madengine
  2.0.0.post53 silently drops this field from the generated sbatch script. After submit, fix
  with `scontrol update jobid=<JID> Reservation=<name>`. Alternatively, `export
  SBATCH_RESERVATION=<name>` in the shell before `madengine run`.
- `deployment_config.slurm.nodelist` and/or `exclude` (specific node names)

Cluster-shape — confirm on the node (see cluster-types.md):
- `NCCL_IB_HCA` (in BOTH env blocks) — the mlx5/rdma device list
- `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` / `slurm.network_interface` — the
  same iface value in all three places (`context.docker_env_vars`, `slurm`, and
  `deployment_config.env_vars`); a mismatch splits control-plane routing.
- `NCCL_IB_GID_INDEX`
- `RDMAV_DRIVERS` / `IBV_DRIVERS` (+ `RCCL_AINIC_ROCE=1` on AINIC, delete on CX7)
- `MAD_SYSTEM_GPU_ARCHITECTURE` / `BUILD_GPU_TARGETS`

Run-shape — the user's run choice:
- `slurm.nodes` equals `distributed.nnodes` equals the cardinality of
  `nodelist` (if used). A mismatch -> sbatch/torchrun disagree on world size.
- `slurm.time` walltime, `gpus_per_node`, `nproc_per_node`.
- `docker_image` tag (the RCCL/build tag you intend to test).

Paths — from `mad.env` via `${VAR}` expansion where possible
(`${HF_HOME}`, `${MAD_DATAHOME}`, ...), explicit `<FILL_...>` otherwise.
madengine expands `${VAR}` from the sourced host env, so keep mad.env and the
manifest consistent. Any workload-specific data paths are explicit
`<FILL_DATA_ROOT>/...` in the workload manifest itself, not exported
from mad.env (which stays cluster/cache-only).

## Placeholder convention in templates

JSON has no comments, so placeholders carry inline guidance:
- `"<FILL_... see references/cluster-types.md>"` — a transport/cluster value;
  replace the whole string with the archetype value from the matrix in
  [cluster-types.md](cluster-types.md). The templates are cluster-agnostic, so
  the per-archetype values live only in that matrix, not inline in the manifest.
- `"RCCL_AINIC_ROCE": "<FILL_RCCL_AINIC_ROCE ...>"` — set per archetype
  (AINIC: `"1"`); delete the key entirely on an archetype that does not need it.
- `${VAR}` — leave as-is; madengine expands from the sourced env.

Re-validate after editing with the static checker (GPU-free, includes JSON
validity plus the fill checklist above):
`bash "$SKILL_DIR/scripts/validate_manifest.sh" <manifest>.json` (source
`mad.env` first so it can also resolve the dockerfile/run.sh under `$MODEL_DIR`).

## docker_mounts direction

`docker_mounts` is keyed `{container_path: host_path}` and madengine renders
each entry as `-v <host_path>:<container_path>` (`execution/container_runner.py`
emits `-v {value}:{key}`; `orchestration/run_orchestrator.py` reads the dict as
`{container_path: host_path}`). The key is the path inside the container; the
value is the path on the host. Swapping the two sides points Docker at a
non-existent host path, which Docker then creates as an empty directory, and the
workload fails with a missing-path error (for example a `MODEL_PATH is missing`
error). When a host data path and its container mount point
differ, the host path belongs on the value side.

## Per-workload notes

- **Primus** (`primus_pyt_megatron_lm_train_llama-3.1-{8b,70b}_overlay`): uses the
  `rccl_overlay` Dockerfile; `launcher: torchrun`, `nproc_per_node: 8`.
  `multiple_results` = `perf_primus-megatron-Llama-3.1-{8B,70B}.csv`. 70B is
  heavier — give it more walltime / more nodes.
- **Primus GPT-OSS-120B** (`primus_pyt_megatron_lm_train_gpt-oss-120b_overlay`):
  same `rccl_overlay` Dockerfile/`torchrun` shape as the Llama-3.1 templates.
  Verified working at 2 nodes (TP1 x PP2 x VP2 x EP8 = 16 GPU parallelism) with
  `timeout: 14400` (4h) — the CLI default (7200s/2h) times out on this model;
  keep the explicit 14400. `multiple_results` =
  `perf_primus-megatron-GPT-OSS-120B.csv`.
- **sglang_disagg** (`sglang-disagg-deepseek-r1-overlay`): disaggregated
  prefill/decode serving of DeepSeek-R1 on SGLang.
  `launcher: sglang-disagg`, `scripts/sglang_disagg/run.sh` entrypoint,
  `nproc_per_node: 8`, typically 4 nodes (e.g. xP=2 prefill, yD=2 decode; the
  proxy/load-balancer rides rank 0). Extra knobs: `xP`, `yD`, `RUN_MORI` (MoE
  expert-parallel all-to-all via MoRI — keep `1` for multi-node), `DP_MODE`,
  `KV_TRANSFER_BACKEND` (`nixl` or `mooncake`), `GPUS_PER_NODE`, the `MORI_*`
  transport vars (mirror the `NCCL_IB_*` data-plane values), and the benchmark
  sweep (`BENCHMARK_COMBINATIONS`, `BENCHMARK_CONCURRENCY_LEVELS`,
  `BENCHMARK_FAIL_FAST`). `run.sh` self-discovers the rank-ordered node IP list
  in-container (SLURM nodelist, else a rank-0 TCP rendezvous on the forwarded
  `MASTER_ADDR`), so do NOT set `IPADDRS` / `--ipaddrs`: madengine does not
  forward `SGLANG_NODE_IPS`, and a stale hand-filled list breaks role addressing
  (tune `IP_SYNC_TIMEOUT`, default 1800s, for image-load skew instead). The image
  is a single full-overlay build
  (`docker/sglang_disagg_inference_full_overlay.ubuntu.amd.Dockerfile`): base
  SGLang + RCCL + MoRI + NIXL/Mooncake KV-transfer + Mooncake pip in one
  Dockerfile (no base-image chaining). That build bakes in rdma-core
  `RDMA_CORE_VERSION` (default `63.0`), which is what makes queue-pair creation
  work on Broadcom Thor2 `bnxt_re` and removes the need for host `libibverbs`
  bind-mounts; pass an empty `--build-arg RDMA_CORE_VERSION=` to keep the base
  image's instead. The full overlay
  re-adds the `rocm_smi` `DT_NEEDED`
  (`patchelf --add-needed librocm_smi64.so.<N>`) so a newer RCCL on the rocm720
  base does not break `import torch` (see [gotchas.md](gotchas.md#sglang_disagg)).
  Perf lands in `perf_sglang-disagg-DeepSeek-R1.csv`, collected on rank 0 only.
- **sglang_disagg GPT-OSS-120B** (`sglang-disagg-gpt-oss-120b-overlay`): same
  disaggregated prefill/decode shape as DeepSeek-R1, but TP-only
  (`RUN_MORI=0`, `DP_MODE=0` — this model's MoE fits on a single 8-GPU TP=8
  replica per prefill/decode node, no cross-node EP needed), `mooncake`
  KV-transfer. Verified working at 4 nodes (xP=2, yD=2) on gfx942 (MI300X)
  with `base_docker: lmsysorg/sglang:v0.5.14-rocm720-mi30x`. This exact
  combo (GPT-OSS native mxfp4 checkpoint + gfx942) needs **all three** of:
  1. the Dockerfile's `mxfp_supported()` gfx942 patch (already baked into
     `docker/sglang_disagg_inference_full_overlay.ubuntu.amd.Dockerfile`,
     no manifest action needed — just rebuild the image against it);
  2. `--moe-runner-backend triton` (already baked into
     `scripts/sglang_disagg/models.yaml`'s `GPT-OSS-120B.base_flags`, no
     manifest action needed);
  3. `"SGLANG_USE_AITER": "0"` in **both** `context.docker_env_vars` and
     `deployment_config.env_vars` (set in the template) — sglang's
     `Mxfp4MoEMethod.apply()` ignores the CLI `--moe-runner-backend` and
     always builds an aiter-specific quant-info object when
     `SGLANG_USE_AITER` is truthy, crashing the triton runner with
     `AttributeError: 'AiterMoeQuantInfo' object has no attribute
     'use_mxfp8'`. This does not disable `--attention-backend aiter`
     (attention backend selection is CLI-driven, independent of this env
     var).
  `multiple_results` = `perf_sglang-disagg-GPT-OSS-120B.csv`.

## Scaling a manifest (e.g. 2-node -> 4-node)

Change `slurm.nodes`, `distributed.nnodes`, and the `nodelist` cardinality
together. Everything else (env, mounts, image) stays the same. Real examples
exist at 2 and 4 nodes for primus and sglang_disagg.
