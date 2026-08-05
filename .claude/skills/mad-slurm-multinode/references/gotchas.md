# Gotchas — cross-cutting and per-workload

Keywords: source mad.env MODEL_DIR run-script, MAD_DOCKER_BUILDS shared storage,
MAD_SECRETS_HFTOKEN HF 401 single-quoted, docker_mounts container_path host_path -v,
run_logs shared NFS mount, SLURM_SUBMIT_DIR sbatch cwd rundir, WORKDIR not
exported shlex.quote docker_mounts additional_docker_run_options,
RCCL_AINIC_ROCE RDMAV_DRIVERS ionic,
NCCL_IB_HCA mlx5 rdma per-cluster, perf.csv login-node aggregation,
slurm.nodes distributed.nnodes nodelist world size,
heterogeneous nodes NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME network_interface,
routable interface eth0 eth1 IPv6 link-local fe80 gloo connect timeout subnet

Cross-cutting and per-workload pitfalls observed in real runs. SKILL.md links
here; this file is read before a run.

## Cross-cutting

- **`source mad.env` in the same shell before `madengine run`.** `MODEL_DIR` is
  exported there; without it the container's `scripts/` directory resolves empty
  and the run dies with a missing run-script error.
- **`MAD_DOCKER_BUILDS` lives on shared storage** visible to every compute
  node. The login/build node saves the image tar there; workers load it from
  the same path. A node-local path makes workers rebuild (or fail).
- **The HF token lives in `mad.env`, not in the manifest** — a stray
  `MAD_SECRETS_HFTOKEN` key in the manifest causes HF 401s (madengine renders it
  single-quoted, so the container gets the literal `${MAD_SECRETS_HFTOKEN}`).
  Full explanation in [references/manifests.md](manifests.md)
  ("Secrets (HF token)").
- **`docker_mounts` format is `{container_path: host_path}`.** madengine
  renders each entry as `-v <host_path>:<container_path>`. Swapping the two
  sides (a tempting mistake when the paths differ) hands Docker a non-existent
  host path, which it silently creates as an empty directory, and the workload
  fails with a "path does not exist" error. The direction is worth re-checking
  whenever a host data directory maps to a container-internal path that differs
  from the host path. See
  [references/manifests.md](manifests.md) ("docker_mounts direction").
- **Prefer `${SLURM_SUBMIT_DIR}` over `$WORKDIR`/relative paths in
  `additional_docker_run_options` host mounts — it is a real, always-set SLURM
  var that already equals the shared `rundir`.** `$WORKDIR` is a doc-only
  convention (nothing in mad.env/madengine actually exports it), and a
  relative host path (e.g. `./slurm_output/run_logs`) resolves against
  whatever CWD the per-node `madengine run` process happens to have. SLURM
  itself sets `SLURM_SUBMIT_DIR` to the directory `sbatch` was invoked from
  (`deployment/slurm.py` runs `sbatch` without a `cwd=` override, and Step 6
  always launches from `$WORKDIR/rundir`), and that value is inherited
  end-to-end — through the generated job script, `srun` (no `--export`
  restriction), and the final `docker run` (`additional_docker_run_options`
  is concatenated unquoted and `console.sh()` runs with `env=None`, i.e. full
  inherited environment) — so `${SLURM_SUBMIT_DIR}` reliably expands to
  `rundir` at every node with zero manual filling. This only works for
  `additional_docker_run_options`, NOT `docker_mounts`: madengine renders
  `docker_mounts` values through `shlex.quote()`, which single-quotes the
  whole string and blocks `$VAR`/`${VAR}` expansion outright — don't duplicate
  a shared-path mount there. The `sglang-disagg-deepseek-r1-overlay` template's
  `/run_logs` mount (`-v ${SLURM_SUBMIT_DIR}/slurm_output/run_logs:/run_logs`)
  is the reference example: `scripts/sglang_disagg/run.sh` requires `/run_logs`
  and fails fast (no `/tmp` fallback) if it is missing or not writable, but
  only checks writability, not actual cross-node sharedness — a node-local dir
  that happens to exist would pass that check and silently break the
  cross-node readiness rendezvous (see the `sglang_disagg` section below).
- **`NCCL_IB_HCA` is per-cluster, not portable.** CX7 uses `mlx5_*`; AINIC
  uses `rdma0..7`. Copying an mlx5 list onto an AINIC node (or vice versa)
  inits zero NICs. Verify on the node.
- **AINIC transport vars appear in BOTH manifest blocks.**
  `RCCL_AINIC_ROCE=1`, `RDMAV_DRIVERS=ionic`, `IBV_DRIVERS=ionic` go in
  `context.docker_env_vars` AND `deployment_config.env_vars`. Missing from
  either, RCCL falls back to verbs/sockets and the run measures the
  wrong transport.
- **Multi-node perf CSV: trust the login-node aggregation.** Throughput is
  often printed only on the last global rank, so node-0's local CSV can look
  empty even on a healthy run. Read the aggregated `perf.csv` in `rundir`,
  not a single node's file.
- **`slurm.nodes` equals `distributed.nnodes`** and matches `--nodelist`
  cardinality, or sbatch/torchrun disagree on world size.
- **Node environments can be heterogeneous across a cluster — don't trust a
  single detect probe.** The interface that carries the routable control-plane
  IP is not guaranteed to have the same name on every node (e.g. one node routes
  on `eth0` while another routes on `eth1`), and a given named interface may hold
  only a non-routable IPv6 link-local address (`fe80::...`) on some nodes. If you
  pin `NCCL_SOCKET_IFNAME`/`GLOO_SOCKET_IFNAME`/`network_interface` by name based
  on one probe node, the bootstrap can silently fail on the actually-allocated
  nodes. Typical symptom: torchrun rendezvous and `world_size` form fine (that
  goes over the hostname's routable IP), but `initializing torch distributed`
  hangs and gloo times out connecting peers over `fe80::` link-local addresses.
  Mitigation: verify the routable interface on the *allocated* nodes (not just
  the probe node), and prefer selecting the interface by its routable subnet
  rather than hard-coding an interface name. This is an environment
  (cluster-provisioning) inconsistency, not a workload bug — flag it to the
  cluster owner if a uniform-environment guarantee is expected.

## sglang_disagg

Keywords: launcher sglang-disagg run.sh xP yD RUN_MORI DP_MODE KV_TRANSFER_BACKEND
nixl mooncake, detokenizer hang health check No response from detokenizer,
MoRI overlay #366 inter-node decode freeze, RCCL overlay rsmi_init libtorch_hip
undefined symbol torch broken, rocm720 base librocm_smi64, patchelf add-needed
DT_NEEDED smifix, single full-overlay Dockerfile no base chaining, ENABLE_RDMA62
rdma-core v62 optional stage, mooncake baked launcher build-layer removed runtime pip,
self-discover node IPs rendezvous IP_SYNC_TIMEOUT SGLANG_NODE_IPS not forwarded,
per-node docker load shared tar, perf CSV rank0 BENCHMARK_FAIL_FAST, circuit
breaker prefill workers Service Unavailable BENCHMARK_POINT_RETRIES transient
sweep retry.

- **The launcher is `sglang-disagg` with `scripts/sglang_disagg/run.sh` as the
  entrypoint** (PR #142 native launcher). `run.sh` reads topology from
  `--xp/--yd` (or `xP`/`yD` env) and `--kv-transfer-backend` (`mori`/`mooncake`/
  `nixl`), then always execs the unified `sglang_disagg_mori_io_ep.sh` (it
  branches internally on `KV_TRANSFER_BACKEND`; the older Mooncake/NIXL-only
  `sglang_disagg_server.sh` was retired as a functional subset). There is no
  `slurm_multi` wrapper — one `madengine run` is launched per node and the
  roles (proxy / prefill x xP / decode x yD) are derived from `NODE_RANK` + the
  ordered IP list. Do not reintroduce a `*_mn` slurm wrapper; it duplicates
  topology logic and drifts from `run.sh`.

- **A newer RCCL build drops the `rocm_smi` `DT_NEEDED` and breaks
  `import torch` — re-add it with `patchelf`.** The RCCL stage of the single
  full-overlay Dockerfile
  (`docker/sglang_disagg_inference_full_overlay.ubuntu.amd.Dockerfile` — base
  SGLang + RCCL + MoRI + NIXL/Mooncake KV-transfer + Mooncake pip in one file, no
  base-image chaining) uses a rocm-systems RCCL (e.g. develop `78e8ba0`) on top of
  the rocm720 sglang base and prepends the overlay lib dir to `LD_LIBRARY_PATH`, so
  the overlay `librccl.so` resolves before the base one. That build links `librccl`
  *without* the `DT_NEEDED librocm_smi64.so` the base librccl carried, so
  `rocm_smi` is never transitively loaded and every later stage (and the run) dies
  on `import torch` with `libtorch_hip.so: undefined symbol: rsmi_init`. Fix
  (already baked into the Dockerfile): re-add the dependency on the overlay librccl
  during the RCCL stage —
  `patchelf --add-needed librocm_smi64.so.<N> "$(readlink -f .../librccl.so)"` —
  then sanity-check `python3 -c "import torch"` *with the overlay librccl resolved
  first* (the exact case that regressed). With the smifix the full overlay
  **base -> RCCL (+smifix) -> MoRI -> NIXL/Mooncake KV-transfer -> Mooncake pip**
  runs green (validated 4-node DeepSeek-R1, 56/56 sweep points). If you don't need
  a specific RCCL version, skip the RCCL stage and the base RCCL also works; if you
  keep it, apply the smifix. The Dockerfile sanity-checks `import torch` after each
  stage — a failure at the NIXL stage usually means an earlier stage (RCCL) is the
  real culprit, not NIXL.

- **Keep runtime installs out of the launcher — bake them into the image.** The
  launcher (`scripts/sglang_disagg/sglang_disagg_mori_io_ep.sh`) used to
  `pip install` py-spy/flask/pyyaml and build rdma-core v62 at job start; that
  mutated the container runtime env and silently corrupted the Python libs (the
  model flags then stopped parsing and the server came up with defaults). Those pip
  deps and the Mooncake KV backend are now baked into the full-overlay Dockerfile;
  rdma-core v62 is an optional stage in the SAME Dockerfile, gated by
  `--build-arg ENABLE_RDMA62=1` (for hosts whose RDMA stack needs a newer
  libibverbs/librdmacm/libmlx5 than the base ships — no host `libibverbs`
  bind-mounts needed; unset/default skips the stage entirely, mirroring
  `docker/primus_megatron_train_rccl_overlay...Dockerfile`'s `RDMA_CORE_VERSION`
  gate). There is no separate rdma-core-variant Dockerfile anymore — it was a
  ~92%-identical copy of the base overlay and was merged in to cut duplication.
  Do not reintroduce a runtime build-layer in the launcher.

- **The MoRI overlay (#366, e.g. commit `a14e6992`) is the one that fixes the
  mid-decode inter-node freeze.** (Commit hashes here and below — RCCL, MoRI,
  NIXL — are illustrative pins valid when written; they can drift after a squash,
  so track the PR/branch as the source of truth, not the SHA.) Without it, multi-node decode hangs partway
  through generation (observed ~token 31 during warmup): the MoE expert-parallel
  all-to-all stalls, the decode worker stops responding, and the proxy reports
  `No response from detokenizer` / the benchmark stream aborts
  (`ClientPayloadError` / `TransferEncodingError`). It is `RUN_MORI=1` +
  `MORI_RDMA_DEVICES`/`MORI_SOCKET_IFNAME`/`MORI_IB_GID_INDEX` that route this
  transport — set them alongside the NCCL/IB vars.

- **`No response from detokenizer` on idle decode ranks can be a false
  positive.** In a `yD>1` decode pool, idle decode ranks can trip the
  detokenizer health check even when generation is healthy (sglang upstream
  #20756). A *recent* sglang base fixes this; pin a base new enough to include it
  (v0.5.12.post1+ worked) rather than chasing it at the transport layer. A true
  hang (above) and this false positive look similar in the proxy log — confirm by
  checking whether the *active* decode rank is still emitting tokens
  (`py-spy dump` / `gstack` on the decode PID) before blaming the network.

- **`run.sh` self-discovers the rank-ordered node IPs in-container — do NOT set
  `--ipaddrs` / `IPADDRS`.** madengine's container runner does not forward
  `SGLANG_NODE_IPS`, so a manifest that relied on it fell back to
  `IPADDRS=localhost` and `DP_MODE=1` failed (roles dialed loopback and never
  connected). `run.sh` now derives the rank-ordered IP list from the forwarded
  `MASTER_ADDR`/`NODE_RANK`/`NNODES`: the SLURM nodelist when `scontrol`/`getent`
  exist, else a rank-0 TCP rendezvous on `MASTER_ADDR`. `NNODES` falls back to `xP+yD` (proxy/router is
  co-located on rank 0, no separate node). Leave `IPADDRS` / `--ipaddrs` unset in
  the manifest; only override them for a reproduction launched outside madengine.
  Raise `IP_SYNC_TIMEOUT` (default 1800s) if image-load skew delays peers.

- **Let madengine distribute the image — don't hand-roll per-node `docker
  load`.** For manifest-driven runs madengine already fans the image out on every
  node via `container_runner._ensure_local_image_available()` (inspect → `docker
  load` the `MAD_DOCKER_BUILDS/<tag>.tar` → build → pull); rank 0 `docker save`s
  the tar, a TCP barrier syncs, then workers load it (see `launch-and-results.md`
  and ROCm/madengine@d28f2f5). So with `local_image: true` + `MAD_DOCKER_BUILDS`
  on shared FS the ~20 GB overlay image is distributed for you — no manual
  `docker save`/`docker load` step is required. A manual per-node `docker load`
  is only relevant for the standalone holder/`srun` reproduction that launches the
  container *outside* madengine's dispatcher.

- **Perf CSV is produced only on rank 0; make the benchmark fail-fast.** Set
  `MAD_COLLECT_METRICS=true`/`MAD_SKIP_PERF_COLLECTION=false` on rank 0 only, and
  `BENCHMARK_FAIL_FAST=1` so a zero-result sweep aborts instead of writing an
  empty `perf_sglang-disagg-DeepSeek-R1.csv`. The sweep is
  `BENCHMARK_COMBINATIONS` (isl/osl) x `BENCHMARK_CONCURRENCY_LEVELS`; read the
  rank-0 aggregated CSV, not a non-zero node's local file.

- **A single transient gateway circuit-open should not nuke the whole sweep —
  retry points.** Occasionally a point fails at warmup with
  `Service Unavailable ... No available prefill workers (all circuits open or
  unhealthy)` even though the servers recover seconds later: the sgl-model-gateway
  circuit breaker opened on a transient prefill detok hiccup. Under
  `BENCHMARK_FAIL_FAST=1` that one blip otherwise aborts a ~50-minute run. The
  sweep retries each point `BENCHMARK_POINT_RETRIES` times (default 2) with a
  `BENCHMARK_RETRY_COOLDOWN_SECONDS` (default 45s) pause so the breaker can
  re-close; a point that fails *every* attempt still fails the run, so real
  regressions are not masked. Distinguish this transient from a true hard detok
  freeze (heartbeat frozen for minutes, never recovering) by checking
  `last_heartbeat` in `/run_logs/<job>/prefill_NODE*.log` /
  `decode_NODE*.log` — a hard freeze needs a real fix (e.g. the MoRI overlay),
  not a retry.

## primus_megatron (training)

Keywords: primus megatron scaleout MoE training, primus_turbo DeepEP token dispatcher,
use_turbo_deepep moe_enable_deepep flex alltoall, moe_shared_expert_overlap assert,
print_rank_last throughput last global rank, multi-node perf collection rank-0.

- **DeepEP is opt-in and OFF by default (dispatcher = `alltoall` over RCCL).**
  The baseline MoE token dispatcher is Megatron `alltoall` (a `torch.distributed`
  all-to-all over the EP group, carried by RCCL on the RoCE/IB fabric) — *not*
  MoRI. To switch to the Primus-Turbo **DeepEP** dispatcher, set
  `PRIMUS_USE_DEEPEP=1` in the manifest env (a primus scaleout manifest can carry
  it defaulting to `"0"`), which makes
  `scripts/primus/megatron-lm/primus_megatron-lm_benchmark_report.sh`
  (invoked by the model `run.sh`) pass `--use_turbo_deepep true` to Primus. Primus
  then (`_is_turbo_deepep_enabled`) auto-sets `moe_enable_deepep=True` and
  `moe_token_dispatcher_type='flex'` and swaps in
  `PrimusTurboDeepEPTokenDispatcher`. DeepEP lives inside `primus_turbo` (module
  `primus_turbo.pytorch.deep_ep`), not as a standalone `deep_ep` package.

- **DeepEP requires `tensor_model_parallel_size == 1` + `enable_primus_turbo`, and
  is incompatible with `moe_shared_expert_overlap`.** With DeepEP on, Primus ROCm
  `validate_args` asserts
  `AssertionError: DeepEP not support moe_shared_expert_overlap, please set
  moe_shared_expert_overlap=False` (DeepSeek-V3 enables shared-expert overlap by
  default). So enabling DeepEP must also pass `--moe_shared_expert_overlap false`
  (done automatically alongside `--use_turbo_deepep true`). Measured 4-node x
  8-GPU DeepSeek-V3 proxy config (MI300X, BF16, seq 4096): DeepEP ~295-299 TFLOP/s/GPU,
  ~13.1-13.3k tokens/s/GPU vs alltoall ~270 / ~12.05k (~+9%).

- **Throughput is printed on the global last rank (`print_rank_last`), not rank 0
  — read the aggregated perf, not node 0's local CSV.** On a multi-node run the
  Megatron/Primus training_log line (`TFLOP/s/GPU`, `tokens/s/GPU`) is emitted by
  the last global rank, which usually lands on the *last* node, while madengine's
  designated metric collector is rank 0 (node 0). madengine's login-node
  `collect_results` handles this (it picks the richest per-node `multiple_results`
  CSV via `_select_best_multiple_results_csv`, and treats an empty local perf on a
  `skip_perf_collection` SLURM node as SUCCESS). Use a madengine that has this
  multi-node aggregation (present in ROCm/madengine `develop`); otherwise the run
  trains fine but is mis-reported as FAILED with an empty `perf.csv`.

- **Per-model global batch size must be normalized to the world size, or Megatron
  aborts at startup on non-standard node counts.** The config `global_batch_size`
  is tuned for a single 8-GPU node; at any node count where
  `GBS % (MBS * world_size) != 0` (e.g. a 3-node/24-GPU run: `512 % (4*24) != 0`)
  Megatron aborts with `global batch size ... is not divisible by micro batch
  size ... times data parallel size`. In our consolidated
  `scripts/primus/megatron-lm/primus_megatron-lm_benchmark_report.sh` this is
  handled generically for *every* model branch: `scaleout_gbs_override` reads
  MBS/GBS from the run's config YAML and, when `NUM_GPUS > 8`, rounds GBS up to
  the next multiple of `MBS * NUM_GPUS` (via `normalize_global_batch_size`) and
  passes it as an explicit `--global_batch_size` override; single-node runs emit
  no override so behavior there is byte-for-byte unchanged. Any new per-model
  branch that hardcodes GBS should route it through `scaleout_gbs_override` to
  stay node-count-portable. (Upstream fix for the 8B branch:
  [mkuznet1/MAD#1](https://github.com/mkuznet1/MAD/pull/1) — our tree already
  generalizes it, so no per-branch change is needed here.)

- **`rocm/primus:v26.4` auto-loads `librccl-anp.so`, which deadlocks a bundled
  RCCL overlay — set `NCCL_NET_PLUGIN=none`.** The v26.4 base image ships an
  environment default of `NCCL_NET_PLUGIN=librccl-anp.so` (the ANP net plugin).
  When the image carries a *bundled* RCCL overlay (the whole point of the
  `rccl_overlay` Dockerfile), that plugin is incompatible with the overlay
  `librccl` and RCCL init hangs at the first collective — the run never starts and
  eventually times out. Set `NCCL_NET_PLUGIN=none` in the manifest env (both
  `context.docker_env_vars` and `deployment_config.env_vars`, like the other
  transport vars) to disable the plugin and let the bundled `librccl` drive the
  IB/RoCE net path directly. The primus template ships this key set to `none`.
