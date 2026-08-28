# Gotchas — cross-cutting and per-workload

Keywords: source mad.env MODEL_DIR run-script, MAD_DOCKER_BUILDS shared storage,
MAD_SECRETS_HFTOKEN HF 401 single-quoted, docker_mounts container_path host_path -v,
run_logs shared NFS mount, SLURM_SUBMIT_DIR sbatch cwd rundir, WORKDIR not
exported shlex.quote docker_mounts additional_docker_run_options,
RCCL_AINIC_ROCE RDMAV_DRIVERS ionic,
NCCL_IB_HCA mlx5 rdma per-cluster, perf.csv login-node aggregation,
slurm.nodes distributed.nnodes nodelist world size,
heterogeneous nodes NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME network_interface,
routable interface eth0 eth1 IPv6 link-local fe80 gloo connect timeout subnet,
madengine --timeout 0 None sbatch template, -o output csv ignored classic slurm,
docker commit ENTRYPOINT cat exit 126, NFS root_squash docker mount permission denied

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
- **Never pass `--timeout 0` to `madengine run`.** `cli/commands/run.py` maps `0`
  to Python `None` and the SLURM job template renders it literally
  (`{{ timeout | default(3600) }}` does not fire for `None`), so the in-job
  `madengine run --timeout None` dies on argument parsing with exit code 2 —
  after the allocation is already up. Always pass a positive number.
- **`-o <file>` does not move the aggregation in the classic SLURM path.**
  Results still land in cwd `perf.csv` / `perf_entry.csv`; only the reporter
  reads the `-o` name, so a run ends with a cosmetic
  `⚠️ Performance CSV not found: <file>` (`cli/utils.py`) while every metric was
  in fact stored. Read `perf.csv` and treat that warning as noise, or leave `-o`
  at its default. (The equivalent warning was fixed for `slurm_multi`, not for
  the classic path.)
- **A hand-built image handed to madengine must have no `ENTRYPOINT`.**
  madengine keeps the container alive with `docker run -t -d <img> cat`, so a
  baked `ENTRYPOINT ["bash"]` turns that into `bash cat` → exit 126 → the next
  `docker exec ... whoami` fails with "container is not running". `docker commit`
  of a `--entrypoint bash ... sleep infinity` debug container bakes exactly that,
  and `docker commit --change 'ENTRYPOINT []'` does **not** clear it. Clear it
  with a one-line overlay build
  (`printf 'FROM <committed>\nENTRYPOINT []\nCMD []\n' | docker build -t <tag> -`)
  and verify `docker image inspect <img> --format '{{json .Config.Entrypoint}}'`
  returns `null`. Dockerfile-built images are unaffected.
- **Docker cannot mount a host dir that squashed root cannot traverse.** The
  docker daemon runs as root and NFS `root_squash` maps root to `nobody`, so a
  data directory that is only group-readable (e.g. mode `drwxrws---` owned by a
  developer group) fails every `-v` with
  `error while creating mount source path ...: mkdir ...: permission denied`,
  even though the user's own shell reads it fine. Stage run data under a path
  whose whole chain is world-traversable (`o+x`), not just group-readable.
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
DT_NEEDED smifix, single full-overlay Dockerfile no base chaining, RDMA_CORE_VERSION
rdma-core 63.0 bnxt_re Thor2 stage, mooncake baked launcher build-layer removed runtime pip,
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
  rdma-core is a stage in the SAME Dockerfile, controlled by
  `--build-arg RDMA_CORE_VERSION` (default `63.0`, built from source — it is what
  makes queue-pair creation work on Broadcom Thor2 `bnxt_re`; no host `libibverbs`
  bind-mounts needed). Pass an empty `RDMA_CORE_VERSION=` to keep the base image's
  rdma-core and skip the stage. There is no separate rdma-core-variant Dockerfile anymore — it was a
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
  `scripts/primus_megatron-lm/primus_megatron-lm_benchmark_report.sh`
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
  `scripts/primus_megatron-lm/primus_megatron-lm_benchmark_report.sh` this is
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

Keywords: Kimi-K2-Thinking 384 experts MoE, rdzv timeout DistStoreError MIOpen,
PRIMUS_SANITY_TRAIN_ITERS proxy 3-layer, log_error_patterns torchrun SIGSEGV Traceback,
moe_router_num_groups null no group routing.

- **Kimi-K2-Thinking needs >= 24 nodes (192 GPUs, EP=192). Smaller is
  arithmetically impossible, and the failure is invisible.** At 1T total params
  the per-GPU budget is weights (bf16) + grads (fp32, `grad_reduce_in_fp32`) +
  Adam state (12 B/param). Expert optimizer state is only sharded across
  expert-DP = `world_size / EP`, so when `EP == world_size` it is not sharded at
  all. At the original EP=8 / 2-node shape that totals ~1126 GiB/GPU against
  252 GiB usable on MI355X. Measured working point: 24 nodes, `PRIMUS_EP=192`,
  `micro_batch_size=1`, full recompute -> ~188 GiB resident, ~236 tok/s/GPU
  steady state (61 layers, seq 4096).
  Three separate walls hit in order, each masking the next: DDP `param_data`
  allocation, then forward activations (fixed by recompute), then the MoE
  permutation buffers in backward (fixed by MBS=1).

- **Set `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` for this model.** With
  the previous `garbage_collection_threshold:0.8`, once occupancy approaches the
  ceiling the caching allocator re-runs a full GC on *every* allocation and
  spins for many minutes instead of raising `OutOfMemoryError`. That looks
  exactly like a collective deadlock -- GPUs pinned at 100%, an unchanging
  `py-spy` frame, a frozen log -- and it cost a long detour blaming RCCL and the
  MoE all-to-all. The give-away is that the stuck frame is a plain
  `torch.empty(...)`, not a kernel launch or a collective.

- **A "silent" worker death here is almost never silent -- it is a hidden
  `OutOfMemoryError`.** `primus/cli/main.py` wraps every worker exception in
  `except Exception: ... raise SystemExit(1)`; CPython deliberately prints no
  traceback for `SystemExit`; and torchrun's `--local-ranks-filter 0` hides the
  stderr of whichever rank actually failed. Net effect: `exitcode 1`,
  `error_file: <N/A>`, no traceback anywhere. To recover the real exception
  without rebuilding the image, inject a `sitecustomize.py` via `PYTHONPATH`
  that hooks `sys.exit` and PEP 669 `sys.monitoring` RAISE, and dump
  `SystemExit.__context__` to a per-rank file (a per-rank file also dodges the
  rank filter).

- **Kimi-K2-Thinking is a mock-data throughput benchmark, not a training run.**
  `docker/kimi_k2_configs/kimi_k2_thinking-BF16-pretrain.yaml` sets `mock_data:
  true`, `train_iters: 50` and `disable_last_saving: true` unconditionally. The
  architecture is the real 61-layer / 384-expert one, so the tok/s and TFLOP/s
  figures are representative, but no dataset is read and no checkpoint is
  written. Do not quote it as a 1T-parameter pretraining result.

- **The two Kimi YAMLs are injected by the Dockerfile, not shipped in the base
  image.** Unlike every other Primus model here,
  `kimi_k2_thinking.yaml` and `kimi_k2_thinking-BF16-pretrain.yaml` live in this
  repo under `docker/kimi_k2_configs/` and are copied into the Primus tree by
  `docker/primus_megatron_train_rccl_overlay.ubuntu.amd.Dockerfile`. That `COPY`
  is relative to the build context, so the manifest must keep
  `"dockercontext": "./docker"` — without it madengine gives any dockerfile whose
  path contains `primus` a repo-root context and the build fails at the `COPY`.

- **`context.docker_env_vars` reaches every deployment, not just a local
  build-time sanity pass — `PRIMUS_SANITY_TRAIN_ITERS` must not live there by
  default.** madengine renders `context.docker_env_vars` as `--env` on the
  actual `docker run`/`torchrun` invocation on every node, SLURM-deployed runs
  included (verified directly: grepping a real 2-node SLURM job's container
  launch line shows `PRIMUS_SANITY_TRAIN_ITERS=50` passed to the container even
  though the manifest's `deployment_config.env_vars` never set it). An earlier
  version of this template shipped `PRIMUS_SANITY_TRAIN_ITERS`/
  `PRIMUS_SANITY_DATATYPE` unconditionally in `context.docker_env_vars`, on the
  mistaken assumption that block was build-time-only — the effect was that
  *every* deployment of the template, including the "real" SLURM run, silently
  took the 3-layer proxy path and never measured full depth. Fixed: the
  shipped template no longer sets either key. Add `PRIMUS_SANITY_TRAIN_ITERS`
  to `context.docker_env_vars` yourself only when you deliberately want the
  fast 3-layer CI/build-sanity path (`--num_layers 3 --moe_layer_freq 1` in
  `benchmark_report.sh`, ~30 min on 2×MI355X); leave it absent for a real
  measurement — at full depth, `torch.compile` workers can saturate every CPU
  during NCCL bootstrap and the rendezvous can time out (`DistStoreError:
  Timed out`) before the first training step without the rdzv fix below.

- **The torchrun rdzv timeout must be 7200s.** The 600s default is too short for
  MIOpen kernel compilation on a first run with 384 experts, and the job dies at
  `DistStoreError` during NCCL init — this looks like a network fault but is not.
  The `rccl_overlay` Dockerfile patches `run_pretrain.sh` and
  `primus-cli-direct.sh` to `--rdzv-conf timeout=7200`. Rebuild without that
  patch and the failure comes back.

- **`log_error_patterns` must be overridden.** torchrun prints `Traceback (most
  recent call last)` to stdout when a worker is SIGTERMed at cleanup, even after
  all 50 iterations complete cleanly. madengine's default error scan treats any
  `Traceback` as a failure and returns exit code 3, discarding a perfectly valid
  perf CSV. The template's `log_error_patterns` list drops `Traceback`,
  `RuntimeError:`, `Exception:`, `FAILED` and `failed (exitcode:` — keep it.

- **Architecture difference from DeepSeek-V3 that shows up in manifests:** Kimi K2
  uses `n_group=1`, so `moe_router_num_groups` is `null` (no group routing), and
  there are no MTP layers. Both are handled inside `kimi_k2_thinking.yaml`; no
  env overrides are needed.

## pyt_mlperf_training (MLPerf Llama-3.1)

Keywords: mlperf training llama-3.1-8b nemo megatron-core version pairing,
requirements/manifest.json mcore pin, no_weight_decay_cond get_megatron_optimizer,
dist_checkpointing strategies tensorstore ImportError, nemo_toolkit 2.7.3 py3.12
py3.13 nvidia-modelopt numpy<2, nemo.collections.llm missing in NeMo 3.0,
MLPERF_EXECUTION_MODE torchrun_in_alloc, NVTE_CK_JIT=0 clang++ -v undefined symbol
main, nvidia-modelopt filter veto, preprocessed C4 85 GB mlc-r2-downloader,
MLPERF_PEAK_BF16_TFLOPS mfu_pct, run_stop aborted NOT submittable, extract_perf.py.

- **NeMo and megatron-core are a matched pair — take the version from NeMo, not
  from "latest".** This is the expensive failure of this workload: the run
  reaches `torchrun`, then every rank dies with
  `TypeError: get_megatron_optimizer() got an unexpected keyword argument
  'no_weight_decay_cond'` and `ImportError: cannot import name 'tensorstore' from
  'megatron.core.dist_checkpointing.strategies'`. Both are version skew, not a
  ROCm or transport problem, so no amount of base-image swapping fixes them. The
  authoritative pin is in the NeMo tree itself:
  `requirements/manifest.json` -> `vcs-dependencies["megatron-lm"].ref`, whose
  `megatron/core/package_info.py` gives the version (NeMo 2.7.3 -> mcore
  0.15.0rc8, hence `ROCm/Megatron-LM` branch `core_r0.15.0_rocm`). Check that
  before changing `MEGATRON_COMMIT` in the Dockerfile.
- **megatron-core is installed editable, so a version swap costs no
  TransformerEngine rebuild.** Inside a built image:
  `cd /workspace/deps/megatron_lm && git checkout <branch> &&
  pip install --no-build-isolation --no-deps -e . &&
  (cd megatron/core/datasets && make)`. Confirm the two API points before
  spending an allocation on a real run:
  `'no_weight_decay_cond' in inspect.signature(get_megatron_optimizer).parameters`
  and `from megatron.core.dist_checkpointing.strategies import tensorstore`.
- **The base image's Python version decides whether the NeMo stack is
  installable at all.** On a py3.13 ROCm base, `nemo_toolkit[nlp]==2.7.3` cannot
  resolve: `nvidia-modelopt==0.37.0` declares `requires_python <3.13`, and NeMo's
  `numpy<2` (its `tensorstore` pin is compiled against the numpy 1.x ABI) has no
  py3.13 candidate. `nemo_toolkit==3.0.0` does install on py3.13 but ships no
  `nemo.collections.llm` / `nemo.lightning`, which is exactly the API the MLPerf
  reference and the entrypoint are written against. So the template pins a
  **py3.12** ROCm base; numpy 1.26.4 works fine with torch 2.10/2.12 there.
- **`MLPERF_EXECUTION_MODE=torchrun_in_alloc` is the multi-node mode.**
  NeMo-Run's own executors either assume a single node or want to submit their
  own sbatch job, which collides with the allocation madengine already holds, so
  `scripts/pyt_mlperf_training/mlperf_pretrain_entrypoint.py` builds the upstream
  Fiddle recipe and runs it under `torchrun` in the existing allocation.
- **Data is `run.sh`'s job, but it must land on shared FS and be mounted.** The
  preprocessed C4 for the 8B benchmark is ~85 GB in 6 files
  (`c4-train.en_6_text_document.bin` alone is 84 GB) pulled by plain `wget` under
  the MLCommons downloader — budget over an hour and expect to resume a stall by
  hand. The MLCommons *tokenizer* bundle, in contrast, is a full HF repo mirror
  (~32 GB of safetensors) — fetch only `tokenizer.json`,
  `tokenizer_config.json`, `special_tokens_map.json`, `config.json`,
  `generation_config.json`. `TMP_NPY_INDEX` must be shared as well: the Megatron
  data index is built once on rank 0 and read by every rank.
- **`mfu_pct` is meaningless without `MLPERF_PEAK_BF16_TFLOPS`.**
  `extract_perf.py` falls back to the MI325X peak (1307 TFLOP/s), which overstates
  MFU by ~2x on MI355X (dense BF16 peak 2500 TFLOP/s). `run.sh` forwards the env
  var to `--peak-bf16-tflops`; every other metric (step time, tokens/s, TFLOP/s)
  is hardware-independent.
- **`run_stop.status = aborted [NOT submittable]` is expected for a perf run.**
  The run stops at `MLPERF_MAX_STEPS` rather than at the benchmark's convergence
  target, so the headline `run_start -> run_stop` time is a throughput
  measurement, not a submittable MLPerf result.
- **Only when rebuilding the image:** `NVTE_CK_JIT=0` is load-bearing (the CK-JIT
  path probes its compiler with `<compiler> -v` and demands exit 0, but ROCm's
  `clang++.cfg` carries a linker flag that makes a bare `clang++ -v` attempt a
  link and fail with `undefined symbol: main`), `nvidia-modelopt` must be exempt
  from the requirement filter's `nvidia-` veto (`nemo.collections.llm.api`
  imports it unconditionally), and `nvidia-resiliency-ext` must be importable
  (`nemo/lightning/nemo_logger.py` dereferences it). Budget ~35 min for the
  build, ~30 of which is the TransformerEngine compile — that fits inside a
  4-hour interactive allocation but not much less. The image is ~55 GB, the
  `MAD_DOCKER_BUILDS` tar ~15 GB.

## pyt_mlperf_inference (MLPerf Llama-3.1 inference)

Keywords: mlperf inference llama3.1-8b, loadgen mlperf_loadgen, SUT_VLLM.py,
LLM.generate prompt_token_ids TypeError, TokensPrompt, Min duration satisfied NO,
Result is INVALID, Offline target_qps, user.conf min_duration 600000,
requirements.txt vllm==0.6.3 transformers==4.46.2, vllm serve ENTRYPOINT,
built_models.url git submodule update --init --recursive, MAD_OUTPUT_CSV $PWD,
np.int64 ast.literal_eval, cnn_eval.json sample_cnn_eval_5000.json, rouge1 target.

- **There is no AMD harness to wrap for this benchmark — use the reference.**
  AMD has never submitted llama3.1-8b: the public `rocm/amd-mlperf` inference
  tags cover llama2-70b, gpt-oss-120b, wan2.2, llama3.1-405b, mixtral and sdxl,
  and `inference_results_v5.1/closed/AMD/code` holds only llama2-70b-99(.9),
  mixtral-8x7b and stable-diffusion-xl. So the thin-wrapper pattern of
  MAD-internal's `pyt_mlperf_inf_mi355_llama2_70b_99` (3-line Dockerfile over a
  prebuilt `/lab-mlperf-inference` image) has nothing to wrap here. The upstream
  reference needs no ROCm-specific code changes — only the right base image.
- **Never install the harness `requirements.txt` on a ROCm vLLM image.** It pins
  `vllm==0.6.3` (a CUDA wheel that would replace the ROCm build the base image
  exists for) and `transformers==4.46.2` (incompatible with modern vLLM). Only
  three of its packages are actually missing from `vllm/vllm-openai-rocm`:
  `nltk`, `rouge-score`, `absl-py`. The image's `transformers` 5.x runs the
  harness fine.
- **The Offline SUT still uses a vLLM API that no longer exists.**
  `SUT_VLLM.py` calls `self.model.generate(prompt_token_ids=...)`, which modern
  vLLM rejects with `TypeError: LLM.generate() got an unexpected keyword argument
  'prompt_token_ids'` — after a clean model load, so it looks like a ROCm problem
  and is not. The Server path in the same file already builds `TokensPrompt`;
  the Dockerfile patches the Offline call site to match.
- **A healthy Offline run is INVALID until you declare `target_qps`.** Loadgen
  sizes its single coalesced Offline query from `target_qps`, left at 1 upstream,
  so the run ends long before `user.conf`'s `min_duration` (600 s) and reports
  `Result is : INVALID` / `Min duration satisfied : NO` next to a perfectly good
  throughput number. Duration is `target_qps * min_duration / actual_qps`, so
  `MLPERF_INF_OFFLINE_TARGET_QPS` must be **at or above** the achieved
  samples/s — 40 held for one 8x MI355X node at TP=8. Raising it also raises the
  reported throughput (a bigger query batches better): 33.7 -> 38.1 samples/s
  between two otherwise identical runs.
- **`built_models.url` must be empty.** madengine clones that url inside the
  container and then runs `git submodule update --init --recursive`, which fails
  on mlcommons/inference's unrelated bert / deepseek-r1 / wan-2.2 submodules and
  kills the run before the script starts. The harness is baked into the image.
- **Clear the base image `ENTRYPOINT`.** `vllm/vllm-openai-rocm` entrypoints to
  `vllm serve`, which would swallow madengine's `docker run -t -d <image> cat`
  keepalive.
- **Write the results CSV to `$PWD`.** madengine runs the script as
  `cd <model_dir> && bash run.sh` and then copies `<model_dir>/<multiple_results>`
  to the workspace root; a CSV written anywhere else is reported as "declares
  multiple_results but no such file was produced".
- **Data is small — this is not the training workload.** The checkpoint is ~15 GB
  of safetensors (skip `original/consolidated.00.pth`, a duplicate in Meta
  format) and the eval sets are 95 MiB (edge, 5000 samples) and 254 MiB
  (datacenter, 13368) via the MLCommons R2 downloader. Note the reference README
  pins checkpoint revision `be673f32...`, which no longer exists on the Hub —
  use `main`.
- **Do not parse `evaluation.py`'s dict with `ast.literal_eval`.** numpy 2 prints
  `gen_len` as `np.int64(3049739)`, which is a call node, not a literal.
- Accuracy is the cheap correctness signal: a 200-sample pass already lands
  within a few hundredths of the published BF16 ROUGE targets, so it validates
  the whole tract (checkpoint, tokenizer, loadgen, scoring) in minutes.
