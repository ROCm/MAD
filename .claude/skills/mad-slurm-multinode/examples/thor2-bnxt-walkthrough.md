# Example walkthrough — Primus Llama-3.1-70B on a Broadcom-Thor2-RoCE cluster

> Sanitized end-to-end walkthrough: no real node names, SLURM queues, accounts,
> or tokens. Replace every `<...>` with your values.

Scenario: bring up a fresh Broadcom-Thor2-RoCE login node and run a 16-node Primus 70B perf
test over RoCEv2 on Broadcom Thor2 (`bnxt_re`) NICs.

## 0. Inputs gathered from the requester

| Input              | Example value                          |
|--------------------|----------------------------------------|
| Compute/login node | `<thor2-login-node>`                    |
| `$WORKDIR`         | `<WORKDIR>` (shared NFS)                 |
| Archetype          | Broadcom-Thor2-RoCE                     |
| Data root          | `<DATA_ROOT>` (shared NFS)               |
| `MAD_DOCKER_BUILDS`| `<SHARED>/mad_docker_builds` (shared NFS)|
| SLURM              | partition `<PART>` (no account/qos)     |
| Nodes              | 16 (8 GPU/node)                         |

## 1. Preflight + bootstrap

```bash
export SKILL_DIR="<absolute path to the mad-slurm-multinode skill dir>"
bash "$SKILL_DIR/scripts/preflight.sh"   # expect gfx950, docker ok, sbatch ok
cd <WORKDIR>
[ -d MAD ] || git clone <MAD_REPO_URL> --recursive
( cd MAD && git switch "<MAD_BRANCH>" && git submodule update --init --recursive )
[ -d madengine ] || git clone https://github.com/ROCm/madengine --recursive
( cd madengine && git switch develop && git submodule update --init --recursive )  # PR #142 merged
conda env list | grep -q '^madenv ' || conda create -y -n madenv python=3.12
conda activate madenv
pip install -e ./madengine
```

Confirm the data-plane HCAs and GID on an allocated node before filling the env:
```bash
srun -p <PART> -N1 bash "$SKILL_DIR/scripts/detect_cluster_env.sh"
# expect: gfx950, bnxt_re0..7, mgmt iface fenic0, RoCEv2 GID 3
```

## 2. rundir + mad.env (from mad.env.thor2-bnxt.template)

Filled values (illustrative):
```bash
export MAD_SYSTEM_GPU_ARCHITECTURE=gfx950
export MODEL_DIR=<WORKDIR>/MAD/
export MAD_DATAHOME=<DATA_ROOT>/models
export MAD_DOCKER_BUILDS=<SHARED>/mad_docker_builds
export NCCL_SOCKET_IFNAME=fenic0
export NCCL_IB_GID_INDEX=3
export RDMAV_DRIVERS=bnxt_re
export IBV_DRIVERS=bnxt_re
```
```bash
cd <WORKDIR> && mkdir -p rundir && cd rundir
cp "$SKILL_DIR/assets/mad.env/mad.env.thor2-bnxt.template" ./mad.env
# edit placeholders ...
source mad.env
[ -d "$MODEL_DIR/scripts" ] && echo "MODEL_DIR ok"
```

## 3. Manifest (from primus_llama-3.1-70b.template.json)

Edits applied to `run_manifest_primus_70b.json`:
- `slurm.partition=<PART>`, removed `account`/`qos` keys (cluster has none),
  `nodes=16`, and **added `"skip_gpus_directive": true`** (this cluster's SLURM rejects
  the `--gpus-per-node` directive — see references/cluster-types.md)
- `distributed.nnodes=16`
- `NCCL_IB_HCA=bnxt_re0:1,bnxt_re1:1,bnxt_re2:1,bnxt_re3:1,bnxt_re4:1,bnxt_re5:1,bnxt_re6:1,bnxt_re7:1`
  in BOTH `context.docker_env_vars` and `deployment_config.env_vars`
- `NCCL_SOCKET_IFNAME=fenic0`, `GLOO_SOCKET_IFNAME=fenic0`, `NCCL_IB_GID_INDEX=3`
- `RDMAV_DRIVERS=bnxt_re`, `IBV_DRIVERS=bnxt_re`, deleted the `RCCL_AINIC_ROCE` key
- added the bnxt_re tuning vars to both env blocks (QPS/inline/merge/adaptive/
  gdr/dmabuf — see references/cluster-types.md "Broadcom-Thor2 specifics")
- added `"/etc/libibverbs.d": "/etc/libibverbs.d"` to `context.docker_mounts`
- `docker_image=rocm/primus:v26.3-rccl-<branch>-<sha7>`, `base_docker=rocm/primus:v26.3`
```bash
bash "$SKILL_DIR/scripts/validate_manifest.sh" run_manifest_primus_70b.json
```

## 4. Launch + read result

```bash
source mad.env
madengine run --manifest-file run_manifest_primus_70b.json --live-output \
  -o perf_primus_70b.csv
squeue -u $USER          # watch the 16-node job (PR #142 auto-selects healthy nodes)
```
Result lands in `rundir/perf_primus_70b.csv` (aggregated) and
`perf_primus-megatron-Llama-3.1-70B.csv`. Report tok/s/gpu and TFLOPS/gpu from
the aggregated CSV (a worker node's local CSV can read header-only — see
references/launch-and-results.md).

## Notes

- PR #142 (merged in madengine `develop`) runs a GPU health check and
  auto-selects healthy nodes; leave `nodelist` unset to use it, or pin nodes by
  filling `slurm.nodelist` (cardinality must equal `nodes`).
- With `local_image: true` + `MAD_DOCKER_BUILDS` on shared NFS, rank 0 saves the
  image tar once and every worker loads it — no manual `docker load` per node.
- The Primus 70B recipe may sweep BF16+FP8; a BF16 cold-start flake can mark the
  run FAILURE and skip perf collection. Re-run for a clean perf CSV.
