# Example walkthrough — Primus Llama-3.1-8B on a CX7 / Mellanox-RoCE cluster

> Sanitized end-to-end walkthrough: no real node names, SLURM queues, accounts,
> or tokens. Replace every `<...>` with your values.

Scenario: bring up a fresh CX7 login node and run a 2-node Primus 8B perf test.

## 0. Inputs gathered from the requester

| Input              | Example value                          |
|--------------------|----------------------------------------|
| Compute/login node | `<cx7-login-node>`                      |
| `$WORKDIR`         | `~/source/run1`                        |
| Archetype          | CX7 / Mellanox-RoCE                    |
| Data root          | `<DATA_ROOT>` (shared FS)               |
| `MAD_DOCKER_BUILDS`| `<SHARED>/mad_docker_builds`            |
| SLURM              | partition `<PART>`, account `<ACCT>`, qos `<QOS>` |
| Nodes              | 2                                      |

## 1. Preflight + bootstrap

```bash
export SKILL_DIR="<absolute path to the mad-slurm-multinode skill dir>"
bash "$SKILL_DIR/scripts/preflight.sh"   # expect gfx942, docker ok, sbatch ok
cd ~/source/run1
[ -d MAD ] || git clone <MAD_REPO_URL> --recursive
( cd MAD && git switch "<MAD_BRANCH>" && git submodule update --init --recursive )
[ -d madengine ] || git clone <MADENGINE_REPO_URL> --recursive
( cd madengine && git switch "<MADENGINE_BRANCH>" && git submodule update --init --recursive )
conda env list | grep -q '^madenv ' || conda create -y -n madenv python=3.12
conda activate madenv
pip install -e ./madengine
```

## 2. rundir + mad.env (from mad.env.cx7-roce.template)

Filled values (illustrative):
```bash
export MAD_SYSTEM_GPU_ARCHITECTURE=gfx942
export MODEL_DIR=~/source/run1/MAD/
export MAD_DATAHOME=<DATA_ROOT>/models
export MAD_DOCKER_BUILDS=<SHARED>/mad_docker_builds
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
```
```bash
cd ~/source/run1 && mkdir -p rundir && cd rundir
cp "$SKILL_DIR/assets/mad.env/mad.env.cx7-roce.template" ./mad.env
# edit placeholders ...
source mad.env
[ -d "$MODEL_DIR/scripts" ] && echo "MODEL_DIR ok"
```

## 3. Manifest (from primus_llama-3.1-8b.template.json)

Edits applied to `run_manifest_primus_8b.json`:
- `slurm.partition=<PART>`, `account=<ACCT>`, `qos=<QOS>`, `nodes=2`
- `distributed.nnodes=2`
- `NCCL_IB_HCA=mlx5_0:1,mlx5_1:1` in BOTH `context.docker_env_vars` and
  `deployment_config.env_vars`
- `NCCL_SOCKET_IFNAME=eth0`, `GLOO_SOCKET_IFNAME=eth0`, `NCCL_IB_GID_INDEX=3`
- `RDMAV_DRIVERS=mlx5`, `IBV_DRIVERS=mlx5`, deleted the `RCCL_AINIC_ROCE` key
- `docker_image=rocm/primus:v26.2-rccl-<branch>-<sha7>`
```bash
python3 -m json.tool run_manifest_primus_8b.json >/dev/null && echo "manifest valid"
```

## 4. Launch + read result

```bash
source mad.env
madengine run --manifest-file run_manifest_primus_8b.json --live-output
squeue -u $USER          # watch the 2-node job
```
Result lands in `rundir/perf.csv` (aggregated) and
`perf_primus-megatron-Llama-3.1-8B.csv`. Report tok/s/gpu and TFLOPS/gpu from
the aggregated CSV (node_0's local CSV can read empty — see
references/launch-and-results.md).

## Notes

- 70B: same flow with `primus_llama-3.1-70b.template.json`; bump walltime/nodes.
- Data staging (download + preprocessing) is handled by the MAD `run.sh`; the
  skill just fills the workload's data mounts and points the env vars at them.
