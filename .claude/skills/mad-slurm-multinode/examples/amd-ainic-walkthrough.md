# Example walkthrough — Primus Llama-3.1-8B on an AMD-AINIC / Pollara cluster

> Sanitized end-to-end walkthrough: no real node names, reservations, or tokens.
> Replace every `<...>` with your values.

Scenario: a 4-node Primus 8B run on a reserved AINIC partition.

## 0. Inputs gathered from the requester

| Input              | Example value                          |
|--------------------|----------------------------------------|
| Compute/jump node  | `<ainic-jump-node>`                     |
| `$WORKDIR`         | `<SHARED>/run`                          |
| Archetype          | AMD-AINIC / Pollara                    |
| Shared root        | `<SHARED>` (e.g. /shared/<user>)        |
| `MAD_DOCKER_BUILDS`| `<SHARED>/mad_docker_builds`            |
| SLURM              | partition `<PART>`, reservation `<RES>`, nodelist `<N1>,<N2>,<N3>,<N4>` |
| Nodes              | 4                                      |

## 1. Preflight + bootstrap

```bash
export SKILL_DIR="<absolute path to the mad-slurm-multinode skill dir>"
bash "$SKILL_DIR/scripts/preflight.sh"          # expect gfx950, docker ok, sbatch ok
bash "$SKILL_DIR/scripts/detect_cluster_env.sh" # expect rdma0..7, ionic, GID ~1, eno0
cd <SHARED>/run
[ -d MAD ] || git clone <MAD_REPO_URL> --recursive
( cd MAD && git switch "<MAD_BRANCH>" && git submodule update --init --recursive )
[ -d madengine ] || git clone <MADENGINE_REPO_URL> --recursive
( cd madengine && git switch "<MADENGINE_BRANCH>" && git submodule update --init --recursive )
conda env list | grep -q '^madenv ' || conda create -y -n madenv python=3.12
conda activate madenv
pip install -e ./madengine
```

## 2. rundir + mad.env (from mad.env.amd-ainic.template)

Filled values (illustrative):
```bash
export MAD_SYSTEM_GPU_ARCHITECTURE=gfx950
SHARE_DIR=<SHARED>
export MODEL_DIR=$SHARE_DIR/MAD/
export MAD_DOCKER_BUILDS=$SHARE_DIR/mad_docker_builds
export NCCL_SOCKET_IFNAME=eno0
export NCCL_IB_GID_INDEX=1
export RCCL_AINIC_ROCE=1
export RDMAV_DRIVERS=ionic
export IBV_DRIVERS=ionic
```
```bash
cd <SHARED>/run && mkdir -p rundir && cd rundir
cp "$SKILL_DIR/assets/mad.env/mad.env.amd-ainic.template" ./mad.env
# edit placeholders ...
source mad.env
```

## 3. Manifest (from primus_llama-3.1-8b.template.json)

Edits applied to `run_manifest_primus_8b_ainic.json`:
- `slurm.partition=<PART>`, `reservation=<RES>`,
  `nodelist=<N1>,<N2>,<N3>,<N4>`, `nodes=4` (remove `account` if unused)
- `distributed.nnodes=4`
- `NCCL_IB_HCA=rdma0:1,rdma1:1,rdma2:1,rdma3:1,rdma4:1,rdma5:1,rdma6:1,rdma7:1`
  in BOTH env blocks
- `NCCL_SOCKET_IFNAME=eno0`, `GLOO_SOCKET_IFNAME=eno0`, `NCCL_IB_GID_INDEX=1`
- `RDMAV_DRIVERS=ionic`, `IBV_DRIVERS=ionic`, `RCCL_AINIC_ROCE=1` in BOTH blocks
- `slurm.network_interface=eno0`
- `docker_image=rocm/primus:v26.2-rccl-<drop>-<sha7>`

For a custom RCCL drop you can also set `context.docker_build_arg` with
`BUILD_GPU_TARGETS=gfx950`, `RCCL_REPO`, `RCCL_COMMIT` — this rebuilds RCCL via
the `rccl_overlay` Dockerfile shipped in MAD.
```bash
python3 -m json.tool run_manifest_primus_8b_ainic.json >/dev/null && echo "valid"
```

## 4. Launch + read result

```bash
source mad.env
madengine run --manifest-file run_manifest_primus_8b_ainic.json --live-output
squeue -u $USER
```
Confirm the AINIC/ionic transport was selected in the rank-0 RCCL debug log
before trusting the number (otherwise it fell back to verbs/sockets — see
references/cluster-types.md). Aggregated perf in `rundir/perf.csv`.

## Notes

- 4 nodes minimum is recommended for AINIC to exercise multi-rail collectives.
- Hold the same node set across compared runs (`nodelist`) for apples-to-apples.
- Deep AINIC transport validation / RCCL build-vs-build deltas are a separate
  workflow, out of scope for this perf-run skill.
