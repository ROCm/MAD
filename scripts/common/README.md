# Running the multinode workloads

This covers the SLURM multinode workloads under `scripts/vllm_dissag` (vLLM disaggregated
prefill/decode), `scripts/sglang_disagg` (SGLang disaggregated) and `scripts/vllm_multinode`
(vLLM colocated multinode). Each is a **model card** in that directory's `models.json`,
backed by one batch script (the launcher). There are two ways to run a card, and they give the
job the same allocation and the same environment:

- **Through madengine**, which builds and pushes the image, picks healthy nodes, submits the
  launcher, and collects the results.
- **With `sbatch` directly**, exporting the card's environment and submitting its launcher
  yourself.

## What a run needs

| Piece | Where it lives |
|---|---|
| The card: launcher, node count, image recipe, environment | `scripts/<dir>/models.json` |
| The recipe: serve flags and env per model (vLLM disagg) | `scripts/vllm_dissag/models.yaml` |
| The image | `docker/<name>.ubuntu.amd.Dockerfile`, named by the card's `dockerfile` |
| The cluster's allocation defaults | `scripts/common/clusters/<cluster>.json` ([README](clusters/README.md)) |
| The cluster's in-job facts: weights, fabric, ports | `scripts/common/cluster.sh` |

The environment the launcher starts with is layered, weakest first:
1. `cluster.sh` defaults
2. the recipe in `models.yaml`
3. the card's `env_vars`
4. anything you set yourself: madengine's `env_vars`, or your exported environment under `sbatch`

## Through madengine

From the root of this repository, on a host with docker, build the card's image and push it
where the compute nodes can pull it:

```bash
pip install git+https://github.com/ROCm/madengine.git
export MAD_DOCKERHUB_USER=<user> MAD_DOCKERHUB_PASSWORD=<token>
madengine build --tags pyt_vllm_disagg_mori_deepseek-v3 --registry docker.io/<namespace> \
    --additional-context-file scripts/common/clusters/m2m.json \
    --manifest-output build_manifest.json
```

Then, from the SLURM login node, run it:

```bash
madengine run --manifest-file build_manifest.json \
    --additional-context-file scripts/common/clusters/m2m.json \
    --additional-context '{"slurm": {"nodes": 2, "time": "06:00:00"}}' \
    --timeout 21600 --live-output -o perf.csv
```

- **`--additional-context`** is merged over the cluster profile key by key. Put the run's
  shape and any overrides there:
  - `slurm.nodes`, `slurm.time`, `slurm.nodelist`
  - `env_vars` for the benchmark knobs or the topology, e.g.
    `{"env_vars": {"xP": "2", "yD": "2", "DURATION": "900"}}`
- **`--use-image <image>`** on `madengine build` skips the build and runs a prebuilt image.
- **Results:** `perf.csv` gets one row per model; a failed run is a `FAILURE` row with the
  reason. The job's own output is under `slurm_results/`.

## With sbatch directly

The card's `env_vars` are the launcher's contract. Export them, name the image, and submit the
launcher with the cluster's allocation:

```bash
cd scripts/vllm_dissag
export DOCKER_IMAGE_NAME=<registry>/<image>:<tag> \
       MODEL_NAME=DeepSeek-V3 xP=1 yD=1 WIDE_EP=1 RUN_MORI=1 RUN_DEEPEP=0 \
       BENCHMARK_COMBINATIONS=1024/1024
sbatch --partition=amd-rccl --nodes=2 --ntasks=2 --gpus-per-node=8 --exclusive \
       --time=06:00:00 --export=ALL run_xPyD_models.slurm
```

- **Options:** the `sbatch` options come from the cluster profile's `slurm` block, plus the
  run's node count and time; see the [mapping table](clusters/README.md).
- **`--time`:** each launcher's own `#SBATCH --time=24:00:00` exceeds most partition limits,
  so pass it.
- **Image access:** the launcher pulls `DOCKER_IMAGE_NAME` on every node and tolerates a failed
  pull. For a private registry, log in on the nodes first, or the job fails later at
  `docker run`.
- **Node count:** give `--nodes` the card's `distributed.nnodes` (or the `-N` in its `args`).
- **Previewing:** `DRY_RUN=1` prints each node's server command without starting anything:

  ```bash
  DRY_RUN=1 NODE_RANK=0 ... bash vllm_disagg.sh
  ```

  See `scripts/vllm_dissag/tests/argv_assert.sh` for complete examples.

## When a run fails

The launchers fail fast, on every node, and say why:

- **A server that dies during start-up:** the launcher prints its first error lines (repeats
  removed), the tail of its log, and which processes hold each GPU's memory.
- **Every other node:** the launcher writes `/run_logs/<job>/ABORTED`, so the other nodes stop
  at their next barrier instead of waiting out the time limit.
- **A cancelled job:** its containers are stopped, so a cancelled job does not leave GPU memory
  held on the nodes.
- **Per-node logs** are under `$LOG_PATH/<job>/`:
  - `prefill_NODE*.log` and `decode_NODE*.log`: the servers
  - `pd_vllm_bench_NODE*.log`: the benchmark
  - `proxy_NODE*.log`: the router

  `LOG_PATH` defaults to `/shared_inference/$USER/model_blog_logs`, from `cluster.sh`.

A recipe is validated on specific shapes and GPUs, and its `models.yaml` entry says which. Run
a card at one of those shapes before suspecting the recipe.

## Checks that need no GPUs

```bash
bash scripts/vllm_dissag/tests/argv_assert.sh      # the serve argv per connector and mode
python3 scripts/common/check_srun_quotes.py        # no apostrophe truncates an srun body
python3 scripts/common/check_gpu_arch_declarations.py   # cards and recipes agree on GPUs
```
