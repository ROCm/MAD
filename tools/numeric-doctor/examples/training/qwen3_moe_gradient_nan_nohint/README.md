# Qwen3 MoE training: exploding / NaN gradient norm

torchtitan's Qwen3 MoE debug model trained on 8 AMD Instinct GPUs (gfx950) with FSDP 4 x TP 2 x EP 4,
bf16, deterministic mode. The gradient norm, normally around 1-4, jumps to values in the
thousands and beyond on some steps, and after a few such steps the run goes NaN:
`RuntimeError: Loss is not finite`. The loss itself trains normally between the jumps.

## Run

```bash
cd examples/training/qwen3_moe_gradient_nan_nohint
./run.sh            # login node: fetches what is missing, then submits itself on one 8-GPU node
./run.sh 4          # the same on 4 nodes (32 GPUs)
./run.sh status     # newest job log
ACCOUNT=<account> PARTITION=<partition> ./run.sh     # your site's account and partition
```

The runtime is Apptainer where it is installed, else Docker (`RUNTIME=apptainer|docker` to choose).
With Apptainer the image is built on the login node (compute nodes may have no internet); with Docker
each node builds it from `Dockerfile` the first time it runs. The job asks Slurm for
`--gpus-per-node=$NGPU` only where Slurm has GPU gres (`GPUS_FLAG=` to override); with plain
`sbatch run.sh` on such a cluster, add `--gpus-per-node=8` yourself.

On a GPU node, or with `sbatch run.sh`, it runs in place. The output ends with one line per step
(`step  loss  grad_norm`); logs go to `out/<date>-<jobid>/`. Under `sbatch -N <n> run.sh` it trains on all
of the job's nodes. Settings: `STEPS=100`, `NGPU=8`,
`EXTRA_ARGS=<more torchtitan flags>`; self-submission uses `ACCOUNT`, `PARTITION` (default: the cluster's)
and `TIME=01:00:00`.

## What is here

| Path | What |
| --- | --- |
| `run.sh` | prepares what is missing, runs the training command below, prints loss / grad_norm per step |
| `Dockerfile` | the same image for Docker |
| `qwen3_moe.def` | the Apptainer image recipe: public Python 3.12 base + PyTorch 2.14.0 ROCm wheel + torchtitan's dependencies |
| `sif/qwen3-moe.sif` | that image (built by `run.sh` if missing) |
| `src/torchtitan` | torchtitan source at commit `ecae62f` (fetched by `run.sh` if missing) |
| `model/seed_checkpoint/step-0` | the initial weights: torchtitan's seed checkpoint of `qwen3_moe_debug`, seed 42 (created by `run.sh` if missing) |

The data (`c4_test`) and tokenizer are torchtitan's test assets in `src/torchtitan/tests/assets`.

## The training command

```bash
MODULE=qwen3 CONFIG=qwen3_moe_debug NGPU=8 LOG_RANK=0 ./run_train.sh \
  --parallelism.tensor_parallel_degree 2 --parallelism.expert_parallel_degree 4 \
  --debug.deterministic --debug.seed=42 --metrics.enable_tensorboard --metrics.log_freq=1 \
  --training.steps=100 --checkpoint.enable --checkpoint.export_dtype=bfloat16 --checkpoint.load_only
```

(run from `src/torchtitan`, with `model/seed_checkpoint/step-0` linked as the run's `checkpoint/step-0`).

On N > 1 nodes every node keeps the FSDP 4 x TP 2 x EP 4 layout above, and the nodes are HSDP replicas:
the same flags plus `--parallelism.data_parallel_replicate_degree N`, launched with one `torchrun --nnodes N`
per node through `srun` (rank 0's log is `train.log`; node k's local rank 0 goes to `train.node<k>.log`).
The local batch is the same as on one node; the global batch is N times larger.
