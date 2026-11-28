# 20-layer soft-MoE transformer, mostly torch.compile'd: the loss goes NaN

A small decoder-only transformer trained from scratch on one AMD Instinct GPU, bf16 autocast,
with layers 3-18 compiled. The loss falls normally, then a step's loss is NaN and training stops
with `RuntimeError: Loss is not finite`. The step varies between runs (46 and 78 seen in 100 steps).

The model has 20 layers, numbered 1 to 20. Each is a pre-norm block: RMSNorm, causal self-attention
with RoPE, residual; RMSNorm, soft mixture-of-experts SwiGLU MLP, residual. The MLP's hidden units form
4 experts; a router (temperature 0.02, logits soft-capped at 50 as in Gemma 2, zero-initialised so it
starts uniform) gives each token a softmax over the experts, and expert e's output is weighted by
`sqrt(p_e)`, so the squared weights sum to 1 and the mixture keeps its experts' variance.
**Layers 3-18 are each wrapped in `torch.compile`** (inductor); layers 1-2 and 19-20, the embedding, final
norm, head and loss run eagerly. The data is a synthetic token stream (a random sparse Markov chain over Zipf-distributed
tokens) cut into rows of 1025 tokens, 8 rows per step, in order, 25 steps per epoch; pad positions are
masked out of the loss. AdamW, lr 3e-4 with 10 warmup steps, gradient clipping at 1.0, seed 0.

## Run

```bash
cd examples/training/softmoe_train_nan
./run.sh            # on a GPU node: trains on GPU 0, prints the last steps and a verdict
                    # on a login node (no /dev/kfd): submits itself with sbatch; ./run.sh status
```

Logs go to `out/<date>/` (`train_g0.log`, `steps.txt`). The run ends with
`RESULT step_reached=<n> nan_step=<n|-> final_loss=<x|->` and `VERDICT NAN` or `VERDICT CLEAN`.
Settings: `STEPS=100`, `GPUS=1` (one independent trainer per GPU), `COMPILE_LAYERS=3-18`
(the compiled layers, e.g. `1-20`), `TRAIN_ARGS=<more train.py flags>`, `RUNTIME=docker|podman|apptainer`;
self-submission takes `ACCOUNT`, `PARTITION` (default: the cluster's) and `TIME=00:30:00`.

To use a saved image instead of pulling from Docker Hub (rate limits, no internet), point
`IMAGE_TAR` at a `docker save` archive (`.tar`, `.tar.gz` or `.tar.zst`):

```bash
IMAGE_TAR=/path/to/image.tar.zst ./run.sh                     # loads it on every run
IMAGE_TAR=/path/to/image.tar.zst IMAGE=<name:tag> ./run.sh    # loads it only where <name:tag> is missing
```

Without containers, in any environment with PyTorch:

```bash
python3 train.py --steps 100                          # GPU
python3 train.py --device cpu --amp none --dim 128 --heads 4 --hidden 352 --vocab 512 \
                 --seq-len 64 --steps 30              # small, on CPU
```

## What is here

| Path | What |
| --- | --- |
| `train.py` | the model, data and training loop (PyTorch only) |
| `run.sh` | runs `train.py` in the image below (Docker, Podman or Apptainer), prints loss / grad_norm per step and a verdict |
| `sif/` | Apptainer: the image as a SIF (pulled by `run.sh` if missing) |

The image is AMD's public `rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.10.0`,
pinned by digest; nothing is installed into it.
