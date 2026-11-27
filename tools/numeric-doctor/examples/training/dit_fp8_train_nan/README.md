# FP8 DiT training: the gradient norm is NaN at the first step

A small diffusion transformer trained in FP8 with torchao on gfx950 stops at step 0 on every GPU with
`Gradient norm is not finite at step 0: nan`. The loss of that step is finite (about 3) and looks normal.

The workload: a DiT with adaLN-Zero blocks and a rectified-flow (velocity) loss, hidden 3072, 24 heads, 2 blocks,
fused QKV projection, MLP ratio 4, batch 4 x 4096 latent patches = 16384 tokens per step, bf16 weights, random init,
synthetic low-rank latents. AdamW with linear warmup and gradient clipping at 1.0. The block linears (`qkv`, `proj`,
`fc1`, `fc2`) are converted with torchao's `convert_to_float8_training` with its default recipe, so they run FP8
matmuls. One trainer per GPU on all 8 GPUs of a node, 30 steps, with PyTorch TunableOp enabled
(`PYTORCH_TUNABLEOP_ENABLED=1`, one results file per trainer), as in the training setup this came from. The image is
AMD's public `rocm/primus:v26.3` plus the trainer script (`Dockerfile`); nothing is patched.

| File | Content |
| --- | --- |
| `run.sh` | builds the image, runs one trainer per GPU (Docker, Podman or Apptainer), prints a verdict |
| `Dockerfile` | the image: the public `rocm/primus:v26.3` pinned by digest + the trainer script |
| `dit_fp8_train.py` | the DiT model + training loop; prints loss / grad norm, stops on a non-finite one |
| `capture.spec` | the same run as a `scripts/jobs/wave.sh` spec for debug-numeric |

## Environment

| | |
| --- | --- |
| Image name (built by `run.sh`) | `dit-fp8-train-nan:primus-v26.3` |
| Base (Docker Hub) | `rocm/primus:v26.3` |
| Base digest | `sha256:1a02b74a94d82131f3f119a94ccbed45bb5b4cd77a0616cb4e6a29e64a048482` |
| ROCm / HIP / PyTorch | 7.2.1 / 7.2.53211 / 2.10.0+git94c6e04 |
| torchao | 0.15.0+gite9c7bead9 |
| GPU | 8x MI350X (gfx950) |
| Training environment | `PYTORCH_TUNABLEOP_ENABLED=1`, `PYTORCH_TUNABLEOP_FILENAME=<out>/g<gpu>_tunableop_%d.csv` |

## Run

Needs one node with 8 gfx950 GPUs and Docker, Podman or Apptainer. The base is public (about 22 GB compressed; the
first build pulls it, about 6 min). `RUNTIME=docker|podman|apptainer` chooses the runtime.

```bash
cd examples/training/dit_fp8_train_nan
./run.sh                                   # about 4 min after the pull
sbatch -p <gfx950 partition> run.sh        # the same on one exclusive SLURM node
```

`./run.sh pull` only prepares the image. Options: `GPUS=8 STEPS=30 SEED=<random> TRAIN_ARGS=... OUT_DIR=...
RUNTIME=... SIF_DIR=./sif`. Logs and the TunableOp results files go to `out/<date>/`.

The first step takes about 3 minutes (TunableOp tunes each new GEMM shape on first use); later steps take well under a
second. Expected end of the output (`dit-fp8-train-nan_<job>.out` under SLURM):

```text
RESULT trial=0 step_reached=0 nan_step=0 nan_where=grad_norm loss0=3.3879 loss_last=3.3879 seed=21101 fp8=torchao tokens=16384 secs=182
...

=== summary: seed base 21101, host <node> ===
8/8 trials with NaN
VERDICT REPRODUCED
```

A trainer prints `trial <t> step <n> loss <l> grad_norm <g> lr <lr>` every 5 steps, then
`Gradient norm is not finite at step <n>: nan` (or `Loss is not finite ...`) and its `RESULT` line. `nan_step` is
the first step with a non-finite loss or gradient norm (`-` if clean), `nan_where` which of the two. A clean trial
reaches step 29 with `nan_step=-`.

The NaN signal for debug-numeric:

```text
NAN_SIGNAL='Gradient norm is not finite|Loss is not finite'
```

Without `run.sh`, one GPU with the image alone:

```bash
docker run --rm --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=0 \
  -e PYTORCH_TUNABLEOP_ENABLED=1 dit-fp8-train-nan:primus-v26.3 \
  python3 dit_fp8_train.py --steps 30 --seed 1753 --log-every 5
```

## Result (MI350X, 2026-11-20)

| Job | Trials with NaN |
| --- | --- |
| 145661 | **8/8**, all at step 0 |
| 145736 | **8/8**, all at step 0 |
| 145756 | **8/8**, all at step 0 |
| 154139 (this folder's `run.sh`, 2026-11-26) | **8/8**, all at step 0 |

## Notes

- The image has no passwd entry for the host uid; the `Dockerfile` and `run.sh` set `USER`/`LOGNAME`/`HOME`. If
  trials crash, `run.sh` prints `WARNING: only N of M trials printed a RESULT line` and the first error.
- Not run on MI355X.
