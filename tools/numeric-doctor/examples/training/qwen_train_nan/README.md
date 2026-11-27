# Qwen2.5-0.5B fp32 training goes NaN on gfx950

Training `Qwen/Qwen2.5-0.5B` in fp32 (TF32 matmuls enabled) with 3 training processes per GPU, some trials
reach a NaN loss after a few to a few hundred steps; the others train normally to the end. It happens on
about 4 in 10 trials and the failing trials change from run to run.

The workload is the public Qwen2.5-0.5B architecture in fp32 with `allow_tf32=True`, random init, AdamW,
random tokens, sequence length 1280, batch size 1, 300 steps, and 3 training processes (`mp.spawn`) per GPU on
all 8 GPUs of a node. The image is AMD's public `rocm/pytorch` release image plus the trainer script
(`Dockerfile`); nothing is patched. Training from the pretrained checkpoint fails the same way (see Result).

| File | Content |
| --- | --- |
| `run.sh` | builds the image, runs the training on every GPU (Docker, Podman or Apptainer), prints a verdict |
| `Dockerfile` | the image: a public `rocm/pytorch` base pinned by digest + the trainer script |
| `qwen_train.py` | Qwen2 model (plain PyTorch, Qwen2.5-0.5B config built in) + training loop + NaN detector |
| `capture.spec` | the same run as a `scripts/jobs/wave.sh` spec for debug-numeric |

The model is the real Qwen2 architecture, not a look-alike. With the official checkpoint loaded into both,
it matches Hugging Face `transformers` 5.20.2 (`Qwen2ForCausalLM`, fp32, eager attention) to fp32 rounding:
494,032,768 parameters with identical names, max logit difference 5.8e-5 at a logit scale of 24, identical
loss (2.9411 on English text, perplexity 18.9), and gradients within 4.7e-5 relative. It is written in plain
PyTorch only so that nothing has to be installed into the image.

## Environment

| | |
| --- | --- |
| Image name (built by `run.sh`) | `qwen-train-nan:rocm7.2.0` |
| Base (Docker Hub) | `rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.10.0` |
| Base digest | `sha256:bcb2efa1d43f66cc59fa45a1975d621a50b2b06d21a2abdfa2ba821ab32763e4` |
| ROCm / HIP / PyTorch | 7.2.0 / 7.2.26015 / 2.10.0+rocm7.2.0.gitb6ee5fde |
| OS / Python | Ubuntu 24.04 / 3.12 |
| GPU | 8x MI350X (gfx950) |

## Run

Needs one node with 8 gfx950 GPUs (MI350X / MI355X) and Docker, Podman or Apptainer. No registry login is
needed: the base is public. `run.sh` uses the first of `docker`, `podman`, `apptainer` that is installed;
`RUNTIME=docker|podman|apptainer` chooses one.

```bash
cd examples/training/qwen_train_nan
./run.sh                                   # a few minutes; the first build pulls about 10 GB
```

With SLURM, submit from this folder: `sbatch -p <gfx950 partition> run.sh`.

The first run builds `qwen-train-nan:rocm7.2.0` from the `Dockerfile` and reuses it afterwards.
`./run.sh pull` only builds it. By hand: `docker build -t qwen-train-nan:rocm7.2.0 .`

`run.sh` builds with an empty `DOCKER_CONFIG` (the base is public, so it pulls anonymously) and
`--network=host`: a stored but expired Docker Hub login (`401 ... personal access token is expired`) or a daemon
without a default bridge network does not stop the build. When building by hand, prefix `docker build` with
`DOCKER_CONFIG=$(mktemp -d)` if a pull fails with `authentication required`. Podman runs rootless with one
mapped uid and host networking, so it works without `/etc/subuid` ranges.

Apptainer (HPC clusters) cannot build the `Dockerfile`. `run.sh` converts the base once to
`<SIF_DIR>/qwen-train-nan_rocm7.2.0.sif` (default `SIF_DIR=./sif`) and supplies what the `Dockerfile` adds (the
environment and the trainer script) at run time; an existing SIF with that name is used as it is (a SIF with
another name can be linked in under that name).

```bash
RUNTIME=apptainer sbatch -p <gfx950 partition> -t 00:45:00 run.sh pull   # optional: convert the base first
RUNTIME=apptainer sbatch -p <gfx950 partition> run.sh
```

Podman storage and Apptainer layers go to node-local scratch (`SCRATCH_DIR`, removed afterwards unless set),
not to `$HOME`. Apptainer runs without `--rocm`; host `/dev/kfd` is mounted anyway.

Optional settings: `GPUS=8 STEPS=300 SEQ=1280 BS=1 TRIALS=3`, `SEED=` (random by default, printed, so a run can
be repeated), `TRAIN_ARGS=` (extra trainer arguments), `IMAGE=`/`BASE=` (image name / base image). Logs go to
`out/<date>/`.

Expected end of the output (`qwen-train-nan_<job>.out` under SLURM):

```text
RESULT trial=1 step_reached=16 nan_step=16 nan_where=loss nan_module=model.layers.2.self_attn.q_proj loss0=12.1257 loss_last=nan seed=14030 tf32=1 seq=1280 bs=1
...

=== summary: seed base 13029, host <node> ===
9/24 trials with NaN
VERDICT REPRODUCED
```

Each trial prints one `RESULT` line; `run.sh` repeats the failing ones. `nan_step` is the first step with a
non-finite loss, gradient or parameter (`-` if clean); `nan_module` is the first module whose output went
non-finite. A clean trial reaches step 299 with `nan_step=-`; with random init and random tokens the loss stays
near ln(vocab) ≈ 12 (12.10 → 12.09), with the checkpoint it falls from 13.5 to about 11.94. The loss trains
normally and then goes to NaN between two steps; the first non-finite module differs from trial to trial.

To train from the pretrained checkpoint instead, put it in this folder and pass it through:

```bash
mkdir -p models && cd models && for f in config.json model.safetensors; do
  curl -fsSLO "https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/060db6499f32faf8b98477b0a26969ef7d8b9987/$f"; done
cd .. && TRAIN_ARGS="--config models/config.json --weights models/model.safetensors" ./run.sh
```

## Result (MI350X, 2026-11-02)

| Weights | Trials with NaN |
| --- | --- |
| random init (default) | **9/24, 9/24, 7/24, 11/24** (four jobs, four nodes) |
| checkpoint | **9/24** |

`run.sh` prints the image (ID and base digest with Docker/Podman, SIF path with Apptainer), versions and GPU
arch.

## Notes

- The image has no passwd entry for the host uid, and the optimizer imports dynamo, which calls
  `getpass.getuser()`. The `Dockerfile` and `run.sh` set `USER`/`LOGNAME`; without them every trial exits
  silently. If trials crash, `run.sh` prints `WARNING: only N of M trials printed a RESULT line` and the first
  error.
- If SLURM preempts the job, let it requeue rather than cancelling it.
- Without `run.sh`, one GPU with the image alone:
  `docker run --rm --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=0
  qwen-train-nan:rocm7.2.0 python3 qwen_train.py --seq-len 1280 --batch-size 1 --steps 300 --trials 3
  --seed <seed>`
