# JAX MaxText in MAD

The source of truth for JAX MaxText training on AMD Instinct GPUs is the Primus guide:

**[Training a model with Primus and JAX MaxText](https://github.com/AMD-AGI/Primus/blob/main/docs/02-user-guide/jax-maxtext-training.md)**

For ROCm, JAX, Transformer Engine, hipBLASLt, RCCL, MaxText commit, and the rest of the `rocm/jax-training:maxtext-*` stack, use the Primus **[release notes](https://github.com/AMD-AGI/Primus/blob/main/docs/01-getting-started/release-notes.md)**. They are the single source of truth for image contents; look up the section for the tag you are running (MAD’s default base is `rocm/jax-training:maxtext-v26.7`).

That training guide covers supported models, required settings, `primus-cli`, multi-node networking, and profiling. Do not treat this README as a second copy of that material.

This page is only the MAD harness: how MAD discovers Primus configs and how to run them with `madengine`.

## Quick start

```sh
git clone https://github.com/ROCm/MAD
cd MAD
pip install -r requirements.txt

# Primus must exist at scripts/Primus before discovery or Docker build.
bash tools/fetch_primus.sh

export MAD_SECRETS_HFTOKEN="<your Hugging Face token>"
madengine discover --tags maxtext
madengine run --tags maxtext --live-output --timeout 14400
```

There is no single Primus image that covers every backend. `rocm/primus:*` is the PyTorch / Megatron / TorchTitan stack and does **not** include JAX. MAD builds `docker/primus_maxtext` from `docker/primus_maxtext.ubuntu.amd.Dockerfile` on top of `rocm/jax-training:maxtext-*`. Package versions for that base image are in the [Primus release notes](https://github.com/AMD-AGI/Primus/blob/main/docs/01-getting-started/release-notes.md).

## Fetching Primus

JAX models are discovered from `scripts/Primus/examples/maxtext/configs/` (and MaxDiffusion from `examples/maxdiffusion/configs/`). Check Primus out before discovery or image build:

```sh
bash tools/fetch_primus.sh
# or: git submodule update --init scripts/Primus
```

`tools/fetch_primus.sh` is idempotent. Override `PRIMUS_URL`, `PRIMUS_REF`, or `PRIMUS_DIR` for a fork, another branch or commit, or another location.

Cloning MAD with `--recursive` is **not** required. Primus `third_party/` submodules are not used for MAD builds; the base image already has `/workspace/maxtext` (and `/workspace/maxdiffusion`) at the pinned commit.

If `scripts/Primus` is missing, discovery finds **zero** JAX models and prints a warning naming `fetch_primus.sh`. In CI, set `MAD_AUTO_FETCH_PRIMUS=1` so discovery fetches Primus when the checkout is absent (off by default):

```sh
MAD_AUTO_FETCH_PRIMUS=1 madengine run --tags maxtext --live-output --timeout 14400
```

## Discovery and tags

`scripts/jax-maxtext/get_models_json.py` registers one virtual model per Primus MaxText YAML. Tags look like `jax-maxtext/maxtext_<DEVICE>_<config>`, for example `jax-maxtext/maxtext_MI300X_llama2_7B-bf16-pretrain`. Each model also carries tags such as `maxtext`, `jax`, `<DEVICE>`, `<config>`, and `<precision>`.

A `jax-maxtext/default` model is always registered, using the fallback config `scripts/jax-maxtext/run.sh` uses when no `--config_path` is given.

On MI300X machines, madengine skips MI355X models (and vice versa) via `skip_gpu_arch`. You do not need to add `MI300X` or `MI355X` to the tags.

Multi-node-only models (Llama 3.1 405B, Grok-1, Mixtral-8x22B) are excluded from single-node discovery. Set `JAX_MAXTEXT_INCLUDE_MULTINODE=1` to include them.

The live model list tracks whatever configs are in your `scripts/Primus` checkout. List it with discovery, or browse `scripts/Primus/examples/maxtext/configs/`.

```sh
madengine discover --tags maxtext          # all MaxText models
madengine discover --tags maxdiffusion     # all MaxDiffusion models
madengine discover --tags jax              # MaxText + MaxDiffusion
madengine discover --tags nanoo_fp8        # MI300X quantized models
```

## Running

```sh
export MAD_SECRETS_HFTOKEN="your personal Hugging Face token to access gated models"

madengine run --tags maxtext --live-output --timeout 14400
madengine run --tags jax-maxtext/maxtext_MI300X_llama2_7B-bf16-pretrain --keep-model-dir --live-output --timeout 28800
madengine run --tags jax-maxtext/maxtext_MI300X_llama2_7B-nanoo_fp8-pretrain --keep-model-dir --live-output --timeout 28800
```

`tools/run_models.py` remains a drop-in alternative to `madengine run` for the same `--tags`.

MAD starts a container named `container_ci-<mad_model>`. `scripts/jax-maxtext/run.sh` launches Primus; `scripts/jax-maxtext/extract_maxtext_perf.py` parses `tokens_per_second` and `tflops` into `~/MAD/perf.csv`.

For training flags, configs, and `primus-cli` (including Slurm), use the [Primus JAX MaxText training guide](https://github.com/AMD-AGI/Primus/blob/main/docs/02-user-guide/jax-maxtext-training.md).

## MaxDiffusion

MaxDiffusion is discovered the same way from `scripts/Primus/examples/maxdiffusion/configs/<DEVICE>/` by `scripts/jax-maxdiffusion/get_models_json.py`, tagged `jax-maxdiffusion/maxdiffusion_<DEVICE>_<config>`, and run through `scripts/jax-maxdiffusion/run.sh` with `docker/primus_maxdiffusion`. A `jax-maxdiffusion/default` model is always registered.

```sh
madengine run --tags maxdiffusion --live-output --timeout 14400
```

Example tags: `jax-maxdiffusion/maxdiffusion_MI300X_flux_dev-pretrain`, `jax-maxdiffusion/maxdiffusion_MI300X_wan2.1_1.3b-pretrain`.

## Profiling through MAD

Profiler flags live in the Primus YAML `overrides` block. Set them there, then run with `--keep-model-dir` so output is kept. See [Profiling with JAX XPlane Profiler](https://github.com/AMD-AGI/Primus/blob/main/docs/02-user-guide/jax-maxtext-training.md#profiling-with-jax-xplane-profiler) in the Primus guide.

## Related documentation

- [Primus JAX MaxText training guide](https://github.com/AMD-AGI/Primus/blob/main/docs/02-user-guide/jax-maxtext-training.md) — source of truth
- [Primus CLI reference](https://github.com/AMD-AGI/Primus/blob/main/docs/02-user-guide/cli-reference.md)
- [End-to-end training recipes](https://github.com/AMD-AGI/Primus/blob/main/docs/02-user-guide/end-to-end-training-recipes.md)
- [MaxText parameters](https://github.com/AMD-AGI/Primus/blob/main/docs/03-configuration-reference/maxtext-parameters.md)
- [Multi-node networking](https://github.com/AMD-AGI/Primus/blob/main/docs/04-technical-guides/multi-node-networking.md)
- [Release notes](https://github.com/AMD-AGI/Primus/blob/main/docs/01-getting-started/release-notes.md) — source of truth for packages in `rocm/jax-training:maxtext-*`
