# JAX training in MAD: MaxText and MaxDiffusion

MAD runs two JAX backends through Primus: **MaxText** (LLM pretraining) and **MaxDiffusion** (FLUX / WAN diffusion training). Both are discovered from a Primus checkout and launched by `madengine`. This page documents that harness only.

For how MaxText training itself works — supported models, required settings, `primus-cli`, multi-node networking, profiling — the source of truth is the Primus guide:

**[Training a model with Primus and JAX MaxText](https://github.com/AMD-AGI/Primus/blob/main/docs/02-user-guide/jax-maxtext-training.md)**

Do not treat this README as a second copy of that material. Primus has no equivalent MaxDiffusion user guide yet, so for MaxDiffusion the configs under `examples/maxdiffusion/configs/` are the reference.

For ROCm, JAX, Transformer Engine, hipBLASLt, RCCL, MaxText commit, and the rest of the `rocm/jax-training:maxtext-*` stack, use the Primus **[release notes](https://github.com/AMD-AGI/Primus/blob/main/docs/01-getting-started/release-notes.md)**. They are the single source of truth for image contents; look up the section for the tag you are running. Both MAD dockerfiles currently default to `rocm/jax-training:maxtext-v26.7`, which also carries the MaxDiffusion stack.

## Quick start

```sh
git clone https://github.com/ROCm/MAD
cd MAD
pip install -r requirements.txt

# Primus must exist at scripts/Primus before discovery or Docker build.
bash tools/fetch_primus.sh

export MAD_SECRETS_HFTOKEN="<your Hugging Face token>"
madengine discover --tags jax
madengine run --tags maxtext --live-output --timeout 14400
```

There is no single Primus image that covers every backend. `rocm/primus:*` is the PyTorch / Megatron / TorchTitan stack and does **not** include JAX. MAD builds `docker/primus_maxtext` and `docker/primus_maxdiffusion` on top of `rocm/jax-training:maxtext-*`, which is the only published image with JAX.

## Fetching Primus

Models are discovered from `scripts/Primus/examples/maxtext/configs/` and `scripts/Primus/examples/maxdiffusion/configs/`. Check Primus out before discovery or image build:

```sh
bash tools/fetch_primus.sh
# or: git submodule update --init scripts/Primus
```

`tools/fetch_primus.sh` is idempotent. Override `PRIMUS_URL`, `PRIMUS_REF`, or `PRIMUS_DIR` for a fork, another branch or commit, or another location.

Cloning MAD with `--recursive` is **not** required. Primus `third_party/` submodules are not used for MAD builds: the base image ships `/workspace/maxtext` and `/workspace/maxdiffusion`, and each `run.sh` pins `MAXTEXT_PATH` / `MAXDIFFUSION_PATH` at those paths so a mismatched `third_party` copy cannot be picked up.

If `scripts/Primus` is missing, discovery finds **zero** JAX models and prints a warning naming `fetch_primus.sh`. In CI, set `MAD_AUTO_FETCH_PRIMUS=1` so discovery fetches Primus when the checkout is absent (off by default):

```sh
MAD_AUTO_FETCH_PRIMUS=1 madengine run --tags jax --live-output --timeout 14400
```

## Discovery and tags

`scripts/jax-maxtext/get_models_json.py` and `scripts/jax-maxdiffusion/get_models_json.py` each register one virtual model per Primus YAML under `examples/<backend>/configs/<DEVICE>/`. New configs are picked up automatically.

| Backend | Discovered name | Dockerfile | Launcher |
| ------- | --------------- | ---------- | -------- |
| MaxText | `jax-maxtext/maxtext_<DEVICE>_<config>` | `docker/primus_maxtext` | `scripts/jax-maxtext/run.sh` |
| MaxDiffusion | `jax-maxdiffusion/maxdiffusion_<DEVICE>_<config>` | `docker/primus_maxdiffusion` | `scripts/jax-maxdiffusion/run.sh` |

Each model carries the tags `maxtext` or `maxdiffusion`, plus `jax`, `<DEVICE>`, `<config>`, and `<precision>` (`bf16`, `fp8`, or `nanoo_fp8`, inferred from the filename).

```sh
madengine discover --tags maxtext          # all MaxText models
madengine discover --tags maxdiffusion     # all MaxDiffusion models
madengine discover --tags jax              # both
madengine discover --tags nanoo_fp8        # MI300X-style quantized models
```

`<DEVICE>` is whatever directory names exist in the checkout — currently `MI300X`, `MI325X`, and `MI355X`. Discovery sets `skip_gpu_arch` so MI300X configs are skipped on gfx950 hosts and MI355X configs on gfx942, which lets one discovery serve both host types. Only those two devices are mapped, so MI325X configs are never auto-skipped.

Each backend also registers a `default` model (`jax-maxtext/default`, `jax-maxdiffusion/default`) as a smoke test, pinned to one config — Llama 2 7B bf16 on MI300X and WAN 2.1 1.3B on MI355X respectively. They are tagged only `default`, so they never appear in `--tags maxtext`, `--tags maxdiffusion`, or `--tags jax` sweeps and cannot duplicate the per-YAML entry. Reach them by full name:

```sh
madengine run --tags jax-maxtext/default --live-output --timeout 28800
```

Multi-node-only MaxText models (Llama 3.1 405B, Grok-1, Mixtral-8x22B) are excluded from single-node discovery; set `JAX_MAXTEXT_INCLUDE_MULTINODE=1` to include them. MaxDiffusion has the same switch (`JAX_MAXDIFFUSION_INCLUDE_MULTINODE=1`) but no models on its exclusion list today.

## Running

```sh
export MAD_SECRETS_HFTOKEN="your personal Hugging Face token to access gated models"

madengine run --tags maxtext --live-output --timeout 14400
madengine run --tags maxdiffusion --live-output --timeout 14400
madengine run --tags jax-maxtext/maxtext_MI300X_llama2_7B-bf16-pretrain --keep-model-dir --live-output --timeout 28800
madengine run --tags jax-maxdiffusion/maxdiffusion_MI300X_flux_dev-pretrain --keep-model-dir --live-output --timeout 28800
```

`tools/run_models.py` remains a drop-in alternative to `madengine run` for the same `--tags`.

MAD starts a container named `container_ci-<mad_model>`. Inside it, `run.sh` sets `EXP` from `--config_path` and calls Primus `examples/run_pretrain.sh` with `BACKEND=MaxText` or `BACKEND=MaxDiffusion`, skipping the per-run `pip install` (`PRIMUS_SKIP_PIP=1`) so a launch stays off the network. `MAD_SECRETS_HFTOKEN` is forwarded to Primus as `HF_TOKEN`.

Performance is parsed by `extract_maxtext_perf.py` / `extract_maxdiffusion_perf.py` into `primus_perf_output.csv`, which madengine collects as `multiple_results` and aggregates into `~/MAD/perf.csv`. Values are averaged over the trailing steps and reported per GPU: MaxText writes `tok_per_s_per_gpu` and `TFLOPS_per_gpu`; MaxDiffusion writes `fps_per_gpu`, `images_per_sec_per_gpu`, and `TFLOPS_per_gpu`.

For training flags, configs, and `primus-cli` (including Slurm), use the [Primus JAX MaxText training guide](https://github.com/AMD-AGI/Primus/blob/main/docs/02-user-guide/jax-maxtext-training.md).

## Profiling through MAD

Profiler flags live in the Primus YAML `overrides` block. Set them there, then run with `--keep-model-dir` so the output directory survives the run. See [Profiling with JAX XPlane Profiler](https://github.com/AMD-AGI/Primus/blob/main/docs/02-user-guide/jax-maxtext-training.md#profiling-with-jax-xplane-profiler) in the Primus guide.

## Related documentation

- [Primus JAX MaxText training guide](https://github.com/AMD-AGI/Primus/blob/main/docs/02-user-guide/jax-maxtext-training.md) — source of truth for MaxText training
- [Primus CLI reference](https://github.com/AMD-AGI/Primus/blob/main/docs/02-user-guide/cli-reference.md)
- [End-to-end training recipes](https://github.com/AMD-AGI/Primus/blob/main/docs/02-user-guide/end-to-end-training-recipes.md)
- [MaxText parameters](https://github.com/AMD-AGI/Primus/blob/main/docs/03-configuration-reference/maxtext-parameters.md)
- [Multi-node networking](https://github.com/AMD-AGI/Primus/blob/main/docs/04-technical-guides/multi-node-networking.md)
- [Release notes](https://github.com/AMD-AGI/Primus/blob/main/docs/01-getting-started/release-notes.md) — source of truth for packages in `rocm/jax-training:maxtext-*`
