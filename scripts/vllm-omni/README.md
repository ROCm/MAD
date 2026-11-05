# vLLM-Omni Diffusion Benchmarking Scripts

Perf benchmarking for **diffusion** models (text-to-image / text-to-video) served by
[vLLM-Omni](https://github.com/vllm-project/vllm-omni).

## Why a separate engine

Diffusion models are served by `vllm serve <model> --omni` (OpenAI image-generation API)
and benchmarked with the omni diffusion bench
(`benchmarks/diffusion/diffusion_benchmark_serving.py`). Neither exists on the plain
`vllm/vllm-openai-rocm` image used by `scripts/vllm`, so vLLM-Omni gets its own engine,
docker base, and configs rather than a branch inside the LLM harness.

## Directory structure

```
vllm-omni/
├── run.sh                        # Entrypoint invoked by madengine
├── run_vllm_omni.py              # Orchestrator (server + bench/accuracy + CSV)
├── accuracy_runner_vllm_omni.py  # CLIPScore accuracy check
├── models.json                   # MAD registration (tags: vllm-omni-perf, vllm-omni-accuracy)
└── configs/
    ├── default.yaml      # Run shapes (resolution, tp, concurrency) + accuracy rows
    ├── perf.yaml         # Perf tuning + docker_base per model
    └── acc.yaml          # Accuracy evaluations (task defaults + model overrides)
```

Docker base: `docker/vllm-omni.ubuntu.amd.Dockerfile`
(`ARG BASE_DOCKER=vllm/vllm-omni-rocm:v0.22.0`).

## Usage

```bash
madengine run --tags vllm-omni-perf --live-output --keep-model-dir \
  --additional-context '{"model_args": "--config configs/default.yaml --model_repo black-forest-labs/FLUX.1-dev", "gpu_vendor": "AMD", "docker_build_arg": {"MAD_SYSTEM_GPU_ARCHITECTURE": "gfx950", "BASE_DOCKER": "vllm/vllm-omni-rocm:v0.22.0"}}'
```

Or directly inside the omni container:

```bash
./run.sh --model_repo black-forest-labs/FLUX.1-dev --config configs/default.yaml
```

## Config schema

Run shape lives in `configs/default.yaml`; perf tuning (`extra_args` / `env` /
`docker_base`) lives in `configs/perf.yaml` and is gap-filled into the base entry
(base always wins).

| Key | Meaning |
|-----|---------|
| `benchmark` | `serving` (diffusion bench) or `accuracy` (CLIPScore check, see below) |
| `hf_pipeline_tag` | `text-to-image` / `image-to-image` / `text-to-video` / `image-to-video` |
| `tp` | Tensor-parallel size (`-tp`) |
| `height` / `width` | Output resolution |
| `num_images` | Images per request |
| `num_inference_steps` | Denoising steps; unset -> bench default 50 |
| `max_concurrency` | Concurrent requests (`num_prompts` defaults to 5x this) |

## Accuracy

`benchmark: accuracy` rows start the same `vllm serve --omni` server, generate an
image through `/v1/images/generations`, and score it against the prompt with
CLIPScore (`openai/clip-vit-base-patch32`). The evaluation settings (prompt,
resolution, steps, seed, `clip_score_threshold`) come from `configs/acc.yaml`:
task-level defaults keyed by `hf_pipeline_tag`, then per-model overrides. The run
fails if the score is below the threshold (default 0.30).

```bash
./run.sh --model_repo black-forest-labs/FLUX.1-dev --benchmark accuracy
```

## Metrics

Written to `perf_<model>.csv`: `latency_mean`, `latency_median`, `latency_p99` (s),
`throughput_qps` (req/s) and `output_throughput_ops` (outputs/s — equals
`throughput_qps` when `num_images: 1`).
