# ATOM vLLM Plugin Benchmarking Scripts

This directory contains benchmarking scripts for running vLLM with the ATOM vLLM plugin backend. The scripts use standard `vllm serve` and `vllm bench` commands but are configured to run inside the ATOM docker container.

## Overview

The ATOM vLLM plugin backend maintains compatibility with upstream vLLM's CLI, server APIs, and usage patterns. For detailed information about the plugin, see the [ATOM vLLM recipes](https://github.com/ROCm/ATOM/tree/main/recipes/atom_vllm).

## Directory Structure

```
atom_vllm/
├── run.sh               # Main execution script
├── run_vllm.py          # Python benchmark orchestrator
├── models.json          # Model configuration for MAD system
├── vllm_json_to_csv.py  # Utility for converting JSON results to CSV
├── profiling/           # --profile helpers (rocm-trace-lite)
│   ├── kernel_summary_payload.py  # Aggregates rtl_trace.db into a kernel summary JSON
│   ├── roctx_phase_markers.py     # roctx prefill/decode/mixed markers for the vLLM worker
│   └── sitecustomize.py           # Installs the markers in vLLM's spawned processes
└── configs/             # Benchmark configuration files
    ├── default.yaml     # Standard serving benchmarks (1k/1k, mc1-128)
    ├── accuracy.yaml    # Accuracy validation (GSM8K)
    ├── unittest.yaml    # Unit test configurations
    ├── extended.yaml    # Extended benchmarks with multiple I/O lengths
    └── perf.yaml        # Performance tuning overrides
```

## Usage

### Basic Usage

```bash
./run.sh --model_repo <model-name> [OPTIONS]
```

### Options

- `--model_repo <repo>` - HuggingFace model repository (required)
- `--config <yaml>` - Config file path (default: `configs/default.yaml`)
- `--benchmark <name>` - Benchmark type: `latency`, `throughput`, `serving`, `accuracy`, or `all` (default: `all`)
- `--output_csv <csv>` - Output CSV filename (currently ignored by `run.sh`, which always writes `perf_<model>.csv`; use `MAD_OUTPUT_CSV` instead)
- `--perf <yaml>` - Performance overrides file (default: auto-detect `configs/perf.yaml`)
- `--perf-output <yaml>` - Merged config output path
- `--no-perf-merge` - Disable auto-merge of perf.yaml
- `--profile` - Profile the serving benchmark instead of measuring it (see [Profiling](#profiling))

### Environment Variables

- `MAD_OUTPUT_CSV` - Override output CSV filename (default: `perf_<model>.csv`, moved to the parent directory)
- `MAD_SECRETS_HFTOKEN` - HuggingFace access token for gated models; exported as `HF_TOKEN`
- `CHECK_LOCAL_DATA` - Set to `true` to load weights from `MAD_DATAHOME` (default: `false`)
- `MAD_DATAHOME` - Local weights root; used only when `CHECK_LOCAL_DATA=true` and `$MAD_DATAHOME/<model_repo>` exists
- `PREDOWNLOAD_TIMEOUT_SEC` - Per-attempt timeout for the HuggingFace pre-download (default: `3600`)
- `MAD_SYSTEM_GPU_ARCHITECTURE` - GPU architecture for config overrides
- `BENCHMARK` / `CONFIG` - Same as `--benchmark` / `--config`; if set, they take precedence over the flags

## Examples

### Run Llama 3.1 8B serving benchmark

```bash
./run.sh --model_repo meta-llama/Llama-3.1-8B-Instruct --benchmark serving
```

### Run accuracy validation

```bash
./run.sh --model_repo meta-llama/Llama-3.1-8B-Instruct --config configs/accuracy.yaml
```

### Run with custom config

```bash
./run.sh --model_repo meta-llama/Llama-3.3-70B-Instruct --config configs/extended.yaml
```

### MAD entries

`models.json` defines one entry per model (`pyt_atom_vllm_*`), each passing `--model_repo` and `--config`.
It also defines a generic `atom-vllm-perf` entry with empty `args` that writes `perf_auto.csv`. It is meant
to be driven by a dispatcher that supplies `--model_repo`, `--config` and `MAD_OUTPUT_CSV`; run on its own it
fails because no model is given. Unlike the `vllm`, `sglang` and `atom` scripts, this `run.sh` does not infer the
benchmark from the tag name, so it runs `all` unless `BENCHMARK` or `--benchmark` is set.

## Benchmarks

### Latency
Measures per-token latency with fixed batch sizes using `vllm bench latency`.

### Throughput
Measures overall throughput (tokens/sec) using `vllm bench throughput`.

### Serving
Full serving benchmark with concurrent requests using `vllm bench serve`.
- Metrics: throughput (total/generation), TTFT, TPOT, ITL, E2EL

### Accuracy
GSM8K accuracy validation using `lm_eval` with the vLLM server.

## Configuration Files

### default.yaml
Standard serving benchmarks with:
- Input/output length: 1024/1024
- Concurrency levels: 1, 8, 32 (8B); 1, 8, 32, 128 (70B, 405B)
- Models: Llama 3.1/3.3 (8B, 70B, 405B BF16 and FP8)

Rows only define the run shape (model, tp, lengths, concurrency). Server flags
and env vars come from `perf.yaml`.

### accuracy.yaml
Accuracy validation configurations for Llama models with GSM8K evaluation.

### unittest.yaml
Quick unit test configurations with:
- Dummy weights (`--load-format dummy`)
- Smaller problem sizes
- All benchmark types

### extended.yaml
Extended configurations with multiple input/output length combinations:
- Input lengths: 128, 512, 1024, 2048
- Output lengths: 128, 512, 1024

Like `default.yaml`, rows carry no `extra_args` or `env`; those come from `perf.yaml`.

### perf.yaml
Per-model server flags and env vars, gap-filled into the base config rows whose
`model:` shares a repo id with the perf entry:
- Model-specific environment variables (e.g., `ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION`)
- Server flags (e.g., `--async-scheduling`, `--max-model-len`, `--max-num-batched-tokens`)
- KV cache settings (`--kv-cache-dtype fp8`)

Model-specific tuning belongs here, not in `default.yaml` / `extended.yaml`.

## Key Differences from vllm/ Directory

1. **Docker Backend**: Uses ATOM docker (`dockerfile: ../../docker/atom_vllm`)
2. **vLLM Commands**: Scripts use standard `vllm serve` and `vllm bench` commands
3. **Plugin Configuration**: `perf.yaml` carries ATOM-specific tuning, such as:
   - `--async-scheduling`
   - `--compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}'` (non-Llama models)
   - Environment variables like `ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION`
4. **Models**: Optimized for Llama family models with ATOM backend

## Profiling

`--profile` installs [rocm-trace-lite](https://github.com/amathews-amd/rocm-trace-lite) and, for `serving`
rows, replaces the benchmark with a single profiling pass: `vllm serve` runs under `rtl trace --mode full`
with roctx prefill/decode/mixed phase markers from `profiling/`. Each row produces:
- `profile_<model>_<tp>/rtl_trace.db` - GPU kernel trace with the phase markers
- `profile_<model>_<tp>/server_profile.log` - Server log for the profiling run
- `profile/kernel_summary_payload.json` - Kernel summary aggregated from the trace

No performance numbers are produced, so no perf CSV is written and `run.sh` skips the CSV move.

## Output

Results are written to CSV files with the following columns:
- hf_pipeline_tag, model, benchmark, tp, inp, out, dtype
- num_prompts, max_concurrency, bs
- cmd, performance, metric, unit

## Notes

- Authentication: Under MAD, set `MAD_SECRETS_HFTOKEN`; when running `run.sh` directly, set `HF_TOKEN` or use `huggingface-cli login`
- The scripts automatically download models from HuggingFace unless local weights are used (see `CHECK_LOCAL_DATA`)
- Performance overrides from `perf.yaml` are gap-filled (base config wins on conflicts)
- Architecture-specific overrides are applied based on `MAD_SYSTEM_GPU_ARCHITECTURE`

## References

- ATOM vLLM Plugin Guide: https://github.com/ROCm/ATOM/tree/main/recipes/atom_vllm
- Llama 3.1 Recipe: https://github.com/ROCm/ATOM/blob/main/recipes/atom_vllm/Llama.md
- vLLM Documentation: https://docs.vllm.ai/
