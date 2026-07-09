# Qwen3.5-35B-A3B serving/benchmark harness — AMD Instinct MI350 (gfx950)

Aggregated serving (3B-active MoE fits one GPU — **no disagg**). Scales 1→N nodes by editing one
file. Core knob: **(TP, DP)** replicas per node.

## ⭐ RECOMMENDED DEFAULT — SGLang · MXFP4 · TP1 (8 instances/node)

> **Serve Qwen3.5-35B-A3B as 8 independent single-GPU MXFP4 instances per node, on SGLang.**
> Benchmarking on MI350 (gfx950) found this the highest-throughput setup:
> **SGLang > vLLM · MXFP4 > FP8 (and ~½ the HBM) · TP1 > TP2 > TP4**.

```bash
cd Qwen3.5
# production serving — 8 single-GPU MXFP4 instances fill the node:
MODEL=qwen35-moe-mxfp4 TP=1 DP=8 ACTION=serve ./run_sglang.sh
# reproduce the benchmark sweep:
MODEL=qwen35-moe-mxfp4 TP=1 ACTION=sweep      ./run_sglang.sh
```
Why TP1 and not TP-shard: a 3B-active model fits one GPU, so tensor-parallel only adds cross-GPU
comm cost. Fill the node with independent instances instead. (ATOM excluded — crashes at ISL≥8192.)

### Quickstart (run the default in 4 steps)
```bash
# 1. Get the weights (MXFP4, ~23 GB) to a path visible on every node:
hf download amd/Qwen3.5-35B-A3B-MXFP4 --local-dir /path/to/models/Qwen3.5-35B-A3B-MXFP4

# 2. Edit cluster.yaml:  set `nodes:` to your node(s), `models_root: /path/to/models`,
#    and `srun_alloc:` to your reservation/jobid (or "" if nodes are free).

# 3. From inside a Slurm allocation on those nodes, run the default:
cd scripts/Qwen3.5
MODEL=qwen35-moe-mxfp4 TP=1 DP=8 ACTION=serve ./run_sglang.sh   # 8 MXFP4 instances/node

# 4. Query any instance (OpenAI-compatible, ports 8000..8007):
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"/path/to/models/Qwen3.5-35B-A3B-MXFP4",
       "messages":[{"role":"user","content":"Hello"}],"max_tokens":256}'
```
`ACTION=sweep` instead of `serve` reproduces the benchmark. Notes: it's a **reasoning model** (allow
generous `max_tokens`); the SGLang image + all flags are preset in `model.yaml` — nothing else to set.

## Design in one picture
```
Each node (8 GPUs):  DP replicas, each TP-sharded.   DP × TP ≤ 8.   Full util ⇔ DP×TP = 8.
  TP1→DP8 (⭐ best)   TP2→DP4   TP4→DP2
N nodes = N × (8//TP) independent replicas. No inter-node TP. No disagg.
Scale 1→N nodes = add node names to cluster.yaml. Nothing else changes.
```

## Config layering (set a knob ONCE, applied to all engines)
- `defaults:` block in `model.yaml` holds shared logical knobs (`max_model_len`, `gpu_memory_util`,
  `max_num_seqs`). `lib/replica_entry.sh` translates each to the per-engine flag name
  (vLLM/ATOM `--gpu-memory-utilization`, SGLang `--mem-fraction-static`; etc.).
- engine `serve_flags` hold ONLY engine-unique flags. Per-model override > global defaults.

## Files
| File | Role |
|------|------|
| `cluster.yaml` | **SCALE LEVER** — node list, gpus_per_node, reservation, paths, port base |
| `model.yaml` | model defs + shared `defaults{}` + per-engine image/flags/fixes |
| `run_vllm.sh` / `run_sglang.sh` / `run_atom.sh` | thin wrappers → `lib/run_engine.sh` |
| `lib/run_engine.sh` | **single source of truth** launcher: yaml → clean GPUs → place → launch. Two bench paths (see below) |
| `lib/clean_node.sh` | pre-flight GPU clean-state guard (kill ALL containers incl. zombie VRAM, verify free) |
| `lib/placement.py` | (TP,DP,nodes) → one line per replica (node, gpus, port, idx); `--nodes` override |
| `lib/replica_entry.sh` | runs INSIDE each container: start server (clean env) → health-wait → sanity/serve/sweep |
| `lib/check_accuracy.py` + `lib/prompts.json` | 5-prompt accuracy gate |
| `lib/cfg.py` | yaml reader (cluster/model/engine/default lookups) |
| `lib/lib_inferencex.sh` | vendored InferenceX helpers (health-wait, gpu monitor, bench) |
| `utils/bench_serving/` | vendored InferenceX benchmark client (Apache-2.0, see NOTICE) |
| `sweep/sweep_config.sh` | **SUPERSEDED** — one-node ATOM reproducer only; logic folded into `run_engine.sh` |
| `patches/instantiator.py` | torch patch (kept for reference; NOT needed — see ATOM note) |

## Two benchmark paths in `run_engine.sh` (engine-dependent)
- **In-container** (vLLM, SGLang — `bench_external: false`): the bench client runs inside the engine
  container via `replica_entry.sh`. One backgrounded `--overlap` srun per replica; replicas run concurrently.
- **External / sibling** (ATOM — `bench_external: true`): ATOM's own image crashes the bench client, so the
  bench runs from a clean vLLM container. Server + bench are **sibling containers inside ONE `--exclusive`
  srun** per node (`launch_external_bench_node`). This avoids a deadlock the old design hit: a *second*
  `--overlap` bench srun could never schedule because the server srun (`--mem=0`) already claimed all node
  memory. One srun = one allocation = no contention. Server launched detached (`-d`) to avoid an empty-log
  startup race on the 75 GB image (poll real `docker inspect` state, not a `docker ps --filter` first-tick).

## Usage
Models (in `model.yaml`): `qwen35-moe-mxfp4` (⭐ recommended), `qwen35-moe-fp8` (baseline).
```bash
# RECOMMENDED — SGLang + MXFP4 + TP1 (8 instances fill the node)
MODEL=qwen35-moe-mxfp4 TP=1 DP=8 ACTION=serve ./run_sglang.sh
MODEL=qwen35-moe-mxfp4 TP=1 ACTION=sweep      ./run_sglang.sh   # benchmark

# other engines / precisions for comparison
MODEL=qwen35-moe-fp8   TP=1 ACTION=sweep ./run_vllm.sh
MODEL=qwen35-moe-mxfp4 TP=2 ACTION=sweep ./run_sglang.sh

# scale: edit cluster.yaml `nodes`, re-run same command. Target nodes: NODES=node-a,node-b ...
```
`ACTION=sweep`: clean GPUs → boot → accuracy gate → full perf sweep (shapes × conc).
`ACTION=serve`: boot + stay up.  `ACTION=sanity`: accuracy + one smoke point.

## Findings (MI350 / gfx950)

Benchmarked across 2 precisions × {SGLang, vLLM} × TP{1,2,4} × concurrency. Three consistent
findings drive the recommended default:
- **TP1 > TP2 > TP4** — a 3B-active MoE fits one GPU; fill the node with independent instances
  rather than tensor-parallel (TP only adds cross-GPU comm cost).
- **SGLang > vLLM** — higher throughput at matched config.
- **MXFP4 > FP8** — higher throughput AND ~half the HBM (MI355X has native MXFP4 hardware).

→ default = **SGLang · MXFP4 · TP1** (8 single-GPU instances per node).

**Correctness:** both engines serve MXFP4 correctly out-of-box (no garbling). Qwen3.5 is a
reasoning model — allow generous `max_tokens`, or use vLLM `--reasoning-parser qwen3`.

**Scope / caveats:**
- **ATOM excluded** — serves 35B MXFP4 at short context but crashes at ISL≥8192
  (`could not broadcast (513,) into (512,)`). Use SGLang or vLLM.
- Set `max_model_len ≥ ISL+OSL` for your workload (default 17408 covers 16384/1024).

## Discovered FIXES (baked into model.yaml / scripts)
- **Reasoning model**: Qwen3.5 emits a "Thinking Process" before the answer. Use generous
  `max_tokens` (accuracy gate uses 512), or vLLM `--reasoning-parser qwen3` to split reasoning/answer.
- **SGLang**: `--disable-radix-cache` REQUIRED (qwen3_5_moe is hybrid-GDN), else
  `AssertionError: extra_buffer needs CUDA/MUSA/NPU (FLA)`. Plus `--enable-aiter-allreduce-fusion
  --page-size 16` (AMD Day-0 recipe).
- **vLLM**: image ENTRYPOINT is `vllm` → override to `bash`; rolling tag `vllm/vllm-openai-rocm:nightly`
  (pinned nightlies get GC'd). Flags `--enable-expert-parallel --reasoning-parser qwen3`.
- **max_model_len ≥ ISL+OSL**: set to 17408 to cover 16384/1024 (else long-context requests are
  rejected as over-context → 0 throughput).
- **Weights on local scratch**: cache NFS→`/local_datasets` first (NFS cold-load adds ~7 min/boot).
- **All engines**: pre-flight `clean_node.sh` kills zombie VRAM from prior runs (the #1 OOM-at-init cause).

## NOT in scope (by design)
- **Disagg / KV-transfer**: irrelevant — 35B fits one GPU (see the `dsv4/` harness for disagg).
- **TP-sharding as the fill mechanism**: proven suboptimal here — use TP1 × N instances instead.
