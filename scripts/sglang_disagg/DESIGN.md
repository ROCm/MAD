# sglang_disagg — Design & Responsibility Overview

## What this module is

`scripts/sglang_disagg/` is a **benchmarking harness for SGLang Prefill/Decode (PD)
disaggregated inference on AMD (ROCm) GPUs**, orchestrated over a **SLURM** cluster.
It launches separate prefill and decode SGLang servers across nodes, fronts them with
a router/proxy, runs a concurrency-sweep benchmark, and parses the results into CSV.

The code here is almost entirely **bash orchestration + small Python utilities** — it
wraps SGLang (`sglang.launch_server`, `sglang_router`, `sglang.bench_serving`) rather
than implementing the inference itself.

> For a full breakdown of every environment variable, CLI parameter, and config value
> grouped by responsibility (cluster / framework / connectors / parser / benchmarking /
> launcher / model), see [CONFIG.md](CONFIG.md).

## Layered structure

```
Build layer      docker/sglang_disagg_inference.ubuntu.amd.Dockerfile   (SGLang + MoRI image)
                             │
Cluster layer    run_xPyD_models.slurm     (sbatch: node selection, IP discovery, docker run per node)
                             │
Entrypoint       sglang_disagg_mori_io_ep.sh   (per-node role dispatch: prefill / decode / router)
   layer                     │
Config layer     models.yaml   mori_ep_env.sh   (model flags; RDMA/NCCL/MoRI env)
                             │
Sync layer       socket_barrier.py   socket_wait.py   (cross-node rendezvous / shutdown wait)
                             │
Benchmark        benchmark_xPyD.sh   (bench_serving concurrency sweep)
   layer                     │
Parse layer      parse_to_csv.py   benchmark_parser.py   (logs → CSV / perf.csv)
```

## File responsibilities

| File | Responsibility |
|------|----------------|
| `docker/sglang_disagg_inference.ubuntu.amd.Dockerfile` | Build the runtime image: SGLang ROCm base + optional MoRI (pinned commit) + `sglang-router`. |
| `run_xPyD_models.slurm` | Cluster orchestration: validate `MODEL_NAME`/`DP_MODE`, choose KV backend (`RUN_MORI`), select `xP+yD` nodes, discover IPs, `docker run` the entrypoint on every node. |
| `sglang_disagg_mori_io_ep.sh` | Per-node container entrypoint. Derives `PARALLEL_MODE` (tp/dp), computes TP/DP/EP sizes, loads model flags from YAML, and dispatches by `NODE_RANK` into prefill / decode / router roles. |
| `models.yaml` | Per-model CLI flags: `base_flags`, `tp_flags`/`dp_flags`, and role-specific `prefill`/`decode` flags. Backend-agnostic. |
| `mori_ep_env.sh` | NCCL / RDMA / MoRI environment variables, `IB_DEVICES` (NIC selection). |
| `set_env_vars.sh` | Legacy NCCL/IB env (used only by the legacy `sglang_disagg_server.sh`). |
| `socket_barrier.py` | Startup rendezvous — opens a local port and waits until all nodes have opened theirs. |
| `socket_wait.py` | Lifecycle wait — blocks while a remote port (router :2322) stays open; returns when it closes. |
| `benchmark_xPyD.sh` | Concurrency-sweep benchmark driven through the router via `sglang.bench_serving`. |
| `parse_to_csv.py` | Production parser: benchmark log → results CSV + madengine `perf.csv`. |
| `benchmark_parser.py` | Standalone richer diagnostic parser (pandas; latency + throughput metrics). Not wired into the pipeline. |
| `sglang_disagg_server.sh` | Legacy Mooncake-only launcher (bash assoc-array configs, port 2322 servers, no MoRI/DP). Superseded by `sglang_disagg_mori_io_ep.sh`. |
| `salloc_launch.sh` | Example `salloc`/`sbatch` invocation commands. |

## Responsibility diagram

```mermaid
graph TD
    subgraph User["User / Operator"]
        SL[salloc_launch.sh<br/>example invocations]
    end

    subgraph Build["Build-time"]
        DF["Dockerfile<br/>base SGLang ROCm image<br/>+ optional MoRI + sglang-router"]
    end

    subgraph Cluster["Cluster orchestration (runs on login node)"]
        SLURM["run_xPyD_models.slurm<br/>• validate MODEL_NAME / DP_MODE allowlist<br/>• pick RUN_FILE + KV backend (mori/mooncake)<br/>• select xP+yD nodes, discover IPs<br/>• export env, docker run on every node"]
    end

    subgraph Node["Per-node container entrypoint"]
        ENTRY["sglang_disagg_mori_io_ep.sh<br/>• derive PARALLEL_MODE (tp/dp) from DP_MODE<br/>• compute TP/DP/EP sizes<br/>• load model flags from YAML<br/>• dispatch by NODE_RANK"]
        R0["NODE_RANK 0<br/>Prefill#0 + Router/proxy<br/>(co-located)"]
        RP["NODE_RANK 1..xP-1<br/>Prefill workers"]
        RD["NODE_RANK xP..xP+yD-1<br/>Decode workers"]
    end

    subgraph Config["Configuration"]
        YAML["models.yaml<br/>base/tp/dp + prefill/decode flags"]
        ENV["mori_ep_env.sh<br/>NCCL/RDMA/MoRI env, IB_DEVICES"]
    end

    subgraph Sync["Cross-node synchronization"]
        BAR["socket_barrier.py<br/>wait until all nodes open a port"]
        WAIT["socket_wait.py<br/>block until proxy port closes"]
    end

    subgraph SGLang["SGLang processes (external)"]
        PSRV["sglang.launch_server<br/>--disaggregation-mode prefill"]
        DSRV["sglang.launch_server<br/>--disaggregation-mode decode"]
        ROUTER["sglang_router.launch_router<br/>--pd-disaggregation :2322"]
    end

    subgraph Bench["Benchmark + parsing"]
        BENCH["benchmark_xPyD.sh<br/>concurrency sweep via bench_serving"]
        P2C["parse_to_csv.py<br/>→ results.csv + madengine perf.csv"]
        BP["benchmark_parser.py<br/>→ rich metrics table/CSV"]
    end

    SL --> SLURM
    DF -.provides image.-> SLURM
    SLURM -->|docker run + env per node| ENTRY
    ENTRY --> R0 & RP & RD
    ENTRY -.reads.-> YAML
    ENTRY -.sources.-> ENV

    R0 --> PSRV
    R0 --> ROUTER
    RP --> PSRV
    RD --> DSRV

    ENTRY --> BAR
    RP --> WAIT
    RD --> WAIT

    ROUTER -->|routes /generate, /v1/completions| PSRV
    ROUTER --> DSRV
    PSRV -.KV cache transfer RDMA.-> DSRV

    R0 --> BENCH
    BENCH -->|HTTP :2322| ROUTER
    BENCH --> P2C
    BENCH -. optional .-> BP
```

## Runtime sequence (one job)

```mermaid
sequenceDiagram
    participant U as User
    participant S as run_xPyD_models.slurm
    participant N0 as NODE 0 (Prefill+Router)
    participant NP as Prefill workers
    participant ND as Decode workers
    participant B as benchmark_xPyD.sh

    U->>S: sbatch (xP, yD, MODEL_NAME, RUN_MORI)
    S->>S: validate model, select nodes, discover IPs
    S->>N0: docker run entrypoint (NODE_RANK=0)
    S->>NP: docker run entrypoint (NODE_RANK 1..xP-1)
    S->>ND: docker run entrypoint (NODE_RANK xP..)
    Note over N0,ND: socket_barrier.py — all containers rendezvous
    N0->>N0: launch prefill server
    NP->>NP: launch prefill servers
    ND->>ND: launch decode servers
    N0->>N0: poll logs for "server is fired up and ready"
    N0->>N0: launch sglang_router (:2322)
    N0->>N0: probe /v1/completions until PD path ready
    N0->>B: run benchmark (concurrency sweep)
    B->>N0: bench_serving → router → prefill+decode
    B->>B: parse_to_csv.py → CSV + perf.csv
    N0->>N0: kill router + prefill
    NP-->>NP: socket_wait detects router closed → exit
    ND-->>ND: socket_wait detects router closed → exit
```

## Port reference

| Purpose | Modern launcher (`sglang_disagg_mori_io_ep.sh`) | Legacy launcher (`sglang_disagg_server.sh`) |
|---------|--------------------------------------------------|----------------------------------------------|
| Prefill / decode server | `3000` | `2322` |
| Router / proxy | `2322` | `2322` |
| `dist-init` (DP_MODE=1 rendezvous) | `5757` (`DIST_INIT_PORT`) | n/a |
| Startup barrier | `BARRIER_PORT` (default `4342`) | hardcoded `4342` |

## Key design decisions

- **Single unified launcher, two KV backends.** `run_xPyD_models.slurm` always uses
  `sglang_disagg_mori_io_ep.sh`; `RUN_MORI` toggles `--disaggregation-transfer-backend`
  between `mori` (RDMA-based MoRI IO) and `mooncake`. The backend is deliberately kept
  out of `models.yaml` so model configs stay backend-agnostic.
- **Two parallelism modes via `DP_MODE`.** `DP_MODE=0` → TP-only (all models);
  `DP_MODE=1` → DP-attention + MoRI expert parallelism, restricted by an allowlist to
  DeepSeek-V3/R1 (enforced redundantly in both the slurm script and the entrypoint).
- **Router co-located on prefill NODE 0** — no dedicated proxy node. IP layout
  convention: `IP[0..xP-1]` = prefill, `IP[xP..]` = decode. Nodes are alphabetically
  sorted so `srun` PROCID ordering matches the IP array.
- **File-based + socket-based synchronization.** Readiness is detected by grepping
  worker logs on a shared `/run_logs` mount for
  `"The server is fired up and ready to roll!"`; lifecycle coordination uses
  `socket_barrier.py` (startup rendezvous) and `socket_wait.py` (workers block until the
  router's port 2322 closes, then self-terminate). DP ranks are intentionally *not*
  gated on the dist-init port because torch rendezvous would otherwise deadlock.
- **Two parsers with different scopes.** `parse_to_csv.py` is the production one wired
  into `benchmark_xPyD.sh` (max-throughput CSV + madengine `perf.csv`).
  `benchmark_parser.py` is a richer standalone diagnostic (pandas, latency percentiles)
  not called by the pipeline.
- **`sglang_disagg_server.sh` is legacy** (Mooncake-only, superseded by the YAML-driven
  MoRI/EP launcher).
