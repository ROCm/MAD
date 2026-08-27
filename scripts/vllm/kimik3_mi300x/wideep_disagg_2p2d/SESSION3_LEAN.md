# Session 3 — Lean production image & upstream integration

Goal: ship a **production** Kimi-K3 MI300X 2P/2D disagg image and MAD launcher path **without**
PR #193's 33 runtime patchers. Recipe = config only (`models.yaml` + `run_2p2d*.sh` / `vllm_dissag`).

## Build (lean)

```bash
cd scripts/vllm/kimik3_mi300x/wideep_disagg_2p2d/
GH_TOKEN=$(gh auth token) ./build_lean.sh
# -> kimik3-wideep-disagg-lean:latest  (WITH_NIXL=0, VLLM_REF=v3)
```

Same Dockerfile as investigation (`Dockerfile.kimik3_disagg`); lean = `WITH_NIXL=0` + pinned `VLLM_REF`.

| Build | `WITH_NIXL` | Connectors | Use |
|-------|-------------|------------|-----|
| Investigation (default) | 1 | moriio + rixl + DeepEP | full MAD matrix |
| **Session 3 lean** | **0** | **moriio / MoRI-EP only** | K3 2P/2D production |

## What's baked in (no runtime patchers)

`VLLM_REF=kimi-k3-wideep-disagg-fullsource-v3` (`raviguptaamd/vllm`) includes:

| Fix | Was patcher | Status in v3 image |
|-----|-------------|-------------------|
| 4-KV-group block routing | `apply_kimik3_moriio_group_routing.py` | baked |
| Multi-chunk prefill transfer | `apply_kimik3_chunk_gate_fix.py` + chunked allgrp | baked |
| KDA gather sync-free (>500K) | `apply_kimik3_kda_gather_nosync.py` | baked (v3 vs v2) |
| mamba block-id / remote_tp / N−1 | various | baked in connector branch |

Runtime patchers in PR #193 should **no-op** against v3 (idempotent detect). Do not bind-mount `/patchers` for production.

## MAD launcher alignment (done / todo)

| Item | Status | Notes |
|------|--------|-------|
| `vllm_dissag` W1–W5 taxonomy + Slurm guards | **done** | job 223124 F17/F26 |
| `--quantization-config` JSON tokenization | **done** | F24 `_model_config_to_array` |
| JIT cache prefill/decode split (`run_interactive.sh`) | **done** | F25; matches slurm |
| Kimi decode `DECODE_CUDAGRAPH_MODE=NONE` | **done** | matches validated standalone launch |
| PIECEWISE decode cudagraph | **open** | hangs at capture 5/9 — fix in vLLM/image (F25) |
| Lean image default in docs/CI | **this file** | `build_lean.sh` |

## Upstream PR checklist (MAD)

1. **Recipes** — `scripts/vllm/kimik3_mi300x/` (config + README + RESULTS); no patchers in tree.
2. **Launcher** — `scripts/vllm_dissag/` Kimi entry in `models.yaml`, moriio K3 topology, tests.
3. **Docker** — document `build_lean.sh`; do **not** duplicate Dockerfile layers in-repo.
4. **vLLM sidecar PR** — `raviguptaamd/vllm` branch `kimi-k3-wideep-disagg-fullsource-v3` (or upstream cherry-picks).
5. **Validation** — 2P/2D short NIAH 9/9 + MAD live smoke (F18/F26); extended ctx deferred (TODO-P3).

## Known open items (not Session 3 blockers)

- **PIECEWISE decode cudagraph** — ITL win; blocked on capture hang (F25). Standalone launch defaults `DECODE_CG=NONE`.
- **Extended NIAH 500K–900K** — KV headroom @ TP2×DP8 (F23); TODO-P3.
- **Throughput vs PR targets** — F21; tuning separate from image lean.

## Diff target

Session 3 MAD PR should add **launcher + yaml + docs** (<2K lines). Image build stays in recipe folder;
binary produced out-of-band via `build_lean.sh` and pushed to operator registry.
