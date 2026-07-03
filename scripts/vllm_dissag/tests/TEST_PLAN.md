# vllm_dissag consolidation — before/after test plan

Goal: prove the unified `vllm_disagg.sh` produces **the same behavior** as the three legacy launchers
it replaced (`vllm_disagg_server.sh`, `vllm_disagg_mori_ep.sh`, `vllm_disagg_server_deepep.sh`), and
that the one genuinely-new path (moriio + TP) works.

There are two layers of testing:
1. **Offline parity** (no GPUs) — proves the assembled `vllm serve` command line is byte-identical.
2. **Live cluster** (GPUs) — proves the launched servers actually serve + benchmark at par.

---

## 0. What "at par" means

| Cell | CONNECTOR / WIDE_EP / EP_BACKEND | Legacy equivalent | At-par check |
|------|----------------------------------|-------------------|--------------|
| 1 | `rixl` / 0 / — | `vllm_disagg_server.sh` | argv parity + live boot+bench |
| 2 | `moriio` / 0 / — | none (NEW) | smoke argv + live boot+bench only |
| 3 | `moriio` / 1 / `mori` | `vllm_disagg_mori_ep.sh` | argv parity + live ITL vs baseline |
| 4 | `rixl` / 1 / `deepep` | `vllm_disagg_server_deepep.sh` | argv parity + live boot+bench |

---

## 1. Offline parity (run now, no GPUs)

The committed gate compares the live driver's `DRY_RUN=1` argv against golden fixtures captured from
the legacy launchers (`tests/golden/`).

```bash
cd MAD/scripts/vllm_dissag
bash tests/parity_check.sh        # expect: ALL PARITY CELLS BYTE-IDENTICAL ✅, exit 0
```

This asserts, for every (connector × role) cell that has a legacy equivalent:
- prefill/decode × master/child for moriio-wideEP and rixl-deepep (8 cells)
- prefill/decode for rixl-TP (2 cells)
- moriio+TP smoke (has `--tensor-parallel-size` + `MoRIIOConnector`, no `--enable-expert-parallel`)
- invalid cross-pairings (`moriio`+`deepep`, `rixl`+`mori`) abort

Also run the static checks:
```bash
for f in vllm_disagg.sh parallelism.sh connectors/*.sh run_xPyD_models.slurm tests/parity_check.sh; do
  bash -n "$f" && echo "OK $f"; done
python3 -c "import yaml; yaml.safe_load(open('models.yaml')); print('yaml OK')"
```

### Back-compat equivalence (legacy flags)
Confirm the old flags resolve to the same argv as the new axes:
```bash
# RUN_MORI=1 must equal CONNECTOR=moriio WIDE_EP=1 EP_BACKEND=mori
diff <(RUN_MORI=1   DRY_RUN=1 NODE_RANK=0 MODEL_NAME=DeepSeek-V3 MODEL_PATH=/m NIXL_COOKBOOK_PATH=$PWD \
        xP=2 yD=2 IPADDRS=10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4 bash vllm_disagg.sh 2>/dev/null) \
     <(CONNECTOR=moriio WIDE_EP=1 EP_BACKEND=mori DRY_RUN=1 NODE_RANK=0 MODEL_NAME=DeepSeek-V3 MODEL_PATH=/m \
        NIXL_COOKBOOK_PATH=$PWD xP=2 yD=2 IPADDRS=10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4 bash vllm_disagg.sh 2>/dev/null) \
  && echo "RUN_MORI back-compat OK"
```

### If a legacy-equivalent change is intentional
Regenerate the goldens (only when you *mean* to change a reproduced cell), review the diff, commit:
```bash
bash tests/golden/gen_golden.sh
git diff tests/golden/        # inspect — should match your intended change
```

---

## 2. Live cluster validation (GPUs; run per matrix row)

Pre: set `DOCKER_IMAGE_NAME`, pick nodes. `num_nodes = xP + yD`. The proxy is co-located on rank 0.

For each row, a run is "at par / pass" when:
- all workers reach `Application startup complete.` within timeout,
- the rank-0 proxy answers the bring-up curl (moriio path) / barrier (rixl path),
- `benchmark_xPyD.sh` writes `/run_logs/<JOBID>/..._CONCURRENCY.log` with non-empty results, 0 failed,
- ITL median is within ~5% of the documented baseline for the EP cells.

### Row 1 — rixl + TP (1P/1D)
```bash
export DOCKER_IMAGE_NAME=<img>; export CONNECTOR=rixl WIDE_EP=0
export xP=1 yD=1 MODEL_NAME=DeepSeek-V3
sbatch -N 2 -n 2 --nodelist=<n0,n1> run_xPyD_models.slurm
```

### Row 2 — moriio + TP (1P/1D)  ← NEW, highest-risk
```bash
export DOCKER_IMAGE_NAME=<img>; export CONNECTOR=moriio WIDE_EP=0
export xP=1 yD=1 MODEL_NAME=DeepSeek-V3
sbatch -N 2 -n 2 --nodelist=<n0,n1> run_xPyD_models.slurm
```
Watch for: does `MoRIIOConnector` initialize without `--enable-expert-parallel`? Is `kv_parallel_size`
correct in TP? Does the moriio toy proxy / notify path assume DP ranks? If it wedges at connector init
or KV handshake, capture `prefill_NODE0.log` / `decode_NODE1.log` — this is the cell with no precedent.

### Row 3 — MoRI-EP wideEP (2P/2D)  ← compare ITL to baseline
```bash
export DOCKER_IMAGE_NAME=<img>; export CONNECTOR=moriio WIDE_EP=1   # or legacy: RUN_MORI=1
export xP=2 yD=2 MODEL_NAME=DeepSeek-V3
sbatch -N 4 -n 4 --nodelist=<n0,n1,n2,n3> run_xPyD_models.slurm
```
At-par bar: ITL median ~48–51 ms across 1024/1024, 4096/4096, 8192/1024 (per the recipe baselines).

### Row 4 — DeepEP wideEP (2P/2D)
```bash
export DOCKER_IMAGE_NAME=<img>; export CONNECTOR=rixl WIDE_EP=1 EP_BACKEND=deepep  # or legacy: RUN_DEEPEP=1
export xP=2 yD=2 MODEL_NAME=DeepSeek-V3
sbatch -N 4 -n 4 --nodelist=<n0,n1,n2,n3> run_xPyD_models.slurm
```

### A/B confirmation against the OLD launchers (strongest proof)
The legacy launchers were deleted on this branch. To A/B live, check out the parent commit (or
`develop`) in a second worktree and run the same MODEL/xP/yD there; compare the resulting
`_CONCURRENCY.log` medians. Identical-within-noise ITL/throughput = behavior preserved end-to-end.

---

## 3. Sign-off checklist

- [ ] `tests/parity_check.sh` green (offline, byte-identical)
- [ ] static checks (`bash -n`, yaml) green
- [ ] back-compat: `RUN_MORI=1` / `RUN_DEEPEP=1` argv-equal to new axes
- [ ] Row 1 rixl-TP live: boots + benchmarks
- [ ] Row 3 moriio-wideEP live: ITL within ~5% of baseline (vs OLD launcher A/B)
- [ ] Row 4 deepep live: boots + benchmarks
- [ ] Row 2 moriio-TP live: boots + benchmarks (NEW capability)
