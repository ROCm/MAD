# vllm_dissag consolidation — before/after test plan

Goal: prove the unified `vllm_disagg.sh` produces **the same behavior** as the three legacy launchers
it replaced (`vllm_disagg_server.sh`, `vllm_disagg_mori_ep.sh`, `vllm_disagg_server_deepep.sh`), and
that the one genuinely-new path (moriio + TP) works.

There are two layers of testing:
1. **Offline checks** (no GPUs) — validate the assembled `vllm serve` argv + combo gate.
2. **Live cluster** (GPUs) — proves the launched servers actually serve + benchmark at par.

---

## 0. What "at par" means

| Cell | CONNECTOR / WIDE_EP / EP_BACKEND | Legacy equivalent | At-par check |
|------|----------------------------------|-------------------|--------------|
| 1 | `rixl` / 0 / — | `vllm_disagg_server.sh` | live boot+bench |
| 2 | `moriio` / 0 / — | none (NEW) | live boot+bench |
| 3 | `moriio` / 1 / `mori` | `vllm_disagg_mori_ep.sh` | live ITL vs baseline |
| 4 | `rixl` / 1 / `deepep` | `vllm_disagg_server_deepep.sh` | live boot+bench |

---

## 1. Offline checks (run now, no GPUs)

```bash
cd MAD/scripts/vllm_dissag
bash tests/run_all.sh             # gate_check + argv_assert; expect ALL OFFLINE SUITES PASSED
```

- `gate_check.sh` — combo validation: valid/invalid connector × WIDE_EP pairings, back-compat shims,
  invalid cross-pairings (`moriio`+`deepep`, `rixl`+`mori`) abort.
- `argv_assert.sh` — per-cell `vllm serve` flag/env assertions from the driver's `DRY_RUN=1` output
  (e.g. moriio+TP has `--tensor-parallel-size` + `MoRIIOConnector` and no `--enable-expert-parallel`;
  wideEP cells have `--enable-expert-parallel` + `--data-parallel-size`).

Static checks:
```bash
for f in vllm_disagg.sh parallelism.sh connectors/*.sh run_xPyD_models.slurm; do
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

## 2b. MI308X + Thor2 acceptance ladder (non-SLURM)

Bringing a new *platform* up is a different problem from proving a launcher refactor is at
par. There is no legacy launcher to A/B against, several axes interact, and the expensive
configurations are the ones you least want to debug blind. This ladder is the procedure that
was used for MI308X + Broadcom Thor2 and is the one to repeat on the next platform.

**Two rules make it cheap.** *One axis at a time* — move a single knob per rung, so a
regression has one candidate cause. *Cheap gate before expensive confirmation* — EP8 is two
nodes and quick; EP16 is four nodes at 30–45 min per configuration. Settle a knob at EP8 and
carry only the winner up.

Constant unless the rung names it as the variable: `FABRIC_PROFILE=thor2`,
`GPU_MEMORY_UTILIZATION=0.72` at EP16, prefill `--max-num-batched-tokens` above the longest
prompt in the sweep.

| # | Rung | Topology | Variable | Gate |
|---|---|---|---|---|
| 0 | `diag/run_ep_probe.sh` `NDEV=8` | 2-node | — | 0 QP errors. 40 s, versus finding out 40 min into a serve |
| 1 | Serve sanity | EP8 (1P/1D) | — | both roles serve, `/health` OK, coherent completion |
| 2 | NIAH baseline | EP8 | — | ≥ 9.3 mean at every rung 2k–35k — proves the synced tree is sound |
| 3 | Perf baseline | EP8 | — | TPOT flat to con=128 |
| 4 | Cudagraph gate | EP8 | `DECODE_CUDAGRAPH_MODE` | must **boot**, then beat #3 TPOT, then match #2 NIAH |
| 5 | Bring-up | EP16 (2P/2D) | winner of #4 | serves; symmetric heap allocated; no OOM |
| 6 | Decode batched-tokens | EP16 | `GLM_DECODE_BATCHED_TOKENS` | ≥ 3 reps each; report the **median** |
| 7 | NIAH final | EP16 | winner of #6 | the number that ships |
| 8 | Perf final | EP16 | winner of #6 | all three shapes: 1024/1024, 8192/1024, 1024/8192 |
| 9 | Config-plumbing guard | either | — | `DRY_RUN=1`: profile env loads **twice**; profile values reach the container; `FABRIC_PROFILE=-` is byte-identical to base |

### Notes that cost a run to learn

- **On a cudagraph-mode change the real gate is capturability, not latency.** A wrongly
  captured graph produces plausible garbage that a TPOT number will never reveal, which is
  why rung 4 re-runs NIAH rather than just comparing milliseconds.
- **Never benchmark a deployment that is still serving something else.** A second resident
  load generator manufactured a fake regression once. Before every scored run,
  `pgrep -f "vllm bench serve"` must be empty.
- **Treat non-monotonic throughput as a harness bug until proven otherwise.**
- **Score from the per-concurrency benchmark log, not the summary CSV.** The CSV has been
  observed to drop a measured row silently and to report *total* rather than *output*
  throughput. Anchor on the `[RUNNING] prompts N isl … con N` blocks; warm-up blocks are not
  preceded by one and must not be scored.
- **Read the logs on the node that wrote them.** The container is removed at sweep end, and
  rank 0 holds relayed copies of every other node's log — globbing per-node logs across
  hosts double-counts. Dedupe on rank, not on host.
- **A `0 matches` grep looks exactly like a real failure.** During rung 9, a quoting mismatch
  in the check (`KEY='104'` vs the emitted `KEY=104`) reported a passing config as broken.
  Confirm against the raw `docker run` line before believing a negative.

## 3. Sign-off checklist

- [ ] `tests/run_all.sh` green (offline: gate_check + argv_assert)
- [ ] static checks (`bash -n`, yaml) green
- [ ] back-compat: `RUN_MORI=1` / `RUN_DEEPEP=1` argv-equal to new axes
- [ ] Row 1 rixl-TP live: boots + benchmarks
- [ ] Row 3 moriio-wideEP live: ITL within ~5% of baseline (vs OLD launcher A/B)
- [ ] Row 4 deepep live: boots + benchmarks
- [ ] Row 2 moriio-TP live: boots + benchmarks (NEW capability)

**Per platform (§2b), additionally:**

- [ ] rung 0 EP probe clean before any serve attempt
- [ ] rungs 2 and 7 NIAH: EP16 **no worse than** EP8 — the cross-node combine is a *silent*
      corruption class, so this is the correctness canary, not a nice-to-have
- [ ] rung 9 config-plumbing guard green
- [ ] health on every scored run: 0 preemptions, 0 memory-access faults, 0 kernel asserts,
      0 engine deaths
