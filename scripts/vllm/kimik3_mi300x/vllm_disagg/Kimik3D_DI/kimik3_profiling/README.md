# Kimi-K3 2P/2D disagg — ROCtx profiling addon

Self-contained profiling for the Kimi-K3 MXFP4 2P/2D wide-EP disagg serve. Rides on the
existing `../run_2p2d.sh` launcher — it does **not** import the DeepSeek `run_xPyD_models.slurm`
harness. Everything needed at capture time is vendored here; only the optional deep-analysis
step (`TraceLens`) is a `git clone`.

**Validated end-to-end on the real model (4 nodes / 32 GPU workers, 2026-09-07):** one profiled
2P/2D run + a completed request produced `combined_all.pftrace` = 32 lanes, 12.17M kernel events,
**1116 MoRIIO markers** (`mori.rdma.io_transfer bytes=… id=<write_uid>` on the prefill/producer
ranks), graph-mode decode captured, all shards on one shared dir with no gather step.

## What this produces
- `combined_all.pftrace` — one unified prefill+decode Perfetto trace (one lane per GPU worker).
- MoRIIO KV-transfer **markers** in that same trace when marker capture is on (see below):
  `mori.rdma.io_transfer bytes=<n> id=<write_uid>`, `mori.io.session_batch_write id=<n>`.
- `reqid_map.csv` — `write_uid,direction,request_id,transfer_id,layer` (join key from the
  `mori.rdma.io_transfer id=<write_uid>` marker ranges back to each request).
- per-kernel + by-category CSVs — kernel-type rollup (MORI EP / Communication / GEMM / MLA /
  MoE_Fused / …) via the built-in designator in `trace_tools.py`.

## Files (nothing else required for capture)
| File | Role |
|---|---|
| `trace_tools.py` | `combine` (unified trace) + `extract-reqid` + `categorize_kernel` (the kernel-type designator) + `buckets`/`trimmed-summary`. **stdlib only.** |
| `patch_moriio_reqid_map.py` | Idempotent source overlay: makes the MoRIIO connector emit `moriio_reqid_map` log lines. Gated on `MORIIO_REQID_MAP=1`. `--check` / `--revert`. |
| `hooks.sh` | Builds the `rocprofv3 …` capture prefix + fires the reqid patch. |
| `flush_rocprof.sh` | Drains rocprofv3 so it finalizes its shards (see "Ending capture"). |
| `external_copies/traceconv_bin/traceconv` | Vendored Perfetto x86-64 binary (nodes have no internet). Decodes `.pftrace`. |

## How to run (basic profiling)
From `Kimik3D_DI/`, add `RUN_PROFILE=1` to the normal serve invocation:
```
RUN_PROFILE=1 MORIIO_REQID_MAP=1 bash run_2p2d.sh <role> ...   # same args as a normal serve
```
`RUN_PROFILE=1` makes `run_2p2d.sh`:
1. clear stale `/dev/shm` + SysV IPC on the host **before** the container starts (under
   rocprofv3 the prefill DP worker-init is timing-sensitive; leftover shm/IPC from a prior
   serve causes a `WorkerProc` init crash on the producer role — this pre-clean fixes it);
2. mount `kimik3_profiling/` into the container and pass the profiling env;
3. inside the container: source `hooks.sh`, apply the reqid patch (if `MORIIO_REQID_MAP=1`),
   prefix `vllm serve` with `rocprofv3 <flags> --output-format pftrace csv json`;
4. write per-worker shards to `${ROCPROF_DIR_BASE:-/logs/rocprof}/<job>/rocprof_<role>_NODE<n>/`.

Markers are ON by default in the launcher (`ROCPROF_FLAGS=--kernel-trace --marker-trace`,
`MORI_ROCTX=1`, `MORI_ROCTX_TRANSFER=1`). See "Capturing markers" for the details.

### Ending capture — MUST flush before teardown
rocprofv3 finalizes its trace only when **all** the traced processes it wraps have exited.
Two traps this addon handles for you:
- **Do NOT `docker stop`** first — it SIGKILLs pid 1 after grace, killing rocprofv3 mid-write
  → truncated/empty shards (Exit 137, 0 files).
- **Do NOT signal only `vllm serve` (pid 20)** — on the prefill role its DP supervisor
  *respawns* the EngineCore/Worker children, so rocprofv3 never drains (hangs).

Run the flush helper inside each container **before** `docker stop`:
```
docker exec <container> bash /kimik3_profiling/flush_rocprof.sh
```
It SIGTERMs the **whole vLLM process group** (Worker + EngineCore + APIServer + DPCoordinator),
re-signalling each loop so the supervisor can't keep rocprofv3 blocked, while leaving pid 1 and
pid 20 alive so the **container stays Up** until rocprofv3 has written the shards. When it prints
`rocprofv3 finalize complete`, the shards are on disk and it is safe to `docker stop`/`rm`.
(Validated: this drained all 4 roles to full shards; the prefill producer ranks carried 279
markers each.)

### Post-run (offline, on any node with the shards)
Because the shard dir is on shared storage (see Storage), all roles' shards are already
co-located — no gather. Run `combine` once over the job dir:
```
python3 kimik3_profiling/trace_tools.py combine       <rocprof_dir>/<job>    # -> combined_all.pftrace
python3 kimik3_profiling/trace_tools.py extract-reqid  prefill_NODE*.log -o reqid_map.csv
python3 kimik3_profiling/trace_tools.py buckets --in <summary.csv> \
        --out-per-kernel per_kernel.csv --out-by-category by_category.csv    # kernel-type rollup
```

## Preflight (do this once per image before a capture session)
The reqid patch **fails closed** if the connector source drifts. Verify anchors first:
```
python3 kimik3_profiling/patch_moriio_reqid_map.py --check \
  --moriio-dir /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/moriio
```
`PATCHED`/`unpatched` = anchors OK. An "anchors did not match" error = source drifted; the
patch needs a one-line anchor nudge (no image rebuild).

## Storage — one shared dir, no gather
`ROCPROF_DIR_BASE` defaults to the container's `/logs/rocprof`, which the launcher bind-mounts
from the host's shared home (`~/k3disagg/logs`). On a shared cluster FS (WekaFS here — verified
same inode on all 4 nodes) every role writes its shards into the **same** directory, so no
rsync/gather is needed before `combine`. Budget ~4 GiB per role-node per run (~15 GiB for a
2P/2D run at 128/32). If your cluster FS is space-constrained, point `ROCPROF_DIR_BASE` at
node-local scratch instead and gather the four `rocprof_*_NODE*` dirs into one directory before
`combine`.

## Capturing MoRI ROCtx markers (validated working)
Markers are the MoRIIO **host RDMA I/O** ranges, compiled into `libmori_io.so` in the recipe
image (`MORI_REF=624002c8`) and env-gated (default OFF; the launcher turns them ON). They fire
on the process that posts the RDMA transfer — in write-mode disagg that is the **prefill /
kv_producer** side, and only once a real KV transfer runs (i.e. after ≥1 completed request).
So: to see markers you need a healthy prefill pool + a completed request, then flush and inspect
the **prefill** shards (decode/consumer shards legitimately show 0 write-markers).

Launcher defaults (already set in `run_2p2d.sh`):
```
ROCPROF_FLAGS="--kernel-trace --marker-trace"  MORI_ROCTX=1  MORI_ROCTX_TRANSFER=1
```
- `MORI_ROCTX=1` → synchronous host-post ranges (`mori.io.engine_batch_write`,
  `mori.rdma.batch_post.{write,read}`).
- `MORI_ROCTX_TRANSFER=1` → async post-to-completion ranges (`mori.rdma.io_transfer`,
  carries `bytes=` + `id=<write_uid>` — the reqid join key).

Verify markers landed: in a role's `*_results.json`,
`rocprofiler-sdk-tool[0].buffer_records.marker_api` length > 0 (prefill producer ranks) and
`.strings.marker_api` holds the `mori.rdma.io_transfer …` names. A **kernel-only** run
(`ROCPROF_FLAGS=--kernel-trace`) shows `marker_events=0` — that's expected, not a bug.

## Kernel-category rollup: use the summarized path
`categorize_kernel` (the designator) runs on a **TraceLens kernel-summary** CSV (needs a
`duration_sum` column), NOT the raw rocprofv3 `*_kernel_trace.csv` (which has per-event
`Start/End_Timestamp`). So the by-category table comes from `trimmed-summary`/`analyze`
(the TraceLens tier), not `buckets` on the raw CSV. The raw kernels still combine fine
into `combined_all.pftrace`.

## Notes / known gaps
- **KDA kernels → `Other`.** The designator labels the 24 MLA full-attn layers and MoRI-EP all2all
  natively; K3's 69 KDA recurrent-layer kernels currently fall to `Other`. Add a `KDA` bucket to
  `categorize_kernel()` once the first trace shows their real kernel names (2-line change).
- **Deep analysis (optional).** The by-category charts / `.pftrace` decode use `traceconv` (vendored).
  The richer `trace_tools.py analyze` path additionally needs `TraceLens`:
  `git clone https://github.com/AMD-AGI/TraceLens.git && pip install -e TraceLens` (pandas/plotly).
  Not required for basic profiling.
