#!/bin/bash
# ---------------------------------------------------------------------------
# MoRI-IO two-node RDMA block-size sweep across all 8 Pensando Ionic rails.
# Runs INSIDE the workload-testing container (see moriio_sweep.slurm, which
# launches that container on both nodes and calls this).
#
# WHAT THIS ANSWERS
# -----------------
# "What bandwidth and latency does the AINIC fabric deliver between mi355-gpu-44
# and mi355-gpu-45 as a function of transfer block size, with every RDMA NIC
# engaged?"  A 1P/1D prefill->decode KV handoff IS a MoRI-IO RDMA write, so this
# bounds what the serving stack can achieve and shows which block sizes sit on
# the efficient part of the curve.
#
# WHY MoRI'S benchmark.py AND NOT ib_write_bw
# -------------------------------------------
# ib_write_bw measures the NIC. It does not measure MoRI's transfer path: QP
# fan-out per transfer, the chunking layer, the worker-thread pool, or the GPU
# memory registration that ROCm makes awkward. We want the number the serving
# stack will actually see, so we drive tests/python/io/benchmark.py --backend
# rdma -- MoRI's own two-node harness, which already emits exactly the table
# asked for:
#
#   MsgSize (B) | BatchSize | TotalSize (MB) | Max BW (GB/s) | Avg BW (GB/s) | Min Lat (us) | Avg Lat (us)
#
# ib_write_bw stays the right tool for "is the fabric healthy"; preflight below
# runs a cheap equivalent of that check first.
#
# "ENGAGING ALL RDMA NICs"
# -----------------------
# --num-initiator-dev 8 --num-target-dev 8. benchmark_distributed() spawns that
# many processes per node (benchmark.py:1438), each binding GPU = its role_rank
# (benchmark.py:450), and MoRI pairs each GPU with its NUMA-local NIC -- which on
# these nodes is the 8 ionic rails, one per GPU. So initiator GPU i talks to
# target GPU i over rail i: rail-aligned by construction. We deliberately do NOT
# pass --target-dev-offset; that crosses rails on purpose, which fails on a
# rail-only fabric and is a different experiment (fault injection, not measurement).
#
# The harness asserts num_initiator_dev == num_target_dev (benchmark.py:425), so
# NUM_DEV drives both.
#
# THE SWEEP AND WHY THESE BOUNDS
# ------------------------------
# Geometric, 4 KiB -> 32 MiB (benchmark.py:1177, --sweep-step 0 doubles). The low
# end shows per-transfer fixed cost (latency-bound, bandwidth far under line
# rate); the high end shows the asymptote; the knee between them is the
# operationally interesting part. We start at 4 KiB rather than the harness
# default of 8 B because below a page the numbers are dominated by WR posting
# overhead and say nothing about a KV transfer.
#
# For scale: a GLM-5.2 KV page is 43.88 KiB/token (576 B/token/layer x 78
# layers), so one 28,672-token prefill hands over ~1.2 GiB. Real transfers land
# at the TOP of this sweep. The bottom is still worth measuring -- it is where a
# regression in the chunking path shows up first.
#
# MEMORY ARITHMETIC, WHICH IS A REAL CONSTRAINT
# ---------------------------------------------
# benchmark.py:1320 sets buffer_size = max(--buffer-size, --sweep-max-size), and
# :440 allocates (buffer_size+1) * transfer_batch_size bytes PER PROCESS, i.e.
# per GPU. At 32 MiB x BATCH=64 that is ~2.1 GiB/GPU -- fine. At the harness
# default BATCH=256 it would be 8.6 GiB/GPU, which is still allocatable on a
# 288 GB MI355X but needlessly large. BATCH=64 also keeps 64 transfers in flight
# at the small sizes, which is what makes the low end a pipelining measurement
# rather than a serialized-round-trip measurement.
#
# A NOTE ON CHUNKING AT THE TOP OF THE CURVE
# ------------------------------------------
# PlanChunkGeometry (src/io/rdma/common.cpp:459) computes
# softCount = min(ceil(total/chunkBytes), maxChunks), then targetChunkBytes =
# total/finalCount. With the defaults chunk_bytes=64 KiB, max_chunks=64, the
# chunk COUNT saturates at 64, so above 64*64KiB = 4 MiB the effective chunk
# grows with the transfer (a 32 MiB transfer becomes 64 x 512 KiB, not 512 x
# 64 KiB). That is the shipped serving behaviour, so it is what we measure -- but
# it means the top of the curve is partly a max_chunks measurement. CHUNK_SWEEP=1
# below re-runs one fixed block size across chunk_bytes to separate the two.
#
# WHAT IS DELIBERATELY FIXED
# --------------------------
# op-type write   : the disagg path is a WRITE (prefill pushes to decode). Read
#                   is a different QP flow; measure separately if asked.
# qp-per-transfer : 8, from the validated recipe (VLLM_MORIIO_QP_PER_TRANSFER=8
#                   in ../vllm_dissag/connectors/moriio.env.aac), so this
#                   matches serving rather than a synthetic best case.
# worker-threads  : 8, likewise (VLLM_MORIIO_NUM_WORKERS=8).
#
# HOW TO RUN
# ----------
# Same command on both nodes, different --rank. Rank 0 is the INITIATOR and
# prints the populated table; rank 1 is the TARGET and prints an all-zero table
# (expected -- the filter at benchmark.py:1141 drops zero rows from the JSON
# sink, while the printed table still shows them).
#
#     # on mi355-gpu-44
#     ./run_moriio_sweep.sh --rank 0 --master-addr 10.2.80.11
#     # on mi355-gpu-45
#     ./run_moriio_sweep.sh --rank 1 --master-addr 10.2.80.11
#
# Or `sbatch moriio_sweep.slurm`, which does both sides in the workload image.
# ---------------------------------------------------------------------------
set -uo pipefail

RANK=""
MASTER_ADDR=""
MASTER_PORT="${MASTER_PORT:-29511}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rank)        RANK="$2";        shift 2 ;;
    --master-addr) MASTER_ADDR="$2"; shift 2 ;;
    --master-port) MASTER_PORT="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ -z "$RANK" || -z "$MASTER_ADDR" ]] && {
  echo "usage: $0 --rank <0|1> --master-addr <ip> [--master-port N]" >&2; exit 2; }

# --- Fabric configuration. Not guesses: every value is the measured AAC setting
# from ../vllm_dissag/connectors/moriio.env.aac. Overridable so this runs elsewhere.
#
# The 8 ionic rails: driver=ionic, 400000 Mb/s, ACTIVE, one per GPU. The two
# EXCLUDED devices (rocep193s0f0/f1) are driver=bnxt_en at 200000 Mb/s and carry
# the management/default route -- including them makes MoRI raise QPs over the
# mgmt fabric, which times out at connection setup.
: "${MORI_RDMA_DEVICES:=rocep9s0,rocep25s0,rocep105s0,rocep121s0,rocep137s0,rocep153s0,rocep233s0,rocep249s0}"
# GID index 1 is the RoCEv2 entry on Pensando Ionic; index 0 does not connect.
# Proven by ib_write_bw between gpu-44/45; corroborated by MAD commit 61dd42c.
: "${MORI_IB_GID_INDEX:=1}"
# MoRI's NIC-vendor detection matches device NAME prefixes (^mlx5, ^bnxt_re,
# ^ionic). Our devices are named rocep*s0, so the name match MISSES and it falls
# through to `readlink device/driver` -> ionic. Inside a container
# /sys/class/infiniband can be masked, silently dropping it to the mlx5 default.
# Pin it so the path is deterministic regardless of what the container exposes.
: "${MORI_DEVICE_NIC:=ionic}"
# Control plane on the MANAGEMENT interface. The public interface holds the
# default route and answers ICMP, but its TCP is FIREWALLED node-to-node:
# measured gpu-44 -> gpu-45, /dev/tcp/104.238.162.159/22 fails while
# /dev/tcp/10.2.80.16/22 succeeds. Bulk data rides the 8 ionic rails, so the mgmt
# link speed does not enter the measurement.
: "${IFNAME:=enp193s0f1np1}"
: "${NUMA_NODE:=0}"

# From the validated serving recipe, so the measurement matches production.
: "${QP_PER_TRANSFER:=8}"
: "${WORKER_THREADS:=8}"
: "${CHUNK_BYTES:=65536}"
: "${MAX_CHUNKS:=64}"
: "${NUM_DEV:=8}"
: "${BATCH:=64}"

: "${BLOCK_MIN:=4096}"
: "${BLOCK_MAX:=33554432}"
: "${ITERS:=128}"
: "${OP_TYPE:=write}"
: "${MEM_TYPE:=gpu}"
: "${CHUNK_SWEEP:=0}"
: "${CHUNK_SWEEP_BLOCK:=8388608}"
: "${CHUNK_SWEEP_LIST:=16384 32768 65536 131072 262144 524288}"

MORI_REPO="${MORI_REPO:-/opt/mori}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${OUTDIR:-${HERE}/results}"
mkdir -p "$OUTDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${OUTDIR}/moriio_sweep_rank${RANK}_${STAMP}.log"

export MORI_RDMA_DEVICES MORI_IB_GID_INDEX MORI_DEVICE_NIC
export MORI_SOCKET_IFNAME="$IFNAME" GLOO_SOCKET_IFNAME="$IFNAME"
export NCCL_IB_HCA="${NCCL_IB_HCA:-$MORI_RDMA_DEVICES}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-$MORI_IB_GID_INDEX}"
export NCCL_SOCKET_IFNAME="$IFNAME"
# Platform-mandatory: ROCm cannot dmabuf-export HIP-VMM memory, so with
# expandable_segments ON, RegisterRdmaMemoryRegion EFAULTs (errno 14) on the
# first RDMA write. Same reason these are set for the serving path.
export PYTORCH_ALLOC_CONF=expandable_segments:False
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:False
export HSA_ENABLE_IPC_MODE_LEGACY=0
# MoRI registers a large pinned region with the NIC; anything that forks after
# that can COW-remap those pages in the parent while the NIC still DMAs to the
# old physical pages -> host SIGSEGV with no GPU fault. benchmark.py uses
# torch.multiprocessing.spawn, which forks.
export RDMAV_FORK_SAFE=1 IBV_FORK_SAFE=1
export MORI_GPU_ARCHS="${MORI_GPU_ARCHS:-gfx950}"

# --- The engine's OOB address must be THIS node's address on the control
# interface, not the master's (--host is per-node; see _setup_rdma). The shipped
# wrapper derives it with a SIOCGIFADDR ioctl; same thing here, `ip` first.
HOST_IP="$(ip -4 -o addr show dev "$IFNAME" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)"
if [[ -z "$HOST_IP" ]]; then
  # NOTE: 2>/dev/null must sit on the heredoc's own command line. Placed after
  # the PY terminator it parses as a separate null command and the ioctl
  # traceback still reaches the log, obscuring the real message below.
  HOST_IP="$(python3 - "$IFNAME" 2>/dev/null <<'PY'
import fcntl, socket, struct, sys
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print(socket.inet_ntoa(fcntl.ioctl(
    s.fileno(), 0x8915, struct.pack('256s', sys.argv[1][:15].encode()))[20:24]))
PY
)"
fi
if [[ -z "$HOST_IP" ]]; then
  # --host is the LOCAL engine's OOB address, one per node, so there is no
  # sensible default to fall back to. Name the likely cause: inside a container
  # the interface exists only because of --network host.
  {
    echo "FATAL: no IPv4 address on ${IFNAME} (host $(hostname -s), rank ${RANK:-?})."
    echo "  benchmark.py --host is a PER-NODE OOB address, not the master's, so"
    echo "  this cannot be guessed. Check that:"
    echo "    - the container was started with --network host, and"
    echo "    - IFNAME names this cluster's control NIC (AAC: enp193s0f1np1)."
    echo "  Interfaces visible here:"
    ip -4 -o addr show 2>/dev/null | awk '{printf "    %-20s %s\n", $2, $4}'
  } 2>&1 | tee -a "${OUTDIR:-.}/moriio_sweep_rank${RANK:-x}_hostip_fail.log" >&2
  exit 1
fi

{
echo "=============================================================="
echo " MoRI-IO block-size sweep -- 2 nodes x ${NUM_DEV} ionic rails"
echo "   host        : $(hostname -s)   rank ${RANK}   oob ${HOST_IP}"
echo "   master      : ${MASTER_ADDR}:${MASTER_PORT}"
echo "   devices     : ${MORI_RDMA_DEVICES}"
echo "   gid index   : ${MORI_IB_GID_INDEX}   nic: ${MORI_DEVICE_NIC}"
echo "   ctrl ifname : ${IFNAME}   numa: ${NUMA_NODE}"
echo "   block sweep : ${BLOCK_MIN} .. ${BLOCK_MAX} B (geometric x2)"
echo "   op/mem      : ${OP_TYPE} / ${MEM_TYPE}   iters: ${ITERS}   batch: ${BATCH}"
echo "   qp/xfer     : ${QP_PER_TRANSFER}   workers: ${WORKER_THREADS}"
echo "   chunking    : ${CHUNK_BYTES} B x max ${MAX_CHUNKS} (count saturates above $((CHUNK_BYTES*MAX_CHUNKS)) B)"
echo "   devs        : ${NUM_DEV} init x ${NUM_DEV} target, rail-aligned (no offset)"
echo "   alloc/gpu   : ~$(( (BLOCK_MAX+1) * BATCH / 1048576 )) MiB"
echo "   mori repo   : ${MORI_REPO}"
echo "=============================================================="

# --- Preflight. Cheap, and every one of these has caught a real failure here.
echo "[preflight] rail state:"
_want=$(tr ',' '\n' <<<"$MORI_RDMA_DEVICES" | grep -c .)
_have=0
for d in $(tr ',' ' ' <<<"$MORI_RDMA_DEVICES"); do
  if [ -e "/sys/class/infiniband/$d" ]; then
    _st=$(cat "/sys/class/infiniband/$d/ports/1/state" 2>/dev/null || echo "?")
    _rt=$(cat "/sys/class/infiniband/$d/ports/1/rate" 2>/dev/null || echo "?")
    printf "  %-14s state=%-16s rate=%s\n" "$d" "$_st" "$_rt"
    case "$_st" in *ACTIVE*) _have=$((_have+1));; esac
  else
    printf "  %-14s MISSING from /sys/class/infiniband\n" "$d"
  fi
done
echo "  ACTIVE rails: ${_have}/${_want}"
if [ "$_have" -ne "$_want" ]; then
  echo "[preflight] FATAL: not all rails ACTIVE. A partial-fabric number is worse"
  echo "            than no number -- it reads as a MoRI bandwidth regression when"
  echo "            it is really a link that never came up."
  exit 1
fi

# The control plane must connect or torchrun hangs at rendezvous with no error.
if [ "$RANK" != "0" ]; then
  if timeout 10 bash -c "exec 3<>/dev/tcp/${MASTER_ADDR}/${MASTER_PORT}" 2>/dev/null; then
    echo "[preflight] rendezvous port ${MASTER_ADDR}:${MASTER_PORT} already open"
  else
    echo "[preflight] rendezvous port not yet open (normal if rank 0 has not started)"
  fi
fi

# GPU count must cover NUM_DEV: benchmark.py binds GPU = role_rank.
_ngpu="$(python3 -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 0)"
echo "[preflight] visible GPUs: ${_ngpu} (need >= ${NUM_DEV})"
if [ "${_ngpu:-0}" -lt "$NUM_DEV" ]; then
  echo "[preflight] FATAL: fewer GPUs than requested devices."
  exit 1
fi
if [ ! -f "${MORI_REPO}/tests/python/io/benchmark.py" ]; then
  echo "[preflight] FATAL: ${MORI_REPO}/tests/python/io/benchmark.py not found."
  echo "            Point MORI_REPO at a FULL mori checkout. Note that a sparse"
  echo "            checkout (core.sparseCheckout=true) TRACKS tests/ without"
  echo "            materialising it, so the file can be 'in git' and still absent."
  exit 1
fi
echo

cd "$MORI_REPO" || exit 1
export PYTHONPATH="${MORI_REPO}${PYTHONPATH:+:$PYTHONPATH}"

# NUMA pinning: the shipped wrapper refuses multi-NIC HOST-memory runs without
# it, because MatchCpuNics() must return the same rail ordering on both nodes or
# the two sides pair different rails. We use GPU memory, where pairing follows
# the GPU index, but pin anyway so both sides stay deterministic.
NUMACTL=()
if command -v numactl >/dev/null 2>&1; then
  NUMACTL=(numactl --cpunodebind="$NUMA_NODE" --membind="$NUMA_NODE")
else
  echo "[warn] numactl absent -- NIC/CPU affinity left to the scheduler."
fi

# ---- flag capability probe -------------------------------------------------
# benchmark.py's arg set moves with the mori checkout, and argparse rejects an
# unknown flag with exit 2 -- the whole sweep dies before a single transfer.
# That is exactly how run 1 died here, so the probe stays.
#
# IMPORTANT, and the reason this comment was rewritten: the flags that went
# missing were missing from the MOUNTED HARNESS, not from the measured stack.
# MORI_SRC is bind-mounted only to obtain tests/python/io/benchmark.py, which
# the image does not ship. The checkout used for the recorded run was an older
# 6ad812c tree, whose benchmark.py predates --mem-type/--max-chunks/--chunk-bytes.
# The libmori actually exercised is the IMAGE's pinned 42e895472b08, whose own
# benchmark.py DOES accept all three (defaults: --chunk-bytes 65536,
# --max-chunks 64). Mount a 42e8954 checkout and the probe passes them through.
#
# So: probe, don't assume -- but do not read a missing flag as "the shipped
# engine lacks the knob". A flag that is absent from the harness means the
# planner ran on its build defaults, which is a DEFAULT, not a behaviour
# change. The omission is reported loudly and the README says what each one
# means for reading the table.
_BENCH_HELP="$(python3 -m tests.python.io.benchmark --help 2>&1 || true)"
_has_flag() { case "$_BENCH_HELP" in *"$1"*) return 0 ;; *) return 1 ;; esac; }

_OPT_ARGS=()
_MISSING=()
if _has_flag "--mem-type";   then _OPT_ARGS+=(--mem-type "$MEM_TYPE"); else _MISSING+=("--mem-type (build transfers GPU memory unconditionally)"); fi
if _has_flag "--max-chunks"; then _OPT_ARGS+=(--max-chunks "$MAX_CHUNKS"); else _MISSING+=("--max-chunks (planner uses its built-in cap)"); fi
if [ "${#_MISSING[@]}" -gt 0 ]; then
  echo "[flags] this benchmark.py does not accept:"
  printf "[flags]   %s\n" "${_MISSING[@]}"
  echo "[flags] mori HEAD: $(git -C "${MORI_REPO:-.}" log --oneline -1 2>/dev/null || echo unknown)"
  echo "[flags] Those knobs fall back to the build defaults; the sweep is still"
  echo "[flags] a real measurement, just not of those two axes."
fi

_run_bench() {  # $1 = tag ; rest = extra args
  local tag="$1"; shift
  echo "-------- ${tag} --------"
  "${NUMACTL[@]}" timeout "${BENCH_TIMEOUT:-5400}" torchrun \
    --nnodes=2 --node_rank="$RANK" --nproc_per_node=1 \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    -m tests.python.io.benchmark \
      --backend rdma \
      --host "$HOST_IP" \
      --op-type "$OP_TYPE" \
      --num-initiator-dev "$NUM_DEV" \
      --num-target-dev "$NUM_DEV" \
      --transfer-batch-size "$BATCH" \
      --enable-batch-transfer \
      --num-qp-per-transfer "$QP_PER_TRANSFER" \
      --num-worker-threads "$WORKER_THREADS" \
      --iters "$ITERS" \
      --log-level info \
      "${_OPT_ARGS[@]}" \
      "$@"
  local rc=$?
  echo "-------- ${tag} exit=${rc} --------"
  return $rc
}

# Invoked directly rather than through tools/run_internode_io_benchmark.sh so the
# exit status is the benchmark's own and the full arg list lands in the log --
# and so the same driver can run both sweeps.
_SWEEP_ARGS=(--all --sweep-start-size "$BLOCK_MIN" --sweep-max-size "$BLOCK_MAX")
# --sweep-step 0 asks for geometric stepping. Builds without the flag step
# geometrically already -- it exists to offer LINEAR stepping, so its absence
# does not change the ladder this sweep wants.
_has_flag "--sweep-step"  && _SWEEP_ARGS+=(--sweep-step 0)
_has_flag "--chunk-bytes" && _SWEEP_ARGS+=(--chunk-bytes "$CHUNK_BYTES")

_run_bench "block-size sweep" "${_SWEEP_ARGS[@]}"
_rc=$?

# Optional second experiment: hold the block size and vary chunk_bytes, to
# separate "how fast is an 8 MiB transfer" from "how is that transfer chunked".
# Each point is its own torchrun because chunk_bytes is engine config, not a
# per-transfer argument.
if [ "$CHUNK_SWEEP" = "1" ] && [ "$_rc" -eq 0 ]; then
  if ! _has_flag "--chunk-bytes"; then
    # Skip loudly. Running the loop anyway would emit six identical tables
    # under six different chunk= labels -- a result that looks like "chunk size
    # does not matter" when it actually means the knob was never applied.
    echo "[chunk-sweep] SKIPPED: this benchmark.py has no --chunk-bytes."
    echo "[chunk-sweep] Every point would use the same built-in chunking and the"
    echo "[chunk-sweep] table would falsely read as chunk-size-independent."
  else
    for _cb in $CHUNK_SWEEP_LIST; do
      _run_bench "chunk sweep block=${CHUNK_SWEEP_BLOCK} chunk=${_cb}" \
        --buffer-size "$CHUNK_SWEEP_BLOCK" --chunk-bytes "$_cb" || _rc=$?
      sleep 3
    done
  fi
fi

echo
echo "==== moriio sweep rank=${RANK} exit=${_rc} ===="
exit "$_rc"
} 2>&1 | tee "$LOG"

exit "${PIPESTATUS[0]}"
