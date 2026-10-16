"""
Workload shared by the Python (mori) and C++ (mori_cpp) MoRI backends.

MoRI's Python benchmark and bench_engine accept the same spellings for these
flags, so both backends pass exactly WORKLOAD_FLAGS; only the rendezvous and
sweep-range flags differ between them. The Python benchmark has no warmup
option, so backends/mori/run_benchmark.py adds --warmup-iters to it.
"""

NUM_ITERS = 500
WARMUP_ITERS = 50
TRANSFER_BATCH_SIZE = 1
NUM_QP_PER_TRANSFER = 1
NUM_WORKER_THREADS = 1
MAX_TRANSFER_SIZE = 2147483648  # 2 GB: MoRI rejects single transfers above UINT32_MAX (4 GiB - 1)

WORKLOAD_FLAGS = [
    "--backend", "rdma",
    "--op-type", "write",
    "--mem-type", "gpu",
    "--transfer-batch-size", str(TRANSFER_BATCH_SIZE),
    "--enable-batch-transfer",
    "--batch-contiguous",
    "--enable-sess",
    "--num-initiator-dev", "1",
    "--num-target-dev", "1",
    "--num-qp-per-transfer", str(NUM_QP_PER_TRANSFER),
    "--num-worker-threads", str(NUM_WORKER_THREADS),
    "--poll_cq_mode", "polling",
    "--iters", str(NUM_ITERS),
    "--warmup-iters", str(WARMUP_ITERS),
    "--log-level", "info",
]


def test_parameters(size_bytes: int) -> dict:
    return {
        "size_bytes": size_bytes,
        "size_mb": size_bytes / (1024 * 1024),
        "num_iters": NUM_ITERS,
        "warmup_iters": WARMUP_ITERS,
        "operation": "write",
        "transfer_batch_size": TRANSFER_BATCH_SIZE,
        "batch_contiguous": True,
        "num_qp_per_transfer": NUM_QP_PER_TRANSFER,
        "num_worker_threads": NUM_WORKER_THREADS,
    }


def clamp_end_size(end_size: int) -> int:
    if end_size > MAX_TRANSFER_SIZE:
        print(f"WARNING: Clamping sweep to {MAX_TRANSFER_SIZE} bytes "
              f"(MoRI rejects single transfers above UINT32_MAX)")
        return MAX_TRANSFER_SIZE
    return end_size
