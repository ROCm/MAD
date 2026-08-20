"""Synthetic runs, small enough to reason about and to run without a cluster.

Every fixture builds the shape of a real artifact -- an RCCL record with its tail, a chrome-trace
event spread over several lines, a per-node directory layout -- so a test failing here means the
parser changed behaviour, not that the fixture drifted.
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def coll_line(coll: str = "AllReduce", count: int = 1024, dtype: int = 9, nranks: int = 8,
              grank: int = 0, pid: int = 1234, host: str = "worker", tail: bool = True,
              stream: str = "0xdef") -> str:
    """One RCCL collective record, in the shape ``NCCL_DEBUG=INFO`` with ``COLL`` prints.

    ``stream="(nil)"`` is what a rank-local (``nranks=1``) communicator prints, since it does its
    work without a stream.
    """
    line = (f"{host}:{pid}:{pid + 1} [{grank % 8}] NCCL INFO {coll}: opCount 1f "
            f"sendbuff 0x1a recvbuff 0x2b count {count} datatype {dtype} op 0 root 0 "
            f"comm 0xabc [nranks={nranks}]")
    if tail:
        line += f" stream {stream} task 0 globalrank {grank}"
    return line


def topo_line(src: int = 0, dst: int = 1, channel: int = 0, transport: str = "P2P/IPC",
              nranks: int = 8) -> str:
    """One connection RCCL prints while building a communicator, tail included.

    The `comm 0x.. nRanks N` tail is what a real line carries; keeping it here means a torn-line
    fixture is torn the way real ones are.
    """
    return (f"worker:1:2 [{src}] NCCL INFO Channel {channel:02d}/02 : "
            f"{src}[a0] -> {dst}[b1] via {transport} comm 0xabc nRanks {nranks:02d}")


def trace_event(coll: str = "allreduce", nin: int = 512, nout: int = 512, group: int = 8,
                dtype: str = "BFloat16", pg: str = "tp:device", cat: str = "kernel",
                dur: float = 50.0) -> str:
    """A chrome-trace event as torch writes it: pretty-printed over several lines."""
    return (
        '    {\n'
        f'      "ph": "X", "cat": "{cat}", "name": "ncclDevKernel_Generic", "pid": 1, "tid": 7,\n'
        f'      "ts": 1000.0, "dur": {dur},\n'
        f'      "args": {{"Collective name": "{coll}", "In msg nelems": {nin}, '
        f'"Out msg nelems": {nout}, "Group size": {group}, "dtype": "{dtype}", '
        f'"Process Group Description": "{pg}"}}\n'
        '    },'
    )


def write(path: Path, lines: list, compress: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(lines) + "\n"
    if compress:
        path.write_bytes(gzip.compress(body.encode()))
    else:
        path.write_text(body)
    return path


@pytest.fixture
def sglang_run(tmp_path: Path) -> Path:
    """Two prefill and two decode nodes, eight ranks each, plus the profile-point logs."""
    run = tmp_path / "25999"
    for role, nodes in (("prefill", (0, 1)), ("decode", (2, 3))):
        for node in nodes:
            lines = []
            for rank in range(8):
                # Decode moves smaller messages more often, as it does in a real run.
                count = 1024 if role == "prefill" else 128
                calls = 2 if role == "prefill" else 4
                lines += [coll_line(count=count, grank=rank, pid=2000 + rank)] * calls
            lines.append(topo_line())
            write(run / f"{role}_NODE{node}.log", lines)
    for role in ("prefill", "decode"):
        write(run / f"benchmark_25999_x_PROFILE_{role}.log", [f"INFO: profiling {role} worker(s)"])
    return run


@pytest.fixture
def primus_run(tmp_path: Path) -> Path:
    """Two nodes, one stdout each, two datatype phases announced in the log."""
    run = tmp_path / "25577"
    for node in (0, 1):
        lines = []
        # The mount list and the `docker run` line name both configs before either phase starts;
        # only the launcher's `--config` says which one the run is on.
        lines += [f"  /host/primus_configs/llama3.1_70B-{d}-pretrain.yaml:"
                  f"/workspace/configs/llama3.1_70B-{d}-pretrain.yaml" for d in ("FP8", "BF16")]
        lines.append("DOCKER RUN OPERATION: docker run -t -d "
                     "-v /host/configs/llama3.1_70B-FP8-pretrain.yaml:/workspace/x.yaml "
                     "-v /host/configs/llama3.1_70B-BF16-pretrain.yaml:/workspace/y.yaml img")
        for dtype in ("BF16", "FP8"):
            lines.append("[INFO] [main] Executing: bash runner/primus-cli-direct.sh -- train "
                         f"pretrain --config examples/megatron/configs/MI355X/"
                         f"llama3.1_70B-{dtype}-pretrain.yaml")
            for rank in range(2):
                lines += [coll_line(count=2048, grank=rank, pid=3000 + rank)] * 3
            lines.append(" iteration 1/10 | elapsed time per iteration (ms): 250.5 |")
            lines.append(" throughput per GPU (tokens/s/GPU): 1200.5 | TFLOP/s/GPU): 410.2 |")
        write(run / f"node_{node}" / "stdout.out", lines)
    return run
