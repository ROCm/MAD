#!/usr/bin/env python3
"""
Run MoRI's Python IO benchmark (tests/python/io/benchmark.py) with warmup.

The benchmark has no warmup option, while bench_engine (mori_cpp) has
--warmup-iters. This takes the same flag, runs that many untimed transfers
before each timed size, and otherwise runs MoRI's benchmark unchanged.
Launch it like the benchmark (torchrun ... run_benchmark.py <benchmark args>)
with $MORI_ROOT on PYTHONPATH.
"""

import os
import sys

from tests.python.io import benchmark as mori_benchmark

# The benchmark's per-GPU workers are spawned processes that re-import this
# module after the flag has been stripped, so the count travels in the environment.
_WARMUP_ENV = "MORI_BENCH_WARMUP_ITERS"
if "--warmup-iters" in sys.argv:
    _i = sys.argv.index("--warmup-iters")
    os.environ[_WARMUP_ENV] = sys.argv[_i + 1]
    del sys.argv[_i:_i + 2]
WARMUP_ITERS = int(os.environ.get(_WARMUP_ENV, "0"))

_timed_run_and_compute = mori_benchmark.MoriIoBenchmark._run_and_compute


def _run_and_compute(self, buffer_size, transfer_batch_size, iters):
    for _ in range(WARMUP_ITERS):
        self.run_once(buffer_size, transfer_batch_size)
    return _timed_run_and_compute(self, buffer_size, transfer_batch_size, iters)


mori_benchmark.MoriIoBenchmark._run_and_compute = _run_and_compute

if __name__ == "__main__":
    mori_benchmark.benchmark()
