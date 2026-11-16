#!/usr/bin/env python3
"""
Run MoRI's Python IO benchmark (tests/python/io/benchmark.py) with warmup.
"""

import os
import sys

from tests.python.io import benchmark as mori_benchmark

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
