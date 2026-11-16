#!/usr/bin/env python3
"""Regression tests for the two benchmark log parsers.

    parse_to_csv.py      -- wired into benchmark_xPyD.sh; writes both
                            <LOG>_CONCURRENCY.csv and madengine perf.csv
    benchmark_parser.py  -- manual tool; the only one that reports ITL/TTFT/TPOT

Pure stdlib, no GPU, no vLLM, no pandas. The fixtures below reproduce the
exact shapes seen in real runs, which is where these bugs came
from:

  * a warmup cell that emits a result block with NO [RUNNING] header and sits
    before the "iter: 1" marker (isl=32 osl=32, so it must not be scored)
  * a cell that stalls: [RUNNING] then no result block
  * the first real cell, which used to be dropped outright

Run: python3 tests/test_parsers.py
"""

import importlib.util
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _load(name, path, stub_pandas=False):
    if stub_pandas and 'pandas' not in sys.modules:
        sys.modules['pandas'] = types.ModuleType('pandas')
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _result_block(throughput, itl, con, ok=16):
    """A trimmed-down 'vllm bench serve' result block."""
    return f"""============ Serving Benchmark Result ============
Successful requests:                     {ok}
Failed requests:                         0
Total input tokens:                      16,384
Total generated tokens:                  16,384
Request throughput (req/s):              0.11
Output token throughput (tok/s):         {throughput / 2:.2f}
Total token throughput (tok/s):          {throughput:.2f}
Mean TTFT (ms):                          1000.00
Median TTFT (ms):                        1000.00
Mean ITL (ms):                           {itl:.2f}
Median ITL (ms):                         {itl:.2f}
Mean TPOT (ms):                          {itl:.2f}
Median TPOT (ms):                        {itl:.2f}
==================================================
Maximum request concurrency:             {con}
"""


def _running(con, prompts=16, isl=1024, osl=1024):
    return f"[RUNNING] prompts {prompts} isl {isl} osl {osl} con {con} (timeout 1800s)\n"


# Stalled shape: warmup result with no header, then a con=1 cell that stalls,
# then three good cells.
LOG_WITH_STALL = (
    "Warmup run: 16 prompts @ con=1 isl=32 osl=32\n"
    + _result_block(9999.0, 570.85, 1)
    + "Running the benchserving script for iter: 1\n"
    + _running(1)
    + "[STALL] isl=1024 osl=1024 con=1 timed out after 1800s\n"
    + _running(8)
    + _result_block(27.75, 573.94, 8)
    + _running(16)
    + _result_block(55.27, 575.90, 16, ok=32)
    + _running(32)
    + _result_block(110.01, 578.16, 32, ok=64)
)

# Complete shape: warmup, then every requested cell completes.
LOG_CLEAN = (
    "Warmup run: 16 prompts @ con=1 isl=32 osl=32\n"
    + _result_block(9999.0, 60.90, 1)
    + "Running the benchserving script for iter: 1\n"
    + _running(8)
    + _result_block(235.40, 61.99, 8)
    + _running(16)
    + _result_block(411.15, 69.02, 16, ok=32)
    + _running(32)
    + _result_block(702.06, 79.54, 32, ok=64)
)

# Two iterations of the same cell: max throughput wins, one row only.
LOG_TWO_ITERS = (
    "Running the benchserving script for iter: 1\n"
    + _running(8)
    + _result_block(100.0, 50.0, 8)
    + "Running the benchserving script for iter: 2\n"
    + _running(8)
    + _result_block(200.0, 40.0, 8)
)

NIAH_LOG = """\
[niah] start words=2000
=== NIAH summary (mean/min/max across 3 seed(s)) ===
  words=  2000  mean=10.0/10  min=10  max=10  decoys=0.0  RETRIEVAL  (n=3)
  words=  8000  mean=7.0/10  min=6  max=8  decoys=0.0  PARTIAL  (n=3)
  words= 16000  mean=0.0/10  min=0  max=0  decoys=0.0  NONE  (n=3)
"""

FAILS = []


def check(cond, label):
    print(("  PASS  " if cond else "  FAIL  ") + label)
    if not cond:
        FAILS.append(label)


def _write(text, suffix="_xP2_yD2_DeepSeek-V4-Flash-FP8_CONCURRENCY.log"):
    fd, path = tempfile.mkstemp(prefix="benchmark_999999_", suffix=suffix)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    return path


def test_parse_to_csv():
    print("\nparse_to_csv.py")
    m = _load("ptc", os.path.join(ROOT, "parse_to_csv.py"))

    p = _write(LOG_CLEAN)
    res = m.parse_benchmark_log(p)
    cons = sorted(v["concurrency"] for v in res.values())
    check(cons == [8, 16, 32],
          f"clean log: every requested cell present (got {cons})")
    check(all(v["input_tokens"] == 1024 for v in res.values()),
          "clean log: warmup (isl=32) excluded, so no isl=32 row")
    by_con = {v["concurrency"]: v["max_throughput"] for v in res.values()}
    check(by_con.get(8) == 235.40,
          f"clean log: first real cell keeps its own throughput (got {by_con.get(8)})")

    p = _write(LOG_WITH_STALL)
    res = m.parse_benchmark_log(p)
    by_con = {v["concurrency"]: v["max_throughput"] for v in res.values()}
    check(sorted(by_con) == [8, 16, 32],
          f"stalled log: stalled con=1 yields no row (got {sorted(by_con)})")
    check(by_con.get(8) == 27.75,
          f"stalled log: con=8 throughput labelled con=8, not con=1 (got {by_con.get(8)})")
    check(1 not in by_con,
          "stalled log: con=8's number is NOT published under con=1")

    p = _write(LOG_TWO_ITERS)
    res = m.parse_benchmark_log(p)
    check(len(res) == 1 and list(res.values())[0]["max_throughput"] == 200.0,
          "two iterations of one cell collapse to a single max row")


def test_parse_to_csv_niah():
    print("\nparse_to_csv.py --niah")
    m = _load("ptc2", os.path.join(ROOT, "parse_to_csv.py"))
    fd, p = tempfile.mkstemp(suffix="_niah.log")
    with os.fdopen(fd, "w") as f:
        f.write(NIAH_LOG)
    res = m.parse_niah_log(p)
    check(sorted(res) == [2000, 8000, 16000],
          f"summary lines with decoys= and a verdict parse (got {sorted(res)})")
    check(res.get(2000, {}).get("mean") == 10.0 and res[2000]["n"] == 3,
          "mean and seed count read correctly")
    check(res.get(16000, {}).get("mean") == 0.0,
          "a 0.0 rung is kept, not treated as missing")

    out = tempfile.mkstemp(suffix="_perf.csv")[1]
    m.save_niah_perf_csv(res, out, "DeepSeek-V4-Pro-FP8")
    body = open(out).read().splitlines()
    check(len(body) == 4, f"perf.csv has a header + one row per rung (got {len(body)})")
    check("retrieval/10" in body[1], "metric column names the retrieval score")


def test_benchmark_parser():
    print("\nbenchmark_parser.py")
    m = _load("bp", os.path.join(ROOT, "benchmark_parser.py"), stub_pandas=False)

    p = _write(LOG_WITH_STALL)
    rows = m.parse_benchmark_log(p)
    cons = [r["Concurrency"] for r in rows]
    check(cons == [8, 16, 32], f"stalled cell skipped, rest parsed in order (got {cons})")
    check(rows[0]["Median_ITL_ms"] == 573.94,
          f"ITL attributed to the right cell (got {rows[0]['Median_ITL_ms']})")
    check(rows[0]["Model"] == "DeepSeek-V4-Flash-FP8" and rows[0]["xP_yD"] == "2p2d",
          "model and topology recovered from the filename")
    check(all(r["Failed"] == 0 for r in rows), "counters parse as ints, not floats")

    p = _write(LOG_CLEAN)
    rows = m.parse_benchmark_log(p)
    check([r["Concurrency"] for r in rows] == [8, 16, 32],
          "clean log: warmup block before the first [RUNNING] is ignored")

    # The tool must work where pandas is absent; it is the only ITL reporter.
    check(m.render_table(rows, m.COMPACT_COLS).count("\n") == len(rows),
          "render_table emits a header plus one line per row without pandas")
    out = tempfile.mkstemp(suffix=".csv")[1]
    m.write_csv(rows, m.COMPACT_COLS, out)
    lines = open(out).read().strip().splitlines()
    check(len(lines) == len(rows) + 1 and "Median_ITL_ms" in lines[0],
          "write_csv round-trips every row with ITL included")


def test_perf_csv_append():
    """BENCH=validate writes perf.csv twice; the second must not clobber the first.

    A validate run lost all six NIAH rows because both writers truncated, and the
    concurrency sweep runs after NIAH.
    """
    import csv as _csv

    m = _load("ptc3", os.path.join(ROOT, "parse_to_csv.py"))
    d = tempfile.mkdtemp()
    p = os.path.join(d, "perf.csv")

    m.save_niah_perf_csv({2000: {"mean": 10.0, "n": 1},
                          8000: {"mean": 9.0, "n": 1}}, p, "DSV4-Flash")
    m.save_perf_csv({(1024, 1024, 1): {"max_throughput": 63.56, "input_tokens": 1024,
                                       "output_tokens": 1024, "concurrency": 1}},
                    p, "DSV4-Flash")

    rows = list(_csv.DictReader(open(p)))
    check(len(rows) == 3, "NIAH rows survive the concurrency sweep that follows them")
    check(sum("niah" in r["metric"] for r in rows) == 2, "both NIAH rows are still present")
    check(open(p).read().count("model,n_gpus") == 1, "header written exactly once")

    solo = os.path.join(d, "solo.csv")
    m.save_perf_csv({(1024, 1024, 8): {"max_throughput": 1.0, "input_tokens": 1024,
                                       "output_tokens": 1024, "concurrency": 8}},
                    solo, "X")
    check(len(list(_csv.DictReader(open(solo)))) == 1,
          "a single-bench job still writes exactly its own rows")


if __name__ == "__main__":
    test_parse_to_csv()
    test_parse_to_csv_niah()
    test_benchmark_parser()
    test_perf_csv_append()
    print("\n" + ("FAILED: " + "; ".join(FAILS) if FAILS else "all parser tests passed"))
    sys.exit(1 if FAILS else 0)
