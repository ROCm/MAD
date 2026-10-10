#!/usr/bin/env python3
"""extract_primus_perf must recover TPS from Megatron-Bridge logs that omit it."""

import csv
import importlib.util
import sys
import tempfile
from pathlib import Path

EXTRACTOR = Path(__file__).resolve().parents[1] / "extract_primus_perf.py"

_spec = importlib.util.spec_from_file_location("extract_primus_perf", EXTRACTOR)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)


BRIDGE_UNENRICHED = """\
  seq_length ................................ 8192
  world_size ................................ 8
 [2026-09-10 15:00:00] iteration       50/     200 | consumed samples:         1600 | elapsed time per iteration (ms): 450.2 | throughput per GPU (TFLOP/s/GPU): 123.4 | learning rate: 1.000000E-04 | global batch size:    32 | lm loss: 1.234567E+00 | loss scale: 1.0 |
"""

MEGATRON_NEW = """\
iteration      100/     200 | elapsed time per iteration (ms): 100.0 | compute per GPU (TFLOP/s/GPU): 496.3 (avg 496.1) | tokens/s/GPU inst/harmonic mean: 9640.7/9629.8 | global batch size: 32
"""

BRIDGE_MODEL_TFLOP = """\
  seq_length ................................ 8192
  world_size ................................ 8
Step Time : 0.79s GPU utilization: 62.4MODEL_TFLOP/s/GPU
 [2026-09-16 15:26:14] iteration       20/     200 | consumed samples:          160 | elapsed time per iteration (ms): 789.8 | learning rate: 2.000000E-06 | global batch size:     8 | lm loss: 3.581810E+00 | loss scale: 1.0 |
Step Time : 0.79s GPU utilization: 2254.7MODEL_TFLOP/s/GPU
"""

# megatron_bridge pretrain (mamba_130M): no tokens/s/GPU, no dotted args dump.
# seq_length is the YAML dump; world_size comes from Primus env. Last-iter
# elapsed is the steady-state number (iter 1 is warmup).
BRIDGE_PRETRAIN_NO_PRINTED_TPS = """\
[Primus:Env] rank=0, world_size=8, local_rank=0, master=localhost:1234
  seq_length: 2048
ThroughputAverageExtension initialized with seq_len: 2048 world_size: None
iteration        1/       3 | elapsed time per iteration (ms): 50652.0 | global batch size:    32 | lm loss: 1.097315E+01
iteration        2/       3 | elapsed time per iteration (ms): 104.6 | global batch size:    32 | lm loss: 1.097168E+01
iteration        3/       3 | elapsed time per iteration (ms): 88.2 | global batch size:    32 | lm loss: 7.715407E+00
"""

TORCHTITAN_LINE = """\
step: 50  loss: 1.2  memory: 10GiB tps: 1,444  tflops: 300.32  mfu: 23.10%
"""


def _write(tmp: Path, text: str) -> Path:
    path = tmp / "train.log"
    path.write_text(text)
    return path


def test_bridge_unenriched_computes_tps() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log = _write(Path(tmp), BRIDGE_UNENRICHED)
        metrics = _mod.extract_metrics(str(log))
        expected = 8192 * 32 / (450.2 / 1000.0) / 8
        assert metrics.get("tps") is not None, "expected derived TPS, got None"
        assert abs(float(metrics["tps"]) - expected) < 0.15
        assert metrics["tflops"] == "123.4"


def test_log_world_size_overrides_cli_fallback() -> None:
    # Multi-node: run.sh passes the local GPU count (8); the log reports the global 16.
    with tempfile.TemporaryDirectory() as tmp:
        log = _write(Path(tmp), BRIDGE_UNENRICHED.replace("world_size ................................ 8", "world_size ................................ 16"))
        metrics = _mod.extract_metrics(str(log), num_gpus=8)
        expected = 8192 * 32 / (450.2 / 1000.0) / 16
        assert abs(float(metrics["tps"]) - expected) < 0.15


def test_bridge_uses_cli_overrides_when_args_missing() -> None:
    line = (
        "iteration 10/200 | elapsed time per iteration (ms): 200.0 | "
        "throughput per GPU (TFLOP/s/GPU): 50.0 | global batch size: 16 |\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        log = _write(Path(tmp), line)
        metrics = _mod.extract_metrics(str(log), seq_length=4096, num_gpus=8)
        expected = 4096 * 16 / 0.2 / 8
        assert abs(float(metrics["tps"]) - expected) < 0.15


def test_bridge_pretrain_derives_tps_from_env_world_size() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log = _write(Path(tmp), BRIDGE_PRETRAIN_NO_PRINTED_TPS)
        metrics = _mod.extract_metrics(str(log))
        expected = 2048 * 32 / (88.2 / 1000.0) / 8
        assert metrics.get("tps") is not None, "expected derived TPS, got None"
        assert abs(float(metrics["tps"]) - expected) < 0.15


def test_printed_megatron_tps_not_overridden() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log = _write(Path(tmp), MEGATRON_NEW)
        metrics = _mod.extract_metrics(str(log), seq_length=8192, num_gpus=8)
        assert metrics["tps"] == "9629.8"
        assert metrics["tflops"] == "496.1"


def test_bridge_model_tflop_line_fills_tflops() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log = _write(Path(tmp), BRIDGE_MODEL_TFLOP)
        metrics = _mod.extract_metrics(str(log))
        assert metrics["tflops"] == "2254.7"
        assert abs(float(metrics["tps"]) - (8192 * 8 / 0.7898 / 8)) < 0.15


def test_printed_tflops_wins_over_model_tflop_line() -> None:
    mixed = (
        MEGATRON_NEW
        + "Step Time : 0.79s GPU utilization: 9999.9MODEL_TFLOP/s/GPU\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        log = _write(Path(tmp), mixed)
        metrics = _mod.extract_metrics(str(log))
        assert metrics["tflops"] == "496.1"


def test_torchtitan_line_unchanged() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log = _write(Path(tmp), TORCHTITAN_LINE)
        metrics = _mod.extract_metrics(str(log))
        assert metrics["tps"] == "1444"
        assert metrics["tflops"] == "300.32"
        assert metrics["mfu"] == "23.10"


def test_cli_writes_csv() -> None:
    import subprocess

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        log = _write(tmpdir, BRIDGE_UNENRICHED)
        out = tmpdir / "primus_perf_output.csv"
        subprocess.check_call(
            [sys.executable, str(EXTRACTOR), str(log), str(out), "--seq-length", "8192", "--num-gpus", "8"]
        )
        with out.open(newline="") as handle:
            rows = {row["metric"]: row["performance"] for row in csv.DictReader(handle)}
        assert "tokens_per_second" in rows
        assert float(rows["tokens_per_second"]) > 0
        assert rows["tflops"] == "123.4"


if __name__ == "__main__":
    test_bridge_unenriched_computes_tps()
    test_bridge_uses_cli_overrides_when_args_missing()
    test_bridge_pretrain_derives_tps_from_env_world_size()
    test_printed_megatron_tps_not_overridden()
    test_bridge_model_tflop_line_fills_tflops()
    test_printed_tflops_wins_over_model_tflop_line()
    test_torchtitan_line_unchanged()
    test_cli_writes_csv()
    print("extract_primus_perf Megatron-Bridge coverage is correct.")
