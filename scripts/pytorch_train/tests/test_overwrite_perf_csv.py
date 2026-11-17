#!/usr/bin/env python3
"""perf_$MODEL.csv must contain only the current conversion, not prior NaNs."""

import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "pytorch_benchmark_report.py"


def write_summary(path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["avg_fps_gpu", "avg_tflops"])
        writer.writeheader()
        writer.writerow({"avg_fps_gpu": "7.7513", "avg_tflops": "506.32"})


def write_stale_perf(path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "performance",
                "metric",
                "mode",
                "precision",
                "batch_size",
                "seq_len",
                "device",
                "num_gpus",
            ],
        )
        writer.writeheader()
        for _ in range(3):
            writer.writerow(
                {
                    "model": "Flux",
                    "performance": "nan",
                    "metric": "FPS_per_GPU",
                    "mode": "posttrain",
                    "precision": "BF16",
                    "batch_size": "16",
                    "seq_len": "256",
                    "device": "MI350X",
                    "num_gpus": "8",
                }
            )
            writer.writerow(
                {
                    "model": "Flux",
                    "performance": "nan",
                    "metric": "TFLOPS_per_GPU",
                    "mode": "posttrain",
                    "precision": "BF16",
                    "batch_size": "16",
                    "seq_len": "256",
                    "device": "MI350X",
                    "num_gpus": "8",
                }
            )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        summary = tmpdir / "runs_summary.csv"
        perf = tmpdir / "perf_Flux.csv"
        write_summary(summary)
        write_stale_perf(perf)

        subprocess.check_call(
            [
                sys.executable,
                str(SCRIPT),
                "--mode",
                "posttrain",
                "--model",
                "Flux",
                "--input",
                str(summary),
                "--output",
                str(perf),
                "--precision",
                "BF16",
                "--batch_size",
                "16",
                "--seq_len",
                "256",
                "--device",
                "MI350X",
                "--num_gpus",
                "8",
            ],
            cwd=tmpdir,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        with perf.open(newline="") as handle:
            rows = list(csv.DictReader(handle))

        if len(rows) != 2:
            print(f"Expected 2 rows after conversion, got {len(rows)}", file=sys.stderr)
            return 1
        performances = [row["performance"] for row in rows]
        if any(value.lower() == "nan" for value in performances):
            print(f"Stale NaN rows were retained: {performances}", file=sys.stderr)
            return 1
        if performances != ["7.7513", "506.32"]:
            print(f"Unexpected performance rows: {performances}", file=sys.stderr)
            return 1
    print("Overwrite of existing perf CSV is correct.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
