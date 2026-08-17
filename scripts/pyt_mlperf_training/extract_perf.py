#!/usr/bin/env python3
###############################################################################
#
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
###############################################################################
"""Extract MLPerf Training Llama-3.1 perf metrics from a rank-0 stdout log.

The script parses the ``:::MLLOG`` stream (headline MLCommons metric source)
and the NeMo/Lightning per-iteration log lines (``train_step_timing``) to
produce a single-shot perf summary comparable across runs.

Headline metric (matches the "Latency (in minutes)" column at
https://mlcommons.org/benchmarks/training/):

    time_to_train_min = (run_stop.time_ms - run_start.time_ms) / 60000

The metric is officially meaningful only when ``run_stop.status == "success"``.
For shorter smoke / perf-measurement runs the script still reports the wall
time between run_start and run_stop but marks it as non-submittable.

Supporting metrics derived from ``train_step_timing`` (iter >= 1, iter 0 is
treated as CUDA/JIT warmup and dropped):

    - step time:      mean / p50 / p95 / stdev (seconds)
    - samples/sec:    GBS / mean_step_time (cluster)
    - tokens/sec:     GBS * seq_len / mean_step_time (cluster, per-GPU)
    - TFLOP/s/GPU:    6 * N_params * seq_len * GBS / mean_step_time / N_GPU
    - MFU:            TFLOP/s/GPU / peak_bf16_tflops

Usage::

    python3 extract_perf.py <rank0-stdout.out> [--peak-bf16-tflops 1307]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Approximate parameter counts used for TFLOPs estimation. 6*N per token is the
# standard MLPerf/megatron FLOPs-per-token formula (fwd+bwd, no recompute).
PARAM_COUNT: dict[str, int] = {
    "8b": 8_030_000_000,
    "70b": 70_553_000_000,
    "405b": 405_853_000_000,
}

# MI325X dense BF16 matrix peak (TFLOP/s). Override via --peak-bf16-tflops for
# other accelerators.
DEFAULT_PEAK_BF16_TFLOPS = 1307.0

MLLOG_PREFIX = ":::MLLOG"
MLLOG_RE = re.compile(r":::MLLOG\s+(\{.*\})\s*$")

# Matches NeMo/Lightning lines of the form:
#   "Training epoch 0, iteration 5/99 | lr: ... | global_step: 5 |
#    reduced_train_loss: 11.23 | train_step_timing in s: 3.528 |
#    consumed_samples: 384 | val_loss: 17.03"
TRAIN_STEP_RE = re.compile(
    r"Training epoch \d+, iteration (?P<iter>\d+)/\d+.*?"
    r"global_step:\s*(?P<step>\d+).*?"
    r"reduced_train_loss:\s*(?P<loss>[-+0-9.eE]+).*?"
    r"train_step_timing in s:\s*(?P<step_time>[-+0-9.eE]+)"
    r"(?:.*?consumed_samples:\s*(?P<samples>\d+))?"
    r"(?:.*?val_loss:\s*(?P<val_loss>[-+0-9.eE]+))?",
)


@dataclass
class MllogSummary:
    benchmark: Optional[str] = None
    org: Optional[str] = None
    division: Optional[str] = None
    status_submission: Optional[str] = None
    platform: Optional[str] = None
    global_batch_size: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = None
    max_sequence_length: Optional[int] = None
    seed: Optional[int] = None
    train_samples: Optional[int] = None

    init_start_ms: Optional[int] = None
    init_stop_ms: Optional[int] = None
    run_start_ms: Optional[int] = None
    run_stop_ms: Optional[int] = None
    run_stop_status: Optional[str] = None
    run_stop_step: Optional[int] = None
    run_stop_samples: Optional[int] = None

    eval_accuracy_trajectory: list[tuple[int, float]] = field(default_factory=list)
    block_intervals: list[tuple[int, int, Optional[int]]] = field(default_factory=list)
    eval_intervals: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class TrainStep:
    iter: int
    step: int
    step_time: float
    loss: float
    samples: Optional[int]
    val_loss: Optional[float]


def parse_mllog(path: Path) -> MllogSummary:
    summary = MllogSummary()
    pending_block_start: Optional[tuple[int, Optional[int]]] = None
    pending_eval_start: Optional[int] = None

    with path.open("r", errors="replace") as fh:
        for raw in fh:
            if MLLOG_PREFIX not in raw:
                continue
            match = MLLOG_RE.search(raw)
            if not match:
                continue
            try:
                event = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

            key = event.get("key")
            time_ms = event.get("time_ms")
            value = event.get("value")
            metadata = event.get("metadata") or {}

            if key == "submission_benchmark":
                summary.benchmark = value
            elif key == "submission_org":
                summary.org = value
            elif key == "submission_division":
                summary.division = value
            elif key == "submission_status":
                summary.status_submission = value
            elif key == "submission_platform":
                summary.platform = value
            elif key == "global_batch_size":
                summary.global_batch_size = int(value)
            elif key == "gradient_accumulation_steps":
                summary.gradient_accumulation_steps = int(value)
            elif key == "max_sequence_length":
                summary.max_sequence_length = int(value)
            elif key == "seed":
                summary.seed = int(value)
            elif key == "train_samples":
                summary.train_samples = int(value)
            elif key == "init_start":
                summary.init_start_ms = time_ms
            elif key == "init_stop":
                summary.init_stop_ms = time_ms
            elif key == "run_start":
                summary.run_start_ms = time_ms
            elif key == "run_stop":
                summary.run_stop_ms = time_ms
                summary.run_stop_status = metadata.get("status")
                summary.run_stop_step = metadata.get("step")
                summary.run_stop_samples = metadata.get("samples_count")
            elif key == "eval_accuracy":
                samples = metadata.get("samples_count")
                if value is not None and samples is not None:
                    summary.eval_accuracy_trajectory.append((int(samples), float(value)))
            elif key == "block_start":
                pending_block_start = (time_ms, metadata.get("samples_count"))
            elif key == "block_stop":
                if pending_block_start is not None and time_ms is not None:
                    start_ms, start_samples = pending_block_start
                    end_samples = metadata.get("samples_count")
                    samples_in_block = None
                    if end_samples is not None and start_samples is not None:
                        samples_in_block = int(end_samples) - int(start_samples)
                    summary.block_intervals.append((start_ms, time_ms, samples_in_block))
                    pending_block_start = None
            elif key == "eval_start":
                pending_eval_start = time_ms
            elif key == "eval_stop":
                if pending_eval_start is not None and time_ms is not None:
                    summary.eval_intervals.append((pending_eval_start, time_ms))
                    pending_eval_start = None

    return summary


def parse_train_steps(path: Path) -> list[TrainStep]:
    steps: list[TrainStep] = []
    with path.open("r", errors="replace") as fh:
        for raw in fh:
            if "train_step_timing" not in raw:
                continue
            match = TRAIN_STEP_RE.search(raw)
            if not match:
                continue
            steps.append(
                TrainStep(
                    iter=int(match.group("iter")),
                    step=int(match.group("step")),
                    step_time=float(match.group("step_time")),
                    loss=float(match.group("loss")),
                    samples=int(match.group("samples")) if match.group("samples") else None,
                    val_loss=float(match.group("val_loss")) if match.group("val_loss") else None,
                )
            )
    return steps


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _fmt_ms(ms: Optional[int]) -> str:
    if ms is None:
        return "n/a"
    return f"{ms / 1000:.2f} s"


def _fmt_min(ms: Optional[int]) -> str:
    if ms is None:
        return "n/a"
    return f"{ms / 60000:.3f} min"


def _guess_size(benchmark: Optional[str]) -> str:
    if not benchmark:
        return "8b"
    bn = benchmark.lower()
    for key in ("8b", "70b", "405b"):
        if key in bn:
            return key
    return "8b"


def _guess_n_gpus(platform: Optional[str], steps: list[TrainStep]) -> Optional[int]:
    # platform like "1280xNVIDIA GB200" → 1280. Fallback: caller provides via CLI.
    if not platform:
        return None
    match = re.match(r"(\d+)\s*x", platform)
    if match:
        return int(match.group(1))
    return None


def _print_mllog_section(summary: MllogSummary) -> None:
    print("=" * 78)
    print("MLPerf submission header")
    print("=" * 78)
    fields = [
        ("benchmark", summary.benchmark),
        ("division", summary.division),
        ("org", summary.org),
        ("platform", summary.platform),
        ("submission_status", summary.status_submission),
        ("global_batch_size", summary.global_batch_size),
        ("gradient_accumulation_steps", summary.gradient_accumulation_steps),
        ("max_sequence_length", summary.max_sequence_length),
        ("seed", summary.seed),
        ("train_samples (target)", summary.train_samples),
    ]
    for name, value in fields:
        print(f"  {name:32s} {value}")


def _print_time_section(summary: MllogSummary) -> None:
    print()
    print("=" * 78)
    print("MLPerf timing")
    print("=" * 78)

    init_ms = None
    if summary.init_start_ms is not None and summary.init_stop_ms is not None:
        init_ms = summary.init_stop_ms - summary.init_start_ms
    run_ms = None
    if summary.run_start_ms is not None and summary.run_stop_ms is not None:
        run_ms = summary.run_stop_ms - summary.run_start_ms

    submittable = summary.run_stop_status == "success"

    print(f"  init_start -> init_stop       {_fmt_ms(init_ms):>14}")
    print(f"  run_start  -> run_stop        {_fmt_ms(run_ms):>14}   "
          f"= {_fmt_min(run_ms)}  <<< MLPerf headline")
    print(f"  run_stop.status               {summary.run_stop_status!r}"
          + ("   [SUBMITTABLE]" if submittable else "   [NOT submittable]"))
    print(f"  run_stop.step / samples_count {summary.run_stop_step} / {summary.run_stop_samples}")

    if summary.block_intervals:
        total_block_ms = sum(end - start for start, end, _ in summary.block_intervals)
        print(f"  sum(block intervals)          {_fmt_ms(total_block_ms):>14}   "
              f"(n={len(summary.block_intervals)})")
    if summary.eval_intervals:
        total_eval_ms = sum(end - start for start, end in summary.eval_intervals)
        print(f"  sum(eval  intervals)          {_fmt_ms(total_eval_ms):>14}   "
              f"(n={len(summary.eval_intervals)})")


def _print_perf_section(
    steps: list[TrainStep],
    summary: MllogSummary,
    n_gpus: int,
    peak_bf16_tflops: float,
    size: str,
) -> None:
    print()
    print("=" * 78)
    print(f"Per-iteration perf (n_gpus={n_gpus}, size={size}, peak BF16={peak_bf16_tflops} TFLOP/s)")
    print("=" * 78)

    if len(steps) < 2:
        print("  Not enough iterations to compute steady-state perf.")
        return

    cold = steps[0]
    steady = [s.step_time for s in steps[1:]]
    mean = statistics.fmean(steady)
    p50 = _percentile(steady, 50)
    p95 = _percentile(steady, 95)
    sigma = statistics.stdev(steady) if len(steady) >= 2 else 0.0

    gbs = summary.global_batch_size or 0
    seq = summary.max_sequence_length or 0
    n_params = PARAM_COUNT.get(size)

    tokens_per_step = gbs * seq if gbs and seq else 0
    cluster_samples_s = gbs / mean if mean > 0 else 0
    cluster_tokens_s = tokens_per_step / mean if mean > 0 else 0
    per_gpu_tokens_s = cluster_tokens_s / n_gpus if n_gpus else 0
    per_gpu_tflops = (
        (6.0 * n_params * tokens_per_step / mean / n_gpus / 1e12)
        if (mean > 0 and n_params and n_gpus and tokens_per_step)
        else 0.0
    )
    mfu = per_gpu_tflops / peak_bf16_tflops if peak_bf16_tflops else 0.0

    print(f"  iter 0 (cold)                 {cold.step_time:8.3f} s  (dropped from aggregates)")
    print(f"  iter>=1 count                 {len(steady):8d}")
    print(f"  step_time mean/p50/p95/stdev  "
          f"{mean:7.3f} / {p50:7.3f} / {p95:7.3f} / {sigma:.3f} s")
    print()
    print(f"  cluster samples/s             {cluster_samples_s:8.2f}")
    print(f"  cluster tokens/s              {cluster_tokens_s:12.0f}")
    print(f"  per-GPU tokens/s              {per_gpu_tokens_s:12.0f}")
    if n_params:
        print(f"  per-GPU TFLOP/s (BF16)        {per_gpu_tflops:8.1f}")
        print(f"  MFU vs peak {peak_bf16_tflops:6.0f}         "
              f"{mfu * 100:6.2f} %")
    else:
        print(f"  per-GPU TFLOP/s (BF16)        n/a (unknown model size)")


def _print_loss_section(steps: list[TrainStep], summary: MllogSummary) -> None:
    print()
    print("=" * 78)
    print("Loss trajectory")
    print("=" * 78)
    if not steps:
        print("  No train_step_timing lines parsed.")
        return
    first = steps[0]
    last = steps[-1]
    print(f"  first iter  (step={first.step:3d}) loss={first.loss:8.3f}  "
          f"step_time={first.step_time:.2f}s")
    print(f"  last  iter  (step={last.step:3d}) loss={last.loss:8.3f}  "
          f"step_time={last.step_time:.2f}s")

    if summary.eval_accuracy_trajectory:
        print("  eval_accuracy (log ppl):")
        for samples, val in summary.eval_accuracy_trajectory:
            print(f"    samples_count={samples:>10}   value={val:.4f}")


def _compute_result_dict(
    summary: MllogSummary,
    steps: list[TrainStep],
    n_gpus: int,
    size: str,
    peak_bf16_tflops: float,
) -> dict:
    result: dict = {
        "benchmark": summary.benchmark,
        "division": summary.division,
        "platform": summary.platform,
        "global_batch_size": summary.global_batch_size,
        "max_sequence_length": summary.max_sequence_length,
        "n_gpus": n_gpus,
        "size": size,
    }
    if summary.run_start_ms is not None and summary.run_stop_ms is not None:
        result["time_to_train_sec"] = (summary.run_stop_ms - summary.run_start_ms) / 1000.0
        result["time_to_train_min"] = (summary.run_stop_ms - summary.run_start_ms) / 60000.0
    result["run_stop_status"] = summary.run_stop_status
    result["submittable"] = summary.run_stop_status == "success"

    steady = [s.step_time for s in steps[1:]] if len(steps) >= 2 else []
    if steady:
        mean = statistics.fmean(steady)
        result["step_time_mean_s"] = mean
        result["step_time_p50_s"] = _percentile(steady, 50)
        result["step_time_p95_s"] = _percentile(steady, 95)
        result["step_time_stdev_s"] = statistics.stdev(steady) if len(steady) >= 2 else 0.0

        gbs = summary.global_batch_size or 0
        seq = summary.max_sequence_length or 0
        n_params = PARAM_COUNT.get(size)
        if gbs and mean > 0:
            result["cluster_samples_per_s"] = gbs / mean
        if gbs and seq and mean > 0:
            result["cluster_tokens_per_s"] = gbs * seq / mean
            if n_gpus:
                result["per_gpu_tokens_per_s"] = gbs * seq / mean / n_gpus
        if n_params and gbs and seq and n_gpus and mean > 0:
            tflops = 6.0 * n_params * gbs * seq / mean / n_gpus / 1e12
            result["per_gpu_tflops_bf16"] = tflops
            if peak_bf16_tflops:
                result["mfu_pct"] = 100.0 * tflops / peak_bf16_tflops

    return result


def _print_json_tail(result: dict) -> None:
    print()
    print("=" * 78)
    print("JSON tail (machine-readable)")
    print("=" * 78)
    print(json.dumps(result, indent=2))


# Keys emitted (in order) to the MAD-compatible perf CSV. Keys present in the
# computed result dict are written; missing keys are skipped silently.
MAD_CSV_KEYS: list[tuple[str, str]] = [
    ("time_to_train_min", "time_to_train_min"),
    ("step_time_mean_s", "step_time_mean_s"),
    ("step_time_p50_s", "step_time_p50_s"),
    ("step_time_p95_s", "step_time_p95_s"),
    ("step_time_stdev_s", "step_time_stdev_s"),
    ("cluster_samples_per_s", "cluster_samples_per_s"),
    ("cluster_tokens_per_s", "cluster_tokens_per_s"),
    ("per_gpu_tokens_per_s", "per_gpu_tokens_per_s"),
    ("per_gpu_tflops_bf16", "per_gpu_tflops_bf16"),
    ("mfu_pct", "mfu_pct"),
]


def _write_mad_csv(
    csv_path: Path,
    model_name: str,
    result: dict,
    n_nodes: Optional[int],
    gpus_per_node: Optional[int],
    extra_tags: Optional[list[str]] = None,
) -> None:
    # MAD's convention is a flat CSV with columns ``model,performance,metric``.
    # One row per metric; the "metric" column identifies what the "performance"
    # value means. MAD scrapes this file and surfaces it in the results table.
    rows: list[tuple[str, str]] = []
    for csv_key, result_key in MAD_CSV_KEYS:
        if result_key in result and result[result_key] is not None:
            value = result[result_key]
            if isinstance(value, float):
                rows.append((f"{value:.6g}", csv_key))
            else:
                rows.append((str(value), csv_key))

    if n_nodes is not None:
        rows.append((str(n_nodes), "requested_nodes"))
    if gpus_per_node is not None:
        rows.append((str(gpus_per_node), "gpus_per_node"))
    if result.get("global_batch_size") is not None:
        rows.append((str(result["global_batch_size"]), "global_batch_size"))
    if result.get("n_gpus") is not None:
        rows.append((str(result["n_gpus"]), "total_gpus"))

    # Keep the legacy status marker so MAD's existing ``torchrun_launch_success``
    # detection still sees a green signal even when the perf numbers are the
    # primary value. Also include the submittable/status flag from MLLOG.
    status_marker = "torchrun_launch_success" if result.get("submittable") is not None else "launch_success"
    rows.append(("1", status_marker))
    run_status = result.get("run_stop_status")
    if run_status:
        rows.append(("1", f"run_stop_{run_status}"))

    if extra_tags:
        for tag in extra_tags:
            rows.append(("1", tag))

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w") as fh:
        fh.write("model,performance,metric\n")
        for performance, metric in rows:
            fh.write(f"{model_name},{performance},{metric}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="Path to rank-0 stdout log")
    parser.add_argument(
        "--n-gpus",
        type=int,
        default=None,
        help="Override GPU count for per-GPU metrics (defaults to "
        "parsing from submission_platform, else reports n/a).",
    )
    parser.add_argument(
        "--size",
        default=None,
        choices=sorted(PARAM_COUNT.keys()),
        help="Model size for FLOPs estimation (default: parsed from submission_benchmark).",
    )
    parser.add_argument(
        "--peak-bf16-tflops",
        type=float,
        default=DEFAULT_PEAK_BF16_TFLOPS,
        help=f"Per-GPU BF16 peak for MFU (default {DEFAULT_PEAK_BF16_TFLOPS} = MI325X).",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional path to write MAD-compatible perf CSV "
        "(model,performance,metric columns). Overwrites the file.",
    )
    parser.add_argument(
        "--csv-model-name",
        default=None,
        help="Model name used in the first CSV column (default: derived from benchmark).",
    )
    parser.add_argument(
        "--csv-nodes",
        type=int,
        default=None,
        help="Node count to record in the CSV (metric=requested_nodes).",
    )
    parser.add_argument(
        "--csv-gpus-per-node",
        type=int,
        default=None,
        help="GPUs-per-node to record in the CSV (metric=gpus_per_node).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress human-readable report; useful when only --csv-out matters.",
    )
    args = parser.parse_args()

    if not args.log.exists():
        print(f"ERROR: log file not found: {args.log}", file=sys.stderr)
        return 2

    summary = parse_mllog(args.log)
    steps = parse_train_steps(args.log)

    size = args.size or _guess_size(summary.benchmark)
    n_gpus = args.n_gpus or _guess_n_gpus(summary.platform, steps) or 0
    if n_gpus == 0:
        print(
            "WARNING: n_gpus unknown (submission_platform did not encode it and "
            "--n-gpus not provided). Per-GPU metrics will be zero; pass --n-gpus.",
            file=sys.stderr,
        )

    result = _compute_result_dict(summary, steps, n_gpus, size, args.peak_bf16_tflops)

    if not args.quiet:
        _print_mllog_section(summary)
        _print_time_section(summary)
        _print_perf_section(steps, summary, n_gpus, args.peak_bf16_tflops, size)
        _print_loss_section(steps, summary)
        _print_json_tail(result)

    if args.csv_out is not None:
        model_name = args.csv_model_name
        if model_name is None:
            # Derive ``Llama-3.1-8B`` from ``llama31_8b`` submission label.
            bench = summary.benchmark or f"llama31_{size}"
            match = re.match(r"llama31_(\d+b)", bench)
            if match:
                model_name = f"Llama-3.1-{match.group(1).upper()}"
            else:
                model_name = bench
        _write_mad_csv(
            args.csv_out,
            model_name,
            result,
            n_nodes=args.csv_nodes,
            gpus_per_node=args.csv_gpus_per_node,
        )
        if not args.quiet:
            print(f"\nMAD CSV written to {args.csv_out}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
