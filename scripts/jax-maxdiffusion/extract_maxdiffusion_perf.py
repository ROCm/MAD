#!/usr/bin/env python3
"""
Extract JAX/MaxDiffusion performance metrics and write a madengine
multiple_results CSV (one row per metric).

Primus's ``maxdiffusion.throughput`` patch publishes per-device throughput at
train time. This script copies those numbers into madengine ``multiple_results``;
it does not recompute fps/frames from batch size or the config dump.

Sources, preferred first:

  1. JSON-lines ``metrics_file`` (config.metrics_file). run.sh points
     ``PERF_METRICS_FILE`` at a path that outlives madengine's run-dir cleanup
     and passes it here via --metrics-file. Keys::

       perf/per_device_tflops_per_sec
       perf/samples_per_second_per_device   -> fps_per_gpu
       perf/frames_per_second_per_device    -> images_per_sec_per_gpu
       perf/tokens_per_second_per_device    -> tok_per_s_per_gpu

  2. The per-step log line, as a fallback::

       completed step: 18, seconds: 11.722, TFLOP/s/device: 348.548,
       Tokens/s/device: 6756.526, Samples/s/device: 0.0853,
       Frames/s/device: 7.251, loss: 1.540

     Tokens and Frames are omitted for families Primus does not define them for.

Averages skip the first SKIP_WARMUP steps. Existing CSV names are kept so MAD
dashboards do not break; tok_per_s_per_gpu is added when Primus emitted tokens.
"""
import argparse
import csv
import json
import re
import sys

SKIP_WARMUP = 2

# Primus patch order: TFLOP/s, optional Tokens/s, Samples/s, optional Frames/s.
_STEP_RE = re.compile(
    r"completed step:\s*\d+,\s*"
    r"seconds:\s*(?P<seconds>[0-9][0-9.eE+-]*),\s*"
    r"TFLOP/s/device:\s*(?P<tflops>[0-9][0-9.eE+-]*)"
    r"(?:,\s*Tokens/s/device:\s*(?P<tokens>[0-9][0-9.eE+-]*))?"
    r"(?:,\s*Samples/s/device:\s*(?P<samples>[0-9][0-9.eE+-]*))?"
    r"(?:,\s*Frames/s/device:\s*(?P<frames>[0-9][0-9.eE+-]*))?"
)

_JSON_KEYS = {
    "tflops": "perf/per_device_tflops_per_sec",
    "samples": "perf/samples_per_second_per_device",
    "frames": "perf/frames_per_second_per_device",
    "tokens": "perf/tokens_per_second_per_device",
}


def _as_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _avg(steps, key):
    values = [s[key] for s in steps if s.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _fmt(value):
    return f"{value:.4f}" if value is not None else None


def _samples_from_metrics_file(metrics_file: str):
    steps = []
    try:
        with open(metrics_file, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    d = json.loads(raw)
                except ValueError:
                    continue
                tflops = _as_float(d.get(_JSON_KEYS["tflops"]))
                if tflops is None:
                    continue
                steps.append(
                    {
                        "tflops": tflops,
                        "samples": _as_float(d.get(_JSON_KEYS["samples"])),
                        "frames": _as_float(d.get(_JSON_KEYS["frames"])),
                        "tokens": _as_float(d.get(_JSON_KEYS["tokens"])),
                    }
                )
    except OSError:
        return []
    return steps


def _samples_from_log(content: str):
    steps = []
    for m in _STEP_RE.finditer(content):
        tflops = _as_float(m.group("tflops"))
        if tflops is None:
            continue
        steps.append(
            {
                "tflops": tflops,
                "samples": _as_float(m.group("samples")),
                "frames": _as_float(m.group("frames")),
                "tokens": _as_float(m.group("tokens")),
            }
        )
    return steps


def extract_metrics(log_path: str, metrics_file: str = "") -> dict:
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except OSError as e:
        print(f"Error reading log {log_path}: {e}", file=sys.stderr)
        content = ""

    steps = []
    source = ""
    if metrics_file:
        steps = _samples_from_metrics_file(metrics_file)
        if steps:
            source = "metrics_file"
    if not steps:
        steps = _samples_from_log(content)
        if steps:
            source = "log"

    if not steps:
        return {}

    window = steps[SKIP_WARMUP:] or steps
    return {
        "fps": _fmt(_avg(window, "samples")),
        "images_per_sec": _fmt(_avg(window, "frames")),
        "tps": _fmt(_avg(window, "tokens")),
        "tflops": _fmt(_avg(window, "tflops")),
        "_source": source,
        "_nsteps": str(len(window)),
    }


def main():
    parser = argparse.ArgumentParser(description="Extract MaxDiffusion perf metrics to multiple_results CSV")
    parser.add_argument("log_path", help="Path to training log")
    parser.add_argument("output_csv", help="Path to output CSV (primus_perf_output.csv)")
    parser.add_argument("--model-id", default="maxdiffusion_run", help="Model id for the CSV rows")
    parser.add_argument(
        "--metrics-file",
        default="",
        help="Path to the JSON-lines metrics file written by the trainer (config.metrics_file). Preferred over the log.",
    )
    args = parser.parse_args()

    metrics = extract_metrics(args.log_path, args.metrics_file)
    if not metrics or metrics.get("tflops") is None:
        print(
            "Error: no MaxDiffusion perf metrics found. Looked in metrics-file "
            f"'{args.metrics_file}' and for 'completed step: ..., TFLOP/s/device: ...' "
            f"lines in log {args.log_path}.",
            file=sys.stderr,
        )
        sys.exit(1)

    rows = []
    if metrics.get("fps") is not None:
        rows.append({"model": args.model_id, "performance": metrics["fps"], "metric": "fps_per_gpu"})
    if metrics.get("images_per_sec") is not None:
        rows.append(
            {"model": args.model_id, "performance": metrics["images_per_sec"], "metric": "images_per_sec_per_gpu"}
        )
    if metrics.get("tps") is not None:
        rows.append({"model": args.model_id, "performance": metrics["tps"], "metric": "tok_per_s_per_gpu"})
    rows.append({"model": args.model_id, "performance": metrics["tflops"], "metric": "TFLOPS_per_gpu"})

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "performance", "metric"])
        writer.writeheader()
        writer.writerows(rows)

    summary = ", ".join(f"{r['metric']}={r['performance']}" for r in rows)
    print(
        f"Wrote {args.output_csv}: {len(rows)} rows from {metrics.get('_source', '?')} "
        f"({metrics.get('_nsteps', '?')} steps; {summary})"
    )


if __name__ == "__main__":
    main()
