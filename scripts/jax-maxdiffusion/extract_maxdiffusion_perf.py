#!/usr/bin/env python3
"""
Extract JAX/MaxDiffusion performance metrics and write a madengine
multiple_results CSV (one row per metric).

MaxDiffusion writes per-step metrics two ways:

  1. A per-step stdout line (parsed as a fallback)::

       completed step: 12, seconds: 0.83, TFLOP/s/device: 210.4, loss: 0.123

     Under the Primus launcher this line does NOT reliably reach the captured
     training log (raw ``print`` from maxdiffusion.max_logging is dropped while
     the trainer runs), so it cannot be relied on.

  2. A JSON-lines metrics file, written directly by the trainer when
     ``config.metrics_file`` is set (``max_utils.write_metrics_locally`` ->
     ``train_utils.write_metrics``). Each line is a dict, e.g.::

       {"perf/step_time_seconds": 0.83, "perf/per_device_tflops": 174.7,
        "perf/per_device_tflops_per_sec": 210.4, "learning/loss": 0.123,
        "step": 12.0, "run_name": "wan2.1_1.3b_pretrain"}

     This bypasses stdout entirely and is the PREFERRED source. run.sh points
     ``PERF_METRICS_FILE`` (-> config metrics_file) at a path in the persisted
     run dir and passes it here via --metrics-file.

Throughput is derived per the retired jax-maxdiffusion_benchmark_report.py:
   fps_per_gpu             = per_device_batch_size / avg_seconds_per_step
   images_per_sec_per_gpu  = per_device_batch_size * num_frames / avg_seconds_per_step
   TFLOPS_per_gpu          = avg TFLOP/s/device

batch size and frame count are read from the training log's config dump
(both the "Config param <name>: <value>" and the Primus
"<name> : <value> (<type>)" formats are recognized). Averages skip warmup steps.

Output CSV format (model, performance, metric) — matches
scripts/jax-maxtext/extract_maxtext_perf.py so both feed madengine
multiple_results (primus_perf_output.csv) identically:
  model,performance,metric
  wan2.1_1.3b-pretrain,7.23,fps_per_gpu
  wan2.1_1.3b-pretrain,585.6,images_per_sec_per_gpu
  wan2.1_1.3b-pretrain,210.4,TFLOPS_per_gpu
"""
import argparse
import csv
import json
import re
import sys

# Trailing per-step samples: skip the first SKIP_WARMUP steps, then average.
SKIP_WARMUP = 2

_STEP_RE = re.compile(
    r"completed step:\s*(\d+),\s*seconds:\s*([0-9][0-9.eE+-]*),\s*TFLOP/s/device:\s*([0-9][0-9.eE+-]*)"
)


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _config_param(content: str, name: str):
    # MaxDiffusion "Config param <name>: <value>" (raw print) ...
    m = re.search(rf"Config param {re.escape(name)}:\s*(.+)", content)
    if m:
        return _ANSI_RE.sub("", m.group(1)).strip()
    # ... or the Primus config dump "<name>  : <value> (<type>)" format.
    m = re.search(rf"(?:^|\]|\s){re.escape(name)}\s*:\s*(.+?)\s*\((?:bool|int|float|str|list|NoneType|tuple|dict)\)", content, re.MULTILINE)
    return _ANSI_RE.sub("", m.group(1)).strip() if m else None


def _parse_frames(raw):
    if raw is None:
        return None
    t = str(raw).strip().lower()
    if t in ("", "none", "null"):
        return None
    try:
        return int(float(t))
    except ValueError:
        return None


def _effective_num_frames(content: str) -> float:
    """Frames used for throughput. FLUX (image) = 1; WAN uses synthetic-override
    logic (synthetic_override_num_frames when dataset_type=synthetic, else
    num_frames / data_frames)."""
    model_name = (_config_param(content, "model_name") or "").lower()
    pretrained = (_config_param(content, "pretrained_model_name_or_path") or "").lower()
    if "flux" in f"{model_name} {pretrained}":
        return 1.0

    dataset_type = (_config_param(content, "dataset_type") or "").strip().lower()
    override = _parse_frames(_config_param(content, "synthetic_override_num_frames"))
    num_frames = _parse_frames(_config_param(content, "num_frames"))
    data_frames = _parse_frames(_config_param(content, "data_frames"))

    if dataset_type == "synthetic" and override is not None:
        chosen = override
    elif num_frames is not None:
        chosen = num_frames
    elif data_frames is not None:
        chosen = data_frames
    else:
        chosen = None
    return float(chosen) if chosen is not None else 1.0


def _samples_from_metrics_file(metrics_file: str):
    """Return (seconds[], tflops[]) parsed from the JSON-lines metrics file, or
    ([], []) if the file is missing/empty/unparseable."""
    seconds, tflops = [], []
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
                s = d.get("perf/step_time_seconds")
                t = d.get("perf/per_device_tflops_per_sec")
                if s is None or t is None:
                    continue
                try:
                    seconds.append(float(s))
                    tflops.append(float(t))
                except (TypeError, ValueError):
                    continue
    except OSError:
        return [], []
    return seconds, tflops


def _samples_from_log(content: str):
    matches = _STEP_RE.findall(content)
    seconds = [float(m[1]) for m in matches]
    tflops = [float(m[2]) for m in matches]
    return seconds, tflops


def extract_metrics(log_path: str, metrics_file: str = "") -> dict:
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except OSError as e:
        print(f"Error reading log {log_path}: {e}", file=sys.stderr)
        content = ""

    # Prefer the JSON-lines metrics file; fall back to the stdout log line.
    seconds, tflops = ([], [])
    source = ""
    if metrics_file:
        seconds, tflops = _samples_from_metrics_file(metrics_file)
        if seconds:
            source = "metrics_file"
    if not seconds:
        seconds, tflops = _samples_from_log(content)
        if seconds:
            source = "log"

    if not seconds:
        return {}

    # Drop warmup (compile) steps, then average.
    v_seconds = seconds[SKIP_WARMUP:] or seconds
    v_tflops = tflops[SKIP_WARMUP:] or tflops
    avg_seconds = sum(v_seconds) / len(v_seconds)
    avg_tflops = sum(v_tflops) / len(v_tflops)

    batch_raw = _config_param(content, "per_device_batch_size")
    try:
        batch = float(batch_raw) if batch_raw is not None else 1.0
    except ValueError:
        batch = 1.0
    frames = _effective_num_frames(content)

    fps = batch / avg_seconds if avg_seconds > 0 else 0.0
    images_per_sec = batch * frames / avg_seconds if avg_seconds > 0 else 0.0
    return {
        "fps": f"{fps:.4f}",
        "images_per_sec": f"{images_per_sec:.4f}",
        "tflops": f"{avg_tflops:.4f}",
        "_source": source,
        "_nsteps": str(len(v_seconds)),
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
    if not metrics:
        print(
            "Error: no MaxDiffusion perf metrics found. Looked in metrics-file "
            f"'{args.metrics_file}' and for 'completed step: ..., TFLOP/s/device: ...' "
            f"lines in log {args.log_path}.",
            file=sys.stderr,
        )
        sys.exit(1)

    rows = [
        {"model": args.model_id, "performance": metrics["fps"], "metric": "fps_per_gpu"},
        {"model": args.model_id, "performance": metrics["images_per_sec"], "metric": "images_per_sec_per_gpu"},
        {"model": args.model_id, "performance": metrics["tflops"], "metric": "TFLOPS_per_gpu"},
    ]

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "performance", "metric"])
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Wrote {args.output_csv}: {len(rows)} rows from {metrics.get('_source', '?')} "
        f"({metrics.get('_nsteps', '?')} steps; fps_per_gpu={rows[0]['performance']}, "
        f"images_per_sec_per_gpu={rows[1]['performance']}, TFLOPS_per_gpu={rows[2]['performance']})"
    )


if __name__ == "__main__":
    main()
