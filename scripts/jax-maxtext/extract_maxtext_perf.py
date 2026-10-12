#!/usr/bin/env python3
"""
Extract JAX/MaxText performance metrics from a training log and write a madengine
multiple_results CSV (one row per metric).

MaxText-only: this parser handles the MaxText per-step log format and nothing else
(no Megatron/TorchTitan formats). MaxText prints per-step lines such as:

   completed step: 50, seconds: 0.85, TFLOP/s/device: 421.3, Tokens/s/device: 12345.6, ...

Values are averaged over the last N steps, matching the retired JAX report script.

Output CSV format (model, performance, metric) — one row per metric. The source values are
per-device (Tokens/s/device, TFLOP/s/device), i.e. per-GPU, so the metric names match the
per-GPU convention used by the existing MAD JAX/MaxText perf CSVs:
  model,performance,metric
  maxtext_run,12345.6,tok_per_s_per_gpu
  maxtext_run,421.3,TFLOPS_per_gpu
"""
import argparse
import csv
import re
import sys

# Number of trailing per-step samples to average (matches the old JAX report).
AVG_WINDOW = 10


def extract_metrics(log_path: str) -> dict:
    """Parse a MaxText log and return averaged tps/tflops from the trailing steps."""
    tps_re = re.compile(r'Tokens/s/device:\s*([0-9][0-9.eE+-]*)')
    tflops_re = re.compile(r'TFLOP/s/device:\s*([0-9][0-9.eE+-]*)')
    tps_samples = []
    tflops_samples = []

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = tps_re.search(line)
                if m:
                    try:
                        tps_samples.append(float(m.group(1)))
                    except ValueError:
                        pass
                m = tflops_re.search(line)
                if m:
                    try:
                        tflops_samples.append(float(m.group(1)))
                    except ValueError:
                        pass
    except OSError as e:
        print(f"Error reading log {log_path}: {e}", file=sys.stderr)
        return {}

    tps = tflops = None
    if tps_samples:
        window = tps_samples[-AVG_WINDOW:]
        tps = f"{sum(window) / len(window):.4f}"
    if tflops_samples:
        window = tflops_samples[-AVG_WINDOW:]
        tflops = f"{sum(window) / len(window):.4f}"

    return {"tps": tps, "tflops": tflops}


def main():
    parser = argparse.ArgumentParser(description="Extract MaxText perf metrics to multiple_results CSV")
    parser.add_argument("log_path", help="Path to training log (e.g. output/log_mp_pretrain_*.txt)")
    parser.add_argument("output_csv", help="Path to output CSV (e.g. run_directory/primus_perf_output.csv)")
    parser.add_argument("--model-id", default="maxtext_run", help="Model id for the CSV rows")
    args = parser.parse_args()

    metrics = extract_metrics(args.log_path)
    if not metrics or metrics.get("tps") is None:
        print(f"Error: No 'Tokens/s/device:' metric found in log {args.log_path}", file=sys.stderr)
        sys.exit(1)

    rows = [
        {"model": args.model_id, "performance": metrics.get("tps") or "", "metric": "tok_per_s_per_gpu"},
        {"model": args.model_id, "performance": metrics.get("tflops") or "", "metric": "TFLOPS_per_gpu"},
    ]

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "performance", "metric"])
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Wrote {args.output_csv}: {len(rows)} rows "
        f"(tok_per_s_per_gpu={rows[0]['performance']}, TFLOPS_per_gpu={rows[1]['performance']})"
    )


if __name__ == "__main__":
    main()
