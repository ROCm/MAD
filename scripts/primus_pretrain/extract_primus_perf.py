#!/usr/bin/env python3
"""
Extract Primus/Torchtitan performance metrics from training log and write
madengine multiple_results CSV (one row per metric).

Expected log line (last step):
  step: 50  loss: ...  tps: 1,444  tflops: 300.32  mfu: 23.10%

Output CSV format (model, performance, metric) — one row per metric:
  model,performance,metric
  primus_run,1444,tokens_per_second
  primus_run,300.32,tflops
  primus_run,23.10,model_flops_utilization
"""
import argparse
import csv
import re
import sys


def extract_metrics(log_path: str) -> dict:
    """Parse log file and return tps, tflops, mfu from the last step line."""
    tps = tflops = mfu = None
    # Match lines containing step, tps, tflops, mfu (e.g. Torchtitan/Primus format)
    tps_re = re.compile(r"tps:\s*([0-9][0-9.,eE+-]*)")
    tflops_re = re.compile(r"tflops:\s*([0-9][0-9.eE+-]*)")
    mfu_re = re.compile(r"mfu:\s*([0-9][0-9.]*)%?")

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "tps:" in line and "tflops:" in line and "mfu:" in line:
                    m = tps_re.search(line)
                    if m:
                        tps = m.group(1).replace(",", "").strip()
                    m = tflops_re.search(line)
                    if m:
                        tflops = m.group(1).strip()
                    m = mfu_re.search(line)
                    if m:
                        mfu = m.group(1).strip()
    except OSError as e:
        print(f"Error reading log {log_path}: {e}", file=sys.stderr)
        return {}

    return {"tps": tps, "tflops": tflops, "mfu": mfu}


def main():
    parser = argparse.ArgumentParser(description="Extract Primus perf metrics to multiple_results CSV")
    parser.add_argument("log_path", help="Path to training log (e.g. output/log_mp_pretrain_*.txt)")
    parser.add_argument("output_csv", help="Path to output CSV (e.g. run_directory/primus_perf_output.csv)")
    parser.add_argument("--model-id", default="primus_run", help="Model id for the CSV rows")
    args = parser.parse_args()

    metrics = extract_metrics(args.log_path)
    if not metrics or metrics.get("tps") is None:
        print("Warning: No tps/tflops/mfu found in log; writing empty rows.", file=sys.stderr)
        metrics = {"tps": "", "tflops": "", "mfu": ""}

    # One row per metric: model, performance, metric
    rows = [
        {"model": args.model_id, "performance": metrics.get("tps") or "", "metric": "tokens_per_second"},
        {"model": args.model_id, "performance": metrics.get("tflops") or "", "metric": "tflops"},
        {"model": args.model_id, "performance": metrics.get("mfu") or "", "metric": "model_flops_utilization"},
    ]

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "performance", "metric"])
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Wrote {args.output_csv}: {len(rows)} rows (tokens_per_second={rows[0]['performance']}, "
        f"tflops={rows[1]['performance']}, model_flops_utilization={rows[2]['performance']})"
    )


if __name__ == "__main__":
    main()
