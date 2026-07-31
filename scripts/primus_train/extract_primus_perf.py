#!/usr/bin/env python3
"""
Extract Primus performance metrics from training log and write
madengine multiple_results CSV (one row per metric).

Supports two log formats:

1. Torchtitan/Primus step-line format:
   step: 50  loss: ...  tps: 1,444  tflops: 300.32  mfu: 23.10%

2. Megatron 26.5+ format:
   tokens/s/GPU inst/harmonic mean: 9640.7/9629.8
   compute per GPU (TFLOP/s/GPU): 496.3 (avg 496.1)

3. Megatron <=26.4 format:
   tokens per GPU (tokens/s/GPU): 9640.7
   throughput per GPU (TFLOP/s/GPU): 496.3

Output CSV format (model, performance, metric) — one row per metric:
  model,performance,metric
  primus_run,9629.8,tokens_per_second
  primus_run,496.1,tflops
  primus_run,23.10,model_flops_utilization
"""
import argparse
import csv
import re
import sys


def extract_metrics(log_path: str) -> dict:
    """Parse log file and return tps, tflops, mfu from the last matching lines."""
    tps = tflops = mfu = None

    # Torchtitan format regexes
    tt_tps_re = re.compile(r"tps:\s*([0-9][0-9.,eE+-]*)")
    tt_tflops_re = re.compile(r"tflops:\s*([0-9][0-9.eE+-]*)")
    tt_mfu_re = re.compile(r"mfu:\s*([0-9][0-9.]*)%?")

    # Megatron 26.5+ format regexes
    meg_tps_new_re = re.compile(r'tokens/s/GPU inst/harmonic mean:\s*[\d.]+/([\d.]+)')
    meg_tflops_avg_re = re.compile(r'compute per GPU \(TFLOP/s/GPU\):\s*[\d.]+\s+\(avg\s+([\d.]+)\)')
    meg_tflops_new_re = re.compile(r'compute per GPU \(TFLOP/s/GPU\):\s*([\d.]+)')

    # Megatron <=26.4 format regexes
    meg_tps_old_re = re.compile(r'tokens per GPU \(tokens/s/GPU\):\s*([\d.]+)')
    meg_tflops_old_re = re.compile(r'throughput per GPU \(TFLOP/s/GPU\):\s*([\d.]+)')

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                # Torchtitan format: single line with tps + tflops + mfu
                if "tps:" in line and "tflops:" in line and "mfu:" in line:
                    m = tt_tps_re.search(line)
                    if m:
                        tps = m.group(1).replace(",", "").strip()
                    m = tt_tflops_re.search(line)
                    if m:
                        tflops = m.group(1).strip()
                    m = tt_mfu_re.search(line)
                    if m:
                        mfu = m.group(1).strip()
                    continue

                # Megatron 26.5+ TPS (harmonic mean)
                m = meg_tps_new_re.search(line)
                if m:
                    tps = m.group(1).strip()
                    continue

                # Megatron 26.5+ TFLOPS (with avg)
                m = meg_tflops_avg_re.search(line)
                if m:
                    tflops = m.group(1).strip()
                    continue

                # Megatron 26.5+ TFLOPS (no avg)
                m = meg_tflops_new_re.search(line)
                if m:
                    tflops = m.group(1).strip()
                    continue

                # Megatron <=26.4 TPS
                m = meg_tps_old_re.search(line)
                if m:
                    tps = m.group(1).strip()
                    continue

                # Megatron <=26.4 TFLOPS
                m = meg_tflops_old_re.search(line)
                if m:
                    tflops = m.group(1).strip()
                    continue

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
        print(f"Error: No TPS metric found in log {args.log_path}", file=sys.stderr)
        print("Expected one of:", file=sys.stderr)
        print("  - Torchtitan: 'tps: <value>'", file=sys.stderr)
        print("  - Megatron 26.5+: 'tokens/s/GPU inst/harmonic mean: X/Y'", file=sys.stderr)
        print("  - Megatron <=26.4: 'tokens per GPU (tokens/s/GPU): X'", file=sys.stderr)
        sys.exit(1)

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
