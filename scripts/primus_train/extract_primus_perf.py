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

MFU fallback: Megatron 26.5+ logs (unlike Torchtitan) do not print an
"mfu:" field at all, so MFU is instead estimated here as
tflops / gpu_peak_tflops * 100, using dense (no-sparsity) matrix peak
TFLOPS from AMD's published data sheets. The GPU model is read from
MAD_SYSTEM_GPU_PRODUCT_NAME/MAD_SYSTEM_GPU_ARCHITECTURE (set by madengine
in the container env), and precision (bf16 vs fp8) is inferred from the
log filename (e.g. "...-FP8-pretrain.txt" vs "...-BF16-pretrain.txt").
If either can't be determined, model_flops_utilization is left blank
rather than guessed.
"""
import argparse
import csv
import os
import re
import sys

# Dense (no structured sparsity) BF16 matrix peak TFLOP/s per GPU, from AMD
# Instinct data sheets. FP8 dense peak is 2x BF16 dense peak on all of these
# CDNA3/CDNA4 parts. Matched against MAD_SYSTEM_GPU_PRODUCT_NAME (e.g. "AMD
# Instinct MI300X") as a substring, most-specific names first.
_PEAK_BF16_TFLOPS = {
    "MI355X": 2500.0,
    "MI350X": 2300.0,
    "MI325X": 1307.4,
    "MI300X": 1307.4,
}


def _estimate_mfu(tflops: str, log_path: str) -> str | None:
    """Estimate model FLOPs utilization (%) when the log doesn't report it."""
    try:
        achieved_tflops = float(tflops)
    except (TypeError, ValueError):
        return None

    gpu_name = os.environ.get("MAD_SYSTEM_GPU_PRODUCT_NAME") or os.environ.get(
        "MAD_SYSTEM_GPU_ARCHITECTURE", ""
    )
    peak_bf16 = next((v for k, v in _PEAK_BF16_TFLOPS.items() if k in gpu_name), None)
    if peak_bf16 is None:
        return None

    is_fp8 = "fp8" in os.path.basename(log_path).lower()
    peak = peak_bf16 * 2 if is_fp8 else peak_bf16
    return f"{(achieved_tflops / peak * 100.0):.2f}"


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
    if metrics and metrics.get("mfu") is None and metrics.get("tflops") is not None:
        metrics["mfu"] = _estimate_mfu(metrics["tflops"], args.log_path)

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
