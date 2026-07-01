#!/usr/bin/env python3
"""
Parse SGLang benchmark log file and save results to CSV.

Extracts a full metric set per (isl, osl, concurrency) configuration:
  - Request throughput (req/s)
  - Input / Output / Total token throughput (tok/s)
  - Mean E2E latency (ms), Mean TTFT (ms), Mean ITL (ms)

For each configuration the row from the iteration with the MAX total token
throughput is retained (all metrics come from that single best run, so they
stay mutually consistent).
"""

import re
import csv
from pathlib import Path
from typing import Dict, Tuple

# Metric key -> madengine metric name. Order defines CSV row order per config.
METRIC_FIELDS = [
    ("request_throughput_req_s", "request_throughput_req_s"),
    ("input_token_throughput_tok_s", "input_token_throughput_tok_s"),
    ("output_token_throughput_tok_s", "output_token_throughput_tok_s"),
    ("total_token_throughput_tok_s", "total_token_throughput_tok_s"),
    ("mean_e2e_latency_ms", "mean_e2e_latency_ms"),
    ("mean_ttft_ms", "mean_ttft_ms"),
    ("mean_itl_ms", "mean_itl_ms"),
]


def _extract(pattern: str, text: str):
    """Return the first capture group as float (commas stripped), else None."""
    m = re.search(pattern, text)
    return float(m.group(1).replace(",", "")) if m else None


def parse_benchmark_log(log_file: str) -> Dict[Tuple[int, int, int], Dict]:
    """Parse benchmark log, keeping the best-throughput run per configuration."""
    results: Dict[Tuple[int, int, int], Dict] = {}

    with open(log_file, "r") as f:
        content = f.read()

    # Find the start of the first iteration (ignore warmup).
    first_iter_match = re.search(r"RUNNING: the benchserving script for iter: 1", content)
    if not first_iter_match:
        print("Warning: No iteration 1 found. Processing entire file.")
        start_pos = 0
    else:
        start_pos = first_iter_match.start()

    content = content[start_pos:]

    # Split by benchmark result sections; config lives in the preceding section.
    sections = re.split(r"============ Serving Benchmark Result ============", content)

    current_isl = None
    current_osl = None
    current_con = None

    for i, section in enumerate(sections[1:], 1):  # Skip first (pre-first-result) chunk
        prev_section = sections[i - 1]

        # Config emitted by benchmark_xPyD.sh: "RUNNING: prompts  isl X osl Y con Z"
        config_match = re.search(
            r"RUNNING: prompts\s+isl\s+(\d+)\s+osl\s+(\d+)\s+con\s+(\d+)", prev_section
        )
        if config_match:
            current_isl = int(config_match.group(1))
            current_osl = int(config_match.group(2))
            current_con = int(config_match.group(3))

        if current_isl is None or current_osl is None or current_con is None:
            continue

        total_throughput = _extract(r"Total token throughput \(tok/s\):\s+([\d,\.]+)", section)
        if total_throughput is None:
            continue

        row = {
            "isl": current_isl,
            "osl": current_osl,
            "con": current_con,
            "request_throughput_req_s": _extract(
                r"Request throughput \(req/s\):\s+([\d,\.]+)", section
            ),
            "input_token_throughput_tok_s": _extract(
                r"Input token throughput \(tok/s\):\s+([\d,\.]+)", section
            ),
            "output_token_throughput_tok_s": _extract(
                r"Output token throughput \(tok/s\):\s+([\d,\.]+)", section
            ),
            "total_token_throughput_tok_s": total_throughput,
            "mean_e2e_latency_ms": _extract(r"Mean E2E Latency \(ms\):\s+([\d,\.]+)", section),
            "mean_ttft_ms": _extract(r"Mean TTFT \(ms\):\s+([\d,\.]+)", section),
            "mean_itl_ms": _extract(r"Mean ITL \(ms\):\s+([\d,\.]+)", section),
        }

        config_key = (current_isl, current_osl, current_con)
        best = results.get(config_key)
        if best is None or total_throughput > best["total_token_throughput_tok_s"]:
            results[config_key] = row

    return results


def _sorted_results(results: Dict[Tuple[int, int, int], Dict]):
    # Sort by concurrency, then isl, then osl.
    return [
        results[key]
        for key in sorted(results.keys(), key=lambda k: (k[2], k[0], k[1]))
    ]


def save_to_csv(results: Dict[Tuple[int, int, int], Dict], output_file: str):
    """Save a wide per-configuration summary CSV (all metrics)."""
    if not results:
        print("No results to save.")
        return

    fieldnames = [
        "Concurrency",
        "Input tokens",
        "Output tokens",
        "Request throughput (req/s)",
        "Input token throughput (tok/s)",
        "Output token throughput (tok/s)",
        "Total Token throughput (tok/s)",
        "Mean E2E Latency (ms)",
        "Mean TTFT (ms)",
        "Mean ITL (ms)",
    ]

    def _fmt(value):
        return f"{value:.2f}" if value is not None else ""

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for data in _sorted_results(results):
            writer.writerow(
                {
                    "Concurrency": data["con"],
                    "Input tokens": data["isl"],
                    "Output tokens": data["osl"],
                    "Request throughput (req/s)": _fmt(data["request_throughput_req_s"]),
                    "Input token throughput (tok/s)": _fmt(data["input_token_throughput_tok_s"]),
                    "Output token throughput (tok/s)": _fmt(data["output_token_throughput_tok_s"]),
                    "Total Token throughput (tok/s)": _fmt(data["total_token_throughput_tok_s"]),
                    "Mean E2E Latency (ms)": _fmt(data["mean_e2e_latency_ms"]),
                    "Mean TTFT (ms)": _fmt(data["mean_ttft_ms"]),
                    "Mean ITL (ms)": _fmt(data["mean_itl_ms"]),
                }
            )

    print(f"Saved {len(results)} benchmark configurations to {output_file}")


def _get_run_metadata(pipeline: str = "sglang"):
    """Collect run metadata from environment variables."""
    import os

    xP = os.environ.get("xP", "1")
    yD = os.environ.get("yD", "1")
    dp_mode = os.environ.get("DP_MODE", "0")
    run_mori = os.environ.get("RUN_MORI", "0")
    gpus = os.environ.get("GPUS_PER_NODE", "8")

    # Determine backend tag
    if dp_mode == "1":
        backend = "mori_dp"
    elif run_mori == "1":
        backend = "mori_io"
    else:
        backend = "mooncake"

    return {
        "pipeline": pipeline,
        "deployment_type": f"disagg_{xP}P{yD}D",
        "tags": f"{pipeline}_disagg,{backend}",
        "n_gpus": str(int(xP) * int(gpus) + int(yD) * int(gpus)),
        "nnodes": str(int(xP) + int(yD)),
        "gpus_per_node": gpus,
        "docker_image": os.environ.get("DOCKER_IMAGE_NAME", ""),
        "machine_name": os.environ.get("SLURM_JOB_NODELIST", ""),
        "launcher": "slurm_multi",
        "gpu_architecture": "gfx942",
    }


def save_perf_csv(
    results: Dict[Tuple[int, int, int], Dict],
    output_file: str,
    model_name: str = "",
    pipeline: str = "sglang",
):
    """Save results in madengine perf.csv format (one row per config x metric)."""
    if not results:
        print("No results to save to perf.csv.")
        return

    import os

    meta = _get_run_metadata(pipeline)
    xP = os.environ.get("xP", "1")
    yD = os.environ.get("yD", "1")

    fieldnames = [
        "model", "n_gpus", "nnodes", "gpus_per_node", "training_precision",
        "pipeline", "args", "tags", "docker_file", "base_docker", "docker_sha",
        "docker_image", "git_commit", "machine_name", "deployment_type", "launcher",
        "gpu_architecture", "performance", "metric", "relative_change", "status",
        "build_duration", "test_duration", "dataname", "data_provider_type",
        "data_size", "data_download_duration", "build_number",
        "additional_docker_run_options",
    ]

    rows_written = 0
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for data in _sorted_results(results):
            config_key = (
                f"{xP}p{yD}d_isl{data['isl']}_osl{data['osl']}_con{data['con']}"
            )
            for source_key, metric_name in METRIC_FIELDS:
                value = data.get(source_key)
                if value is None:
                    continue
                row = {
                    "model": config_key,
                    "performance": f"{value:.2f}",
                    "metric": metric_name,
                    "status": "SUCCESS",
                }
                row.update(meta)
                # model_name (catalog key) is prepended by madengine; keep the
                # per-config key in `model` so metrics stay disambiguated.
                writer.writerow(row)
                rows_written += 1

    print(f"Saved {rows_written} rows to perf.csv: {output_file}")


def main():
    """Main function."""
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse SGLang benchmark log file and save results to CSV"
    )
    parser.add_argument("log_file", type=str, help="Path to benchmark log file")
    parser.add_argument(
        "-o", "--output", type=str,
        help="Output CSV file name (default: <log_file>_results.csv)",
    )
    parser.add_argument(
        "--perf-csv", type=str, help="Also generate madengine perf.csv at this path"
    )
    parser.add_argument("--model-name", type=str, default="", help="Model name for perf.csv")

    args = parser.parse_args()

    log_file = args.log_file

    if not Path(log_file).exists():
        print(f"Error: Log file not found: {log_file}")
        sys.exit(1)

    print(f"Parsing log file: {log_file}")

    results = parse_benchmark_log(log_file)

    if not results:
        print("No benchmark results found in log file.")
        sys.exit(1)

    if args.output:
        output_file = args.output
    else:
        output_file = Path(log_file).stem + "_results.csv"

    save_to_csv(results, output_file)

    if args.perf_csv:
        save_perf_csv(results, args.perf_csv, args.model_name)

    print(f"\nSummary:")
    print(f"  Total unique configurations: {len(results)}")
    print(f"  Output file: {output_file}")


if __name__ == "__main__":
    main()
