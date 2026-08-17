#!/usr/bin/env python3
"""
Parse vLLM benchmark log file and save results to CSV.
Extracts: Concurrency, Input tokens, Output tokens, Total Token throughput (tok/s)
For each configuration, takes the MAX Total Token throughput across all iterations.

Log format (from benchmark_xPyD.sh):
  [RUNNING] prompts <N> isl <ISL> osl <OSL> con <CON> (timeout <T>s)
  ============ Serving Benchmark Result ============
  Total token throughput (tok/s):          <VALUE>
"""

import re
import csv
from pathlib import Path
from typing import Dict, Tuple
from collections import defaultdict


def parse_benchmark_log(log_file: str) -> Dict[Tuple[int, int, int], Dict]:
    """Parse benchmark log file and extract results, keeping max throughput per configuration."""
    results = defaultdict(lambda: {'concurrency': None, 'input_tokens': None,
                                    'output_tokens': None, 'max_throughput': 0.0})

    with open(log_file, 'r') as f:
        content = f.read()

    # Find the start of the first iteration (ignore warmup)
    first_iter_match = re.search(r'Running the benchserving script for iter: 1', content)
    if not first_iter_match:
        print("Warning: No iteration 1 found. Processing entire file.")
        start_pos = 0
    else:
        start_pos = first_iter_match.start()

    # Process only from first iteration onwards
    content = content[start_pos:]

    # Split by benchmark result sections
    sections = re.split(r'============ Serving Benchmark Result ============', content)

    current_input_seq_len = None
    current_output_seq_len = None
    current_concurrency = None

    for i, section in enumerate(sections[1:], 1):  # Skip first empty section
        # Look for configuration in previous sections (from [RUNNING] line)
        if i > 1:
            prev_section = sections[i-1]

            # vllm format: [RUNNING] prompts <N> isl <ISL> osl <OSL> con <CON>
            config_match = re.search(
                r'\[RUNNING\]\s+prompts\s+\d+\s+isl\s+(\d+)\s+osl\s+(\d+)\s+con\s+(\d+)',
                prev_section
            )
            # Fallback: extract from Namespace(...) in vllm bench serve output
            if not config_match:
                isl_m = re.search(r'random_input_len=(\d+)', prev_section)
                osl_m = re.search(r'random_output_len=(\d+)', prev_section)
                con_m = re.search(r'max_concurrency=(\d+)', prev_section)
                if isl_m and osl_m and con_m:
                    config_match = type('Match', (), {
                        'group': lambda self, n: [None, isl_m.group(1), osl_m.group(1), con_m.group(1)][n]
                    })()
            if config_match:
                current_input_seq_len = int(config_match.group(1))
                current_output_seq_len = int(config_match.group(2))
                current_concurrency = int(config_match.group(3))

        # Extract Total token throughput (tok/s) from benchmark result section
        throughput_match = re.search(r'Total token throughput \(tok/s\):\s+([\d.]+)', section)
        throughput = float(throughput_match.group(1)) if throughput_match else None

        # Only process if we have a valid configuration from [RUNNING] line and throughput
        if current_input_seq_len and current_output_seq_len and current_concurrency and throughput is not None:
            config_key = (current_input_seq_len, current_output_seq_len, current_concurrency)

            results[config_key]['concurrency'] = current_concurrency
            results[config_key]['input_tokens'] = current_input_seq_len
            results[config_key]['output_tokens'] = current_output_seq_len

            # Keep the maximum throughput
            if throughput > results[config_key]['max_throughput']:
                results[config_key]['max_throughput'] = throughput

    return results


def save_to_csv(results: Dict[Tuple[int, int, int], Dict], output_file: str):
    """Save results to CSV file with specified columns."""
    if not results:
        print("No results to save.")
        return

    fieldnames = ['Concurrency', 'Input tokens', 'Output tokens', 'Total Token throughput (tok/s)']

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Sort by concurrency, then input tokens, then output tokens
        for (input_tokens, output_tokens, concurrency), data in sorted(results.items(),
                                                                         key=lambda x: (x[0][2], x[0][0], x[0][1])):
            row = {
                'Concurrency': data['concurrency'],
                'Input tokens': data['input_tokens'],
                'Output tokens': data['output_tokens'],
                'Total Token throughput (tok/s)': f"{data['max_throughput']:.2f}"
            }
            writer.writerow(row)

    print(f"Saved {len(results)} benchmark configurations to {output_file}")


def _get_run_metadata(pipeline: str = "vllm"):
    """Collect run metadata from environment variables.

    Two launchers share this parser, and they describe their topology differently:

      * vllm_dissag  -> disaggregated, xP prefill + yD decode nodes.
      * vllm_multinode -> COLOCATED, one instance spanning NNODES nodes. It exports
        xP=1 yD=0 purely so the shared benchmark log filenames stay unique.

    Deriving the topology from xP/yD is therefore only valid for the disagg path;
    on the colocated path it reported a 2-node/16-GPU run as `disagg_1P0D` with
    1 node and 8 GPUs. NNODES is exported by both launchers and is authoritative,
    and a launcher whose shape is not "xP prefill + yD decode" states its own
    deployment_type/tags via PERF_DEPLOYMENT_TYPE / PERF_TAGS.
    """
    import os
    xP = os.environ.get('xP', '1')
    yD = os.environ.get('yD', '1')
    run_mori = os.environ.get('RUN_MORI', '0')
    run_deepep = os.environ.get('RUN_DEEPEP', '0')
    gpus = os.environ.get('GPUS_PER_NODE', '8')

    # Determine backend tag
    if run_mori == '1':
        backend = 'mori'
    elif run_deepep == '1':
        backend = 'deepep'
    else:
        backend = 'nixl'

    def _as_int(value, default):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    gpus_per_node = _as_int(gpus, 8)
    nnodes = _as_int(os.environ.get('NNODES'), _as_int(xP, 1) + _as_int(yD, 1))

    return {
        'pipeline': pipeline,
        'deployment_type': os.environ.get('PERF_DEPLOYMENT_TYPE') or f'disagg_{xP}P{yD}D',
        'tags': os.environ.get('PERF_TAGS') or f'{pipeline}_disagg,{backend}',
        'n_gpus': str(nnodes * gpus_per_node),
        'nnodes': str(nnodes),
        'gpus_per_node': str(gpus_per_node),
        'docker_image': os.environ.get('DOCKER_IMAGE_NAME', ''),
        'machine_name': os.environ.get('SLURM_JOB_NODELIST', ''),
        'launcher': 'slurm_multi',
        'gpu_architecture': os.environ.get('PERF_GPU_ARCH', 'gfx942'),
    }


def save_perf_csv(results: Dict[Tuple[int, int, int], Dict], output_file: str,
                  model_name: str = "", pipeline: str = "vllm"):
    """Save results in madengine perf.csv format."""
    if not results:
        print("No results to save to perf.csv.")
        return

    meta = _get_run_metadata(pipeline)

    fieldnames = [
        'model', 'n_gpus', 'nnodes', 'gpus_per_node', 'training_precision',
        'pipeline', 'args', 'tags', 'docker_file', 'base_docker', 'docker_sha',
        'docker_image', 'git_commit', 'machine_name', 'deployment_type', 'launcher',
        'gpu_architecture', 'performance', 'metric', 'relative_change', 'status',
        'build_duration', 'test_duration', 'dataname', 'data_provider_type',
        'data_size', 'data_download_duration', 'build_number',
        'additional_docker_run_options',
    ]

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for (input_tokens, output_tokens, concurrency), data in sorted(
            results.items(), key=lambda x: (x[0][2], x[0][0], x[0][1])
        ):
            row = {
                'model': model_name,
                'performance': f"{data['max_throughput']:.2f}",
                'metric': f"tok/s (isl={data['input_tokens']} osl={data['output_tokens']} con={data['concurrency']})",
                'status': 'SUCCESS',
            }
            row.update(meta)
            writer.writerow(row)

    print(f"Saved {len(results)} rows to perf.csv: {output_file}")


def parse_niah_log(log_file: str):
    """Parse a benchmark_niah.py log into {context_words: found_out_of_10}.

    benchmark_niah.py prints one line per size:
        words= 20000  found= 9/10  [...]
    and 'found=ERR' (via the summary) for a request that errored. Only the
    per-size result lines are read; the trailing summary repeats them.
    """
    results = {}
    pat = re.compile(r'^words=\s*(\d+)\s+found=\s*(\d+)/10')
    err = re.compile(r'^words=\s*(\d+)\s+ERROR')
    with open(log_file, 'r', errors='replace') as f:
        for line in f:
            line = line.strip()
            m = pat.match(line)
            if m:
                results[int(m.group(1))] = int(m.group(2))
                continue
            m = err.match(line)
            if m:
                results.setdefault(int(m.group(1)), None)
    return results


def save_niah_perf_csv(results, output_file: str, model_name: str = "",
                       pipeline: str = "vllm"):
    """Save NIAH retrieval accuracy in madengine perf.csv format.

    One row per context size; performance is needles found out of 10. A size that
    errored is reported as FAILURE with performance 0 rather than dropped, so a
    regression that turns a pass into a crash is visible instead of silent.
    """
    if not results:
        print("No NIAH results to save to perf.csv.")
        return

    meta = _get_run_metadata(pipeline)
    fieldnames = [
        'model', 'n_gpus', 'nnodes', 'gpus_per_node', 'training_precision',
        'pipeline', 'args', 'tags', 'docker_file', 'base_docker', 'docker_sha',
        'docker_image', 'git_commit', 'machine_name', 'deployment_type', 'launcher',
        'gpu_architecture', 'performance', 'metric', 'relative_change', 'status',
        'build_duration', 'test_duration', 'dataname', 'data_provider_type',
        'data_size', 'data_download_duration', 'build_number',
        'additional_docker_run_options',
    ]
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for words in sorted(results):
            found = results[words]
            row = {
                'model': model_name,
                'performance': '0' if found is None else str(found),
                'metric': f'needles found /10 (NIAH ctx={words} words)',
                'status': 'FAILURE' if found is None else 'SUCCESS',
            }
            row.update(meta)
            writer.writerow(row)
    print(f"Saved {len(results)} NIAH rows to perf.csv: {output_file}")


def main():
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Parse vLLM benchmark log file and save results to CSV')
    parser.add_argument('log_file', type=str, help='Path to benchmark log file')
    parser.add_argument('-o', '--output', type=str, help='Output CSV file name (default: <log_file>_results.csv)')
    parser.add_argument('--perf-csv', type=str, help='Also generate madengine perf.csv at this path')
    parser.add_argument('--model-name', type=str, default='', help='Model name for perf.csv')
    parser.add_argument('--niah', action='store_true',
                        help='Parse a benchmark_niah.py log (retrieval accuracy) instead of a throughput sweep')

    args = parser.parse_args()

    log_file = args.log_file

    if not Path(log_file).exists():
        print(f"Error: Log file not found: {log_file}")
        sys.exit(1)

    print(f"Parsing log file: {log_file}")

    if args.niah:
        niah = parse_niah_log(log_file)
        if not niah:
            print("No NIAH results found in log file.")
            return
        if args.perf_csv:
            save_niah_perf_csv(niah, args.perf_csv, args.model_name)
        print(f"\nSummary:\n  NIAH context sizes: {len(niah)}")
        return

    results = parse_benchmark_log(log_file)

    if not results:
        print("No benchmark results found in log file.")
        return

    if args.output:
        output_file = args.output
    else:
        output_file = Path(log_file).stem + '_results.csv'

    save_to_csv(results, output_file)

    if args.perf_csv:
        save_perf_csv(results, args.perf_csv, args.model_name)

    print(f"\nSummary:")
    print(f"  Total unique configurations: {len(results)}")
    print(f"  Output file: {output_file}")


if __name__ == '__main__':
    main()
