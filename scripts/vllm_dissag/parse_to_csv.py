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

import os
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

    # The text before result i holds that cell's [RUNNING] header. That is true
    # for i=1 as well: sections[0] is everything from the "iter: 1" marker to
    # the first result, and the first [RUNNING] lives in there. Guarding this
    # with `if i > 1` silently dropped the FIRST requested cell of every run --
    # A run that asked for con=8,16,32 had a CSV reported only 16 and 32, and
    # another lost its real con=1. The 4-results-vs-3-[RUNNING] shape of those
    # logs is not a defect: the extra result is the warmup, which sits before
    # the "iter: 1" marker and is cut by the truncation above.
    for i, section in enumerate(sections[1:], 1):  # Skip first empty section
        # Look for configuration in the text preceding this result.
        prev_section = sections[i-1]

        # vllm format: [RUNNING] prompts <N> isl <ISL> osl <OSL> con <CON>
        # Take the LAST header in prev_section, not the first. A cell that
        # stalls emits its [RUNNING] and then no result block, so prev_section
        # can hold two headers; matching the first one labels this result with
        # the *stalled* cell's concurrency. When a con=1 cell timed out and
        # con=8's throughput was published as con=1 -- a wrong number, not just
        # a missing row. The nearest preceding header is always the right one.
        _running = re.findall(
            r'\[RUNNING\]\s+prompts\s+\d+\s+isl\s+(\d+)\s+osl\s+(\d+)\s+con\s+(\d+)',
            prev_section
        )
        config_match = None
        if _running:
            _isl, _osl, _con = _running[-1]
            config_match = type('Match', (), {
                'group': lambda self, n, _v=(None, _isl, _osl, _con): _v[n]
            })()
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
    """Collect run metadata from environment variables."""
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

    return {
        'pipeline': pipeline,
        'deployment_type': f'disagg_{xP}P{yD}D',
        'tags': f'{pipeline}_disagg,{backend}',
        'n_gpus': str(int(xP) * int(gpus) + int(yD) * int(gpus)),
        'nnodes': str(int(xP) + int(yD)),
        'gpus_per_node': gpus,
        'docker_image': os.environ.get('DOCKER_IMAGE_NAME', ''),
        'machine_name': os.environ.get('SLURM_JOB_NODELIST', ''),
        'launcher': 'slurm_multi',
        'gpu_architecture': 'gfx942',
    }


def parse_niah_log(log_file: str) -> Dict[int, Dict]:
    """Parse NIAH benchmark log file and extract retrieval results per context length.

    Scans for summary lines emitted by benchmark_niah.py, which today carry
    decoys + a verdict between max= and (n=):
      words=  2000  mean=10.0/10  min=10  max=10  decoys=0.0  RETRIEVAL  (n=3)
    Returns {n_words: {'mean': float, 'min': int, 'max': int, 'n': int}}.
    """
    results = {}
    with open(log_file, 'r') as f:
        for line in f:
            # Anything between max= and (n=) is tolerated: the scorer has grown
            # decoys= and a verdict since this was written, and requiring (n=
            # to sit directly after max= made the regex match nothing at all.
            m = re.search(
                r'words=\s*(\d+)\s+mean=([\d.]+)/10\s+min=(\d+)\s+max=(\d+)\b.*?\(n=(\d+)\)',
                line
            )
            if m:
                n_words = int(m.group(1))
                results[n_words] = {
                    'mean': float(m.group(2)),
                    'min': int(m.group(3)),
                    'max': int(m.group(4)),
                    'n': int(m.group(5)),
                }
    return results


def _open_perf_csv(output_file: str, fieldnames):
    """Open perf.csv for writing, appending to an existing file of the same shape.

    BENCH=validate runs NIAH and then the concurrency sweep in one allocation, and
    both write this path. Truncating would leave only whichever ran last (
    lost all six NIAH rows to the sweep that followed it). Append instead, writing
    the header only when creating the file. A file with a different header is
    replaced rather than corrupted.
    """
    exists = os.path.isfile(output_file) and os.path.getsize(output_file) > 0
    if exists:
        with open(output_file, newline='') as f:
            first = f.readline().strip()
        if first != ','.join(fieldnames):
            exists = False
    f = open(output_file, 'a' if exists else 'w', newline='')
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not exists:
        writer.writeheader()
    return f, writer


def save_niah_perf_csv(results: Dict[int, Dict], output_file: str,
                       model_name: str = "", pipeline: str = "vllm"):
    """Save NIAH results in madengine perf.csv format (one row per context length)."""
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

    f, writer = _open_perf_csv(output_file, fieldnames)
    with f:
        for n_words in sorted(results.keys()):
            data = results[n_words]
            row = {
                'model': model_name,
                'performance': f"{data['mean']:.1f}",
                'metric': f"retrieval/10 (niah words={n_words} seeds={data['n']})",
                'status': 'SUCCESS',
            }
            row.update(meta)
            writer.writerow(row)

    print(f"Saved {len(results)} NIAH rows to perf.csv: {output_file}")


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

    f, writer = _open_perf_csv(output_file, fieldnames)
    with f:
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


def main():
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Parse vLLM benchmark log file and save results to CSV')
    parser.add_argument('log_file', type=str, help='Path to benchmark log file')
    parser.add_argument('-o', '--output', type=str, help='Output CSV file name (default: <log_file>_results.csv)')
    parser.add_argument('--perf-csv', type=str, help='Also generate madengine perf.csv at this path')
    parser.add_argument('--model-name', type=str, default='', help='Model name for perf.csv')
    parser.add_argument('--niah', action='store_true',
                        help='Parse NIAH retrieval log instead of throughput sweep (requires --perf-csv)')

    args = parser.parse_args()

    log_file = args.log_file

    if not Path(log_file).exists():
        print(f"Error: Log file not found: {log_file}")
        sys.exit(1)

    print(f"Parsing log file: {log_file}")

    # NIAH mode: parse retrieval scores, write perf.csv only
    if args.niah:
        if not args.perf_csv:
            print("Error: --niah requires --perf-csv")
            sys.exit(1)
        results = parse_niah_log(log_file)
        if not results:
            print("No NIAH results found in log file.")
            return
        save_niah_perf_csv(results, args.perf_csv, args.model_name)
        print(f"\nSummary (NIAH):")
        print(f"  Context lengths parsed: {len(results)}")
        print(f"  perf.csv: {args.perf_csv}")
        return

    # Default: throughput sweep mode
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
