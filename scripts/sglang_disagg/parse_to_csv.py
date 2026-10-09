#!/usr/bin/env python3
"""
Parse SGLang benchmark log file and save results to CSV.
Extracts: Concurrency, Input tokens, Output tokens, Total Token throughput (tok/s)
For each configuration, takes the MAX Total Token throughput across all iterations.
"""

import re
import csv
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict


def parse_benchmark_log(log_file: str) -> Dict[Tuple[int, int, int], Dict]:
    """Parse benchmark log file and extract results, keeping max throughput per configuration.

    The log is read cell by cell. Each sweep cell starts with a line from benchmark_xPyD.sh:

        RUNNING: prompts <N> isl <ISL> osl <OSL> con <CON>

    and owns the text up to the next such line. A cell's result is the
    "Serving Benchmark Result" block inside its own text, so a cell that printed none is
    recorded as a failed cell instead of disappearing, and a result is never filed under
    another cell's concurrency. (Reading the header before each result block did both: when a
    cell aborted before its result, e.g. a warmup that got "Bad Gateway" from a dead server,
    that cell had no row at all.)

    sglang.bench_serving prints "Successful requests" but no failed count, so a cell's failed
    requests are its prompt count minus its successful ones.
    """
    results = defaultdict(lambda: {'concurrency': None, 'input_tokens': None,
                                    'output_tokens': None, 'max_throughput': 0.0,
                                    'failed': 0, 'no_result': False})

    with open(log_file, 'r') as f:
        content = f.read()

    # Find the start of the first iteration (ignore warmup)
    first_iter_match = re.search(r'RUNNING: the benchserving script for iter: 1', content)
    if not first_iter_match:
        print("Warning: No iteration 1 found. Processing entire file.")
        start_pos = 0
    else:
        start_pos = first_iter_match.start()

    # Process only from first iteration onwards
    content = content[start_pos:]

    # The prompt count between "prompts" and "isl" is optional: logs written before
    # benchmark_xPyD.sh printed $p_con there have two spaces and nothing between them.
    # Both forms must parse, or a rerun over an archived log yields an empty CSV.
    headers = list(re.finditer(
        r'RUNNING: prompts\s+(?:(\d+)\s+)?isl\s+(\d+)\s+osl\s+(\d+)\s+con\s+(\d+)', content))

    for n, header in enumerate(headers):
        end = headers[n + 1].start() if n + 1 < len(headers) else len(content)
        cell = content[header.end():end]
        prompts = int(header.group(1)) if header.group(1) else None
        isl, osl, con = int(header.group(2)), int(header.group(3)), int(header.group(4))

        config_key = (isl, osl, con)
        data = results[config_key]
        # Always use values from the RUNNING line (isl, osl, con)
        data['concurrency'] = con
        data['input_tokens'] = isl
        data['output_tokens'] = osl

        if '============ Serving Benchmark Result ============' not in cell:
            data['no_result'] = True
            continue

        # Extract Total token throughput (tok/s) from the cell's result block
        throughput_match = re.search(r'Total token throughput \(tok/s\):\s+([\d.]+)', cell)
        if not throughput_match:
            data['no_result'] = True
            continue
        throughput = float(throughput_match.group(1))

        successful_match = re.search(r'Successful requests:\s+(\d+)', cell)
        if prompts is not None and successful_match:
            data['failed'] = max(data['failed'], prompts - int(successful_match.group(1)))

        # Keep the maximum throughput
        if throughput > data['max_throughput']:
            data['max_throughput'] = throughput

    return results


def cell_failed(data: Dict) -> bool:
    """A sweep cell failed if it printed no result, lost any request, or measured no throughput."""
    return data['no_result'] or data['failed'] > 0 or data['max_throughput'] <= 0


def save_to_csv(results: Dict[Tuple[int, int, int], Dict], output_file: str):
    """Save results to CSV file with specified columns."""
    if not results:
        print("No results to save.")
        return
    
    # Define column order
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


def _get_run_metadata(pipeline: str = "sglang"):
    """Collect run metadata from environment variables."""
    import os
    xP = os.environ.get('xP', '1')
    yD = os.environ.get('yD', '1')
    dp_mode = os.environ.get('DP_MODE', '0')
    run_mori = os.environ.get('RUN_MORI', '0')
    gpus = os.environ.get('GPUS_PER_NODE', '8')

    # Determine backend tag
    if dp_mode == '1':
        backend = 'mori_dp'
    elif run_mori == '1':
        backend = 'mori_io'
    else:
        backend = 'mooncake'

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
        # The launcher detects the allocation's GPU and forwards it; gfx942 only
        # when run by hand without it, which is what this used to say unconditionally.
        'gpu_architecture': os.environ.get('PERF_GPU_ARCH', 'gfx942'),
    }


PERF_CSV_FIELDNAMES = [
    'model', 'n_gpus', 'nnodes', 'gpus_per_node', 'training_precision',
    'pipeline', 'args', 'tags', 'docker_file', 'base_docker', 'docker_sha',
    'docker_image', 'git_commit', 'machine_name', 'deployment_type', 'launcher',
    'gpu_architecture', 'performance', 'metric', 'relative_change', 'status',
    'build_duration', 'test_duration', 'dataname', 'data_provider_type',
    'data_size', 'data_download_duration', 'build_number',
    'additional_docker_run_options',
]


def save_perf_csv(results: Dict[Tuple[int, int, int], Dict], output_file: str,
                  model_name: str = "", pipeline: str = "sglang"):
    """Save results in madengine perf.csv format."""
    if not results:
        print("No results to save to perf.csv.")
        return

    meta = _get_run_metadata(pipeline)

    fieldnames = PERF_CSV_FIELDNAMES

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
                'status': 'FAILURE' if cell_failed(data) else 'SUCCESS',
            }
            row.update(meta)
            writer.writerow(row)

    print(f"Saved {len(results)} rows to perf.csv: {output_file}")


def agentic_perf_rows(json_path: str):
    """(performance, metric, status) rows for one agentic aggregate JSON.

    The agentic replay (scripts/common/agentic_lib.sh) writes an aggregate JSON and no
    perf.csv, so madengine collected nothing from a run that had measured throughput,
    latency and cache hit rate ("0 perf files, 0 successful, 0 failed").
    Values are read as the JSON states them. Status follows the replay's own validator
    (validate_agentic_result.sh): FAILURE when nothing succeeded, when the error rate is
    above AGENTIC_MAX_ERROR_RATE (default 0.10), or when the run marked itself invalid.
    """
    import json
    import os
    with open(json_path) as f:
        d = json.load(f)
    workload = os.path.basename(os.path.dirname(os.path.abspath(json_path)))
    rm = d.get('request_metrics') or {}
    lat = rm.get('latency') or {}
    tput = rm.get('throughput') or {}
    acct = d.get('request_accounting') or {}
    total = acct.get('records_total') or d.get('num_requests_total') or 0
    ok = d.get('num_requests_successful') or 0
    errors = acct.get('records_error_dropped') or 0
    max_err = float(os.environ.get('AGENTIC_MAX_ERROR_RATE', '0.10'))
    invalid = os.path.exists(os.path.join(os.path.dirname(os.path.abspath(json_path)), 'RUN_INVALID.json'))
    failed = ok == 0 or invalid or (total and errors / total > max_err)
    status = 'FAILURE' if failed else 'SUCCESS'
    tag = f"agentic {workload}, {ok}/{total} requests"

    def p50(name):
        return (lat.get(name) or {}).get('p50')

    rows = [
        ((tput.get('total') or {}).get('tokens_per_second'), f"tok/s total ({tag})"),
        ((tput.get('output') or {}).get('tokens_per_second'), f"tok/s output ({tag})"),
        (None if p50('ttft') is None else p50('ttft') * 1000.0, f"ms TTFT p50 ({tag})"),
        (None if p50('tpot') is None else p50('tpot') * 1000.0, f"ms TPOT p50 ({tag})"),
        (None if p50('e2el') is None else p50('e2el') * 1000.0, f"ms E2E latency p50 ({tag})"),
    ]
    hit = ((d.get('server_metrics') or {}).get('cache') or {}).get('gpu_cache_hit_rate')
    if hit is not None:
        rows.append((hit * 100.0, f"% GPU prefix-cache hit ({tag})"))
    return [(f"{v:.2f}", m, status) for v, m in rows if v is not None]


def save_agentic_perf_csv(json_paths, output_file: str, model_name: str = "",
                          pipeline: str = "sglang"):
    """Write the agentic aggregate JSON(s) as madengine perf.csv rows."""
    meta = _get_run_metadata(pipeline)
    n = 0
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=PERF_CSV_FIELDNAMES)
        writer.writeheader()
        for path in json_paths:
            for performance, metric, status in agentic_perf_rows(path):
                row = {'model': model_name, 'performance': performance,
                       'metric': metric, 'status': status}
                row.update(meta)
                writer.writerow(row)
                n += 1
    print(f"Saved {n} agentic rows to perf.csv: {output_file}")


def main():
    """Main function."""
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Parse SGLang benchmark log file and save results to CSV')
    parser.add_argument('log_file', type=str, help='Path to benchmark log file')
    parser.add_argument('-o', '--output', type=str, help='Output CSV file name (default: <log_file>_results.csv)')
    parser.add_argument('--perf-csv', type=str, help='Also generate madengine perf.csv at this path')
    parser.add_argument('--model-name', type=str, default='', help='Model name for perf.csv')
    parser.add_argument('--agentic-json', nargs='+', metavar='JSON',
                        help='Write agentic aggregate JSON(s) to --perf-csv (log_file is then ignored)')

    args = parser.parse_args()

    if args.agentic_json:
        if not args.perf_csv:
            print("Error: --agentic-json requires --perf-csv")
            sys.exit(1)
        save_agentic_perf_csv(args.agentic_json, args.perf_csv, args.model_name)
        return

    log_file = args.log_file

    # Check if file exists
    if not Path(log_file).exists():
        print(f"Error: Log file not found: {log_file}")
        sys.exit(1)

    print(f"Parsing log file: {log_file}")

    # Parse the log file
    results = parse_benchmark_log(log_file)

    if not results:
        print("No benchmark results found in log file.")
        return

    # Generate output filename
    if args.output:
        output_file = args.output
    else:
        output_file = Path(log_file).stem + '_results.csv'

    # Save to CSV
    save_to_csv(results, output_file)

    # Save madengine perf.csv if requested
    if args.perf_csv:
        save_perf_csv(results, args.perf_csv, args.model_name)

    print(f"\nSummary:")
    print(f"  Total unique configurations: {len(results)}")
    print(f"  Output file: {output_file}")


if __name__ == '__main__':
    main()
