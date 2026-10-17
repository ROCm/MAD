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
    """Parse benchmark log file and extract results, keeping max throughput per configuration.

    The log is read cell by cell. Each cell starts with a [RUNNING] line from
    benchmark_xPyD.sh or benchmark_long_context.sh and owns the text up to the next one, so a cell's result is the
    "Serving Benchmark Result" block inside its own text. A cell that printed none (it hit the
    harness timeout and logged [STALL], or `vllm bench serve` itself failed, e.g. its warmup
    got an error from a dead server) is recorded as a failed cell instead of disappearing, and
    a result is never filed under another cell's concurrency. Reading the header before each
    result block did both of those wrong, and dropped the first cell of every sweep.
    """
    results = defaultdict(lambda: {'concurrency': None, 'input_tokens': None,
                                    'output_tokens': None, 'max_throughput': 0.0,
                                    'failed': 0, 'stalled': False, 'no_result': False})

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

    # Two header forms, one per harness:
    #   benchmark_xPyD.sh:          [RUNNING] prompts <N> isl <ISL> osl <OSL> con <CON> (timeout <T>s)
    #   benchmark_long_context.sh:  [RUNNING] isl=<ISL> osl=<OSL> con=<CON> warmups=<W> prompts=<N> ...
    headers = sorted(
        [(m.start(), m.end(), m.group(1), m.group(2), m.group(3)) for m in re.finditer(
            r'\[RUNNING\]\s+prompts\s+\d+\s+isl\s+(\d+)\s+osl\s+(\d+)\s+con\s+(\d+)', content)]
        + [(m.start(), m.end(), m.group(1), m.group(2), m.group(3)) for m in re.finditer(
            r'\[RUNNING\]\s+isl=(\d+)\s+osl=(\d+)\s+con=(\d+)', content)])

    for n, (_, header_end, isl, osl, con) in enumerate(headers):
        end = headers[n + 1][0] if n + 1 < len(headers) else len(content)
        cell = content[header_end:end]
        isl, osl, con = int(isl), int(osl), int(con)

        config_key = (isl, osl, con)
        data = results[config_key]
        data['concurrency'] = con
        data['input_tokens'] = isl
        data['output_tokens'] = osl

        if re.search(r'\[STALL\]', cell):
            data['stalled'] = True
        if '============ Serving Benchmark Result ============' not in cell:
            data['no_result'] = True
            continue

        # Extract Total token throughput (tok/s) from the cell's result block
        throughput_match = re.search(r'Total token throughput \(tok/s\):\s+([\d.]+)', cell)
        if not throughput_match:
            data['no_result'] = True
            continue
        throughput = float(throughput_match.group(1))
        failed_match = re.search(r'Failed requests:\s+([\d,]+)', cell)
        if failed_match:
            data['failed'] = max(data['failed'], int(failed_match.group(1).replace(',', '')))

        # Keep the maximum throughput
        if throughput > data['max_throughput']:
            data['max_throughput'] = throughput

    return results


def cell_failed(data: Dict) -> bool:
    """A sweep cell failed if it stalled, printed no result, lost any request, or measured no
    throughput."""
    return data['stalled'] or data['no_result'] or data['failed'] > 0 or data['max_throughput'] <= 0


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


def _workload_config_columns():
    """Descriptive columns naming the shape the benchmark ran at.

    These are workload CONFIGURATION, not run metadata: madengine knows where a job
    ran (nodes, GPUs, image, launcher) but not the parallelism the workload chose.
    Path A's run_vllm.py carries the same kind of columns (tp, dtype, bs), so a
    narrow CSV is the right home for them — unlike the topology fields that used to
    be hand-written into deployment_type, which madengine owns.

    Only non-empty values are emitted, so a launcher that does not set them produces
    no stray columns.
    """
    import os
    cols = {}
    # EP_TP_SIZE is the TP degree inside each wideEP DP rank. It says nothing about
    # the TP path, whose degree lives in the model's tp: flags, so leave it empty there.
    tp = os.environ.get('EP_TP_SIZE') if os.environ.get('WIDE_EP') == '1' else None
    pp = os.environ.get('PP_SIZE')
    if tp:
        cols['tp'] = tp
    if pp:
        cols['pp'] = pp
    if os.environ.get('ENABLE_EP') == '1' or os.environ.get('WIDE_EP') == '1':
        cols['ep_backend'] = (
            os.environ.get('ALL2ALL_BACKEND')
            or os.environ.get('VLLM_ALL2ALL_BACKEND')
            or 'enabled'
        )
    xP, yD = os.environ.get('xP'), os.environ.get('yD')
    if xP and yD and yD != '0':
        cols['prefill_decode'] = f'{xP}P{yD}D'
    return cols


def _get_run_metadata(pipeline: str = "vllm"):
    """Collect run metadata from environment variables (LEGACY full-schema path).

    Only used by save_perf_csv(narrow=False), i.e. by model cards that do not declare
    `multiple_results` and whose CSV madengine reads directly with no metadata to
    merge. Cards on the narrow contract get all of this from madengine instead, which
    is authoritative; prefer migrating rather than extending this function.

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


def parse_niah_log(log_file: str) -> Dict[int, Dict]:
    """Parse NIAH benchmark log file and extract retrieval results per context length.

    Scans the summary lines emitted by benchmark_niah.py, one per context length:
      words=  2000  mean=9.7/10  min=9  max=10  (n=3)
      words=  2000  mean=9.5/10  min=9  max=10  (n=2)  [1 timeout/err excluded]
      words=  2000  NO-RESULT (3/3 timed out or errored ...)
    Returns {n_words: {'mean', 'min', 'max', 'n', 'errors', 'no_result'}}. A length whose
    every request errored has a NO-RESULT entry rather than no entry, so it cannot pass by
    being absent; `errors` counts requests that timed out or errored, which the mean excludes.
    """
    results = {}
    with open(log_file, 'r') as f:
        for line in f:
            m = re.search(
                r'words=\s*(\d+)\s+mean=([\d.]+)/10\s+min=(\d+)\s+max=(\d+)\s+\(n=(\d+)\)',
                line
            )
            if m:
                excluded = re.search(r'\[(\d+) timeout/err excluded\]', line)
                results[int(m.group(1))] = {
                    'mean': float(m.group(2)),
                    'min': int(m.group(3)),
                    'max': int(m.group(4)),
                    'n': int(m.group(5)),
                    'errors': int(excluded.group(1)) if excluded else 0,
                    'no_result': False,
                }
                continue
            m = re.search(r'words=\s*(\d+)\s+NO-RESULT\s+\((\d+)/(\d+)', line)
            if m:
                results[int(m.group(1))] = {
                    'mean': 0.0, 'min': 0, 'max': 0, 'n': int(m.group(3)),
                    'errors': int(m.group(2)), 'no_result': True,
                }
    return results


def niah_length_failed(data: Dict) -> bool:
    """A context length failed if any of its requests timed out or errored. A low retrieval
    score is a measurement, not a failure."""
    return data['no_result'] or data['errors'] > 0


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

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for n_words in sorted(results.keys()):
            data = results[n_words]
            row = {
                'model': model_name,
                'performance': f"{data['mean']:.1f}",
                'metric': f"retrieval/10 (niah words={n_words} seeds={data['n']})",
                'status': 'FAILURE' if niah_length_failed(data) else 'SUCCESS',
            }
            row.update(meta)
            writer.writerow(row)

    print(f"Saved {len(results)} NIAH rows to perf.csv: {output_file}")


# The full (legacy) madengine perf.csv schema.
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
                  model_name: str = "", pipeline: str = "vllm", narrow: bool = False):
    """Save throughput results for madengine.

    Two schemas, selected by `narrow`:

    * narrow=True  -- the preferred contract. The workload reports only what it
      measured and madengine merges in the run metadata it already owns, via the
      model card's `multiple_results` declaration. Same contract as the templated
      launchers, so rows from different launchers stay comparable.
    * narrow=False -- legacy, and still the default. Writes the full 29-column
      perf.csv with metadata assembled from the environment by _get_run_metadata().
      Required by the disagg model cards that do NOT declare `multiple_results`:
      madengine reads their CSV directly from a conventional path, with no metadata
      to merge, so a narrow CSV there would lose every descriptive column.

    To migrate a model: declare `multiple_results` on its card and pass --narrow.
    """
    if not results:
        print("No results to save to perf.csv.")
        return

    if narrow:
        config_cols = _workload_config_columns()
        fieldnames = (['model', 'benchmark', 'inp', 'out', 'max_concurrency',
                       'performance', 'metric'] + list(config_cols))
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for (input_tokens, output_tokens, concurrency), data in sorted(
                results.items(), key=lambda x: (x[0][2], x[0][0], x[0][1])
            ):
                row = {
                    'model': model_name,
                    'benchmark': 'throughput_sweep',
                    'inp': data['input_tokens'],
                    'out': data['output_tokens'],
                    'max_concurrency': data['concurrency'],
                    'performance': f"{data['max_throughput']:.2f}",
                    'metric': 'tok/s',
                }
                row.update(config_cols)
                writer.writerow(row)
        print(f"Saved {len(results)} rows (narrow schema) to {output_file}")
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
                          pipeline: str = "vllm"):
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
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Parse vLLM benchmark log file and save results to CSV')
    parser.add_argument('log_file', type=str, help='Path to benchmark log file')
    parser.add_argument('-o', '--output', type=str, help='Output CSV file name (default: <log_file>_results.csv)')
    parser.add_argument('--perf-csv', type=str, help='Also generate madengine perf.csv at this path')
    parser.add_argument('--model-name', type=str, default='', help='Model name for perf.csv')
    parser.add_argument('--niah', action='store_true',
                        help='Parse NIAH retrieval log instead of throughput sweep (requires --perf-csv)')
    parser.add_argument('--narrow', action='store_true',
                        help='Emit a narrow results CSV (model/performance/metric[/status]) for a model card '
                             'declaring multiple_results, letting madengine supply the run metadata. '
                             'Ignored with --niah, which is always narrow.')
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
        save_perf_csv(results, args.perf_csv, args.model_name, narrow=args.narrow)

    print(f"\nSummary:")
    print(f"  Total unique configurations: {len(results)}")
    print(f"  Output file: {output_file}")


if __name__ == '__main__':
    main()
