#!/usr/bin/env python3
"""
Parse large EP benchmark logs and generate madengine-compatible perf.csv.

Parses output from:
- MoRI bench_dispatch_combine / test_dispatch_combine_internode (PrettyTable format)
- DeepEP test_internode / test_low_latency (inline GB/s format)

Each benchmark test becomes a row in perf.csv with:
  model = test name (e.g. "mori_internode_dispatch")
  performance = best bandwidth or latency
  metric = "GB/s" or "us"
"""

import re
import csv
import os
from pathlib import Path
from typing import List, Dict


def parse_mori_prettytable(content: str, source: str) -> List[Dict]:
    """Parse MoRI PrettyTable output.

    Format:
    +------ Dispatch (bfloat16) block=... ------+
    | Metrics | RDMA Bandwidth (GB/s) | ... | Latency (us) |
    | Best    | 71.84                 | ... | 1629         |
    | Worst   | 2.25                  | ... | 52106        |
    | Average | 48.54                 | ... | 6910         |
    """
    results = []

    # Match PrettyTable title lines for dispatch/combine phases
    # Formats: "Dispatch Performance (bfloat16)" or "Dispatch (bfloat16) block=X warp=Y rdma=Z"
    table_pattern = re.compile(
        r'\|\s+(Dispatch|Combine)\s+(?:Performance\s+)?\((\w+)\)',
        re.IGNORECASE
    )

    # Match data rows: | Best/Worst/Average | val | val | val | val |
    row_pattern = re.compile(
        r'\|\s+(Best|Worst|Average)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|'
    )

    current_phase = None
    current_dtype = None

    for line in content.split('\n'):
        title_m = table_pattern.search(line)
        if title_m:
            current_phase = title_m.group(1).lower()
            current_dtype = title_m.group(2)
            continue

        row_m = row_pattern.search(line)
        if row_m and current_phase:
            metric_type = row_m.group(1)  # Best/Worst/Average
            rdma_bw = float(row_m.group(2))
            xgmi_bw = float(row_m.group(3))
            ll_bw = float(row_m.group(4))
            latency = float(row_m.group(5))

            if metric_type == 'Best':
                results.append({
                    'test': f'mori_{source}_{current_phase}',
                    'dtype': current_dtype,
                    'rdma_bw': rdma_bw,
                    'xgmi_bw': xgmi_bw,
                    'latency': latency,
                    'metric_type': 'best',
                })

    return results


def parse_deepep_internode(content: str) -> List[Dict]:
    """Parse DeepEP test_internode output.

    Format:
    [tuned] ... X.XX GB/s (RDMA), Y.YY GB/s (NVL) ...
    Best config: X.XX GB/s (RDMA), Y.YY GB/s (NVL) ...
    """
    results = []

    # Match "Best config" or final tuned lines with RDMA/NVL bandwidth
    best_pattern = re.compile(
        r'([\d.]+)\s+GB/s\s+\(RDMA\),\s+([\d.]+)\s+GB/s\s+\(NVL\)'
    )

    # Track which section we're in based on echo markers
    sections = content.split('-----')
    for section in sections:
        # Find the best (last) bandwidth line in each section
        matches = best_pattern.findall(section)
        if not matches:
            continue

        # Determine test type from section header
        section_lower = section.lower()
        if 'internode' in section_lower and 'low' not in section_lower:
            test_type = 'deepep_internode'
        elif 'low_latency' in section_lower or 'low latency' in section_lower:
            test_type = 'deepep_low_latency'
        elif 'intranode' in section_lower:
            test_type = 'deepep_intranode'
        else:
            test_type = 'deepep_unknown'

        # Use the last match as the "best" result
        rdma_bw, nvl_bw = matches[-1]
        results.append({
            'test': test_type,
            'dtype': 'bf16',
            'rdma_bw': float(rdma_bw),
            'xgmi_bw': float(nvl_bw),
            'latency': 0,
            'metric_type': 'best',
        })

    return results


def parse_log_file(log_path: str, source_hint: str = "") -> List[Dict]:
    """Parse a single log file, auto-detecting format."""
    with open(log_path, 'r') as f:
        content = f.read()

    results = []

    # Try MoRI PrettyTable format
    if 'RDMA Bandwidth (GB/s)' in content:
        source = source_hint or Path(log_path).stem
        # Determine if internode or intranode from filename/content
        if 'internode' in log_path.lower() or 'internode' in content.lower():
            src = 'internode'
        else:
            src = 'intranode'
        if '_ll_' in log_path.lower() or 'low.latency' in content.lower()[:500]:
            src += '_ll'
        results.extend(parse_mori_prettytable(content, src))

    # Try DeepEP inline format
    if 'GB/s (RDMA)' in content:
        results.extend(parse_deepep_internode(content))

    return results


def save_perf_csv(results: List[Dict], output_file: str):
    """Save results in madengine perf.csv format."""
    if not results:
        print("No results to save to perf.csv.")
        return

    nnodes = os.environ.get('NNODES', os.environ.get('SLURM_NNODES', '1'))
    gpus = os.environ.get('GPUS_PER_NODE', '8')
    skip_deepep = os.environ.get('SKIP_DEEPEP', '0')

    if skip_deepep == '1':
        backend = 'mori_only'
    else:
        backend = 'deepep+mori'

    meta = {
        'pipeline': 'large_ep',
        'deployment_type': f'ep_bench_{nnodes}N',
        'tags': f'large_ep,{backend}',
        'n_gpus': str(int(nnodes) * int(gpus)),
        'nnodes': nnodes,
        'gpus_per_node': gpus,
        'docker_image': os.environ.get('DOCKER_IMAGE', ''),
        'machine_name': os.environ.get('SLURM_JOB_NODELIST', ''),
        'launcher': 'slurm_multi',
        'gpu_architecture': 'gfx942',
    }

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

        for r in results:
            # Primary metric: RDMA bandwidth
            row = {
                'model': r['test'],
                'performance': f"{r['rdma_bw']:.2f}",
                'metric': f"GB/s RDMA ({r['dtype']})",
                'status': 'SUCCESS',
            }
            row.update(meta)
            writer.writerow(row)

            # Secondary metric: latency (if available)
            if r.get('latency', 0) > 0:
                row_lat = {
                    'model': r['test'],
                    'performance': f"{r['latency']:.0f}",
                    'metric': f"us latency ({r['dtype']})",
                    'status': 'SUCCESS',
                }
                row_lat.update(meta)
                writer.writerow(row_lat)

    print(f"Saved {len(results)} benchmark results to {output_file}")


def main():
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Parse EP benchmark logs and generate perf.csv')
    parser.add_argument('log_dir', type=str, help='Directory containing EP benchmark log files')
    parser.add_argument('-o', '--output', type=str, default='perf.csv', help='Output perf.csv path')

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Error: Log directory not found: {log_dir}")
        sys.exit(1)

    all_results = []

    # Parse individual benchmark log files (skip ep_bench_results.log which
    # is the combined tee output and would duplicate results)
    log_files = sorted(f for f in list(log_dir.glob('*.log')) + list(log_dir.glob('*.txt'))
                       if f.name != 'ep_bench_results.log')
    for log_file in log_files:
        print(f"Parsing: {log_file.name}")
        results = parse_log_file(str(log_file))
        if results:
            print(f"  Found {len(results)} results")
            all_results.extend(results)

    if not all_results:
        print("No benchmark results found in any log files.")
        return

    save_perf_csv(all_results, args.output)

    print(f"\nSummary:")
    print(f"  Total results: {len(all_results)}")
    print(f"  Output: {args.output}")


if __name__ == '__main__':
    main()
