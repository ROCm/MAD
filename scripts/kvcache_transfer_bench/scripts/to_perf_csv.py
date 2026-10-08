#!/usr/bin/env python3
"""
Convert kvcache_transfer_bench's merged results into madengine's perf.csv
format
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

PERF_CSV_FIELDNAMES = [
    'model', 'n_gpus', 'nnodes', 'gpus_per_node', 'training_precision',
    'pipeline', 'args', 'tags', 'docker_file', 'base_docker', 'docker_sha',
    'docker_image', 'git_commit', 'machine_name', 'deployment_type', 'launcher',
    'gpu_architecture', 'performance', 'metric', 'relative_change', 'status',
    'build_duration', 'test_duration', 'dataname', 'data_provider_type',
    'data_size', 'data_download_duration', 'build_number',
    'additional_docker_run_options',
]


def _bytes_to_label(size_bytes: int) -> str:
    if size_bytes >= 1 << 30:
        return f"{size_bytes / (1 << 30):g}GB"
    if size_bytes >= 1 << 20:
        return f"{size_bytes / (1 << 20):g}MB"
    if size_bytes >= 1 << 10:
        return f"{size_bytes / (1 << 10):g}KB"
    return f"{size_bytes}B"


def _detect_gpu_arch() -> str:
    """gfx target from the environment, else from rocminfo."""
    env = os.environ.get('MAD_SYSTEM_GPU_ARCHITECTURE', '').strip()
    if env:
        return env
    try:
        out = subprocess.run(['rocminfo'], capture_output=True, text=True,
                             timeout=60).stdout
        for line in out.splitlines():
            # CPU agents and "Marketing Name" also match, hence the gfx prefix
            if 'Name:' in line:
                value = line.split(':', 1)[1].strip()
                if value.startswith('gfx'):
                    return value
    except (OSError, subprocess.SubprocessError):
        pass
    return 'unknown'


def _detect_gpus_per_node() -> str:
    """GPU count from the environment, else from the render nodes present."""
    env = os.environ.get('GPUS_PER_NODE', '').strip()
    if env.isdigit() and int(env) > 0:
        return env
    try:
        count = len([d for d in os.listdir('/dev/dri') if d.startswith('renderD')])
        if count:
            return str(count)
    except OSError:
        pass
    return '8'


def _run_metadata(model_name: str) -> dict:
    """Collect run metadata from the environment, probing hardware as needed."""
    gpus_per_node = _detect_gpus_per_node()
    return {
        'model': model_name,
        'n_gpus': str(2 * int(gpus_per_node)),
        'nnodes': '2',
        'gpus_per_node': gpus_per_node,
        'pipeline': 'kvcache_transfer_bench',
        'docker_image': os.environ.get('DOCKER_IMAGE_NAME', ''),
        'machine_name': os.environ.get('SLURM_JOB_NODELIST', ''),
        'launcher': 'slurm_multi',
        'gpu_architecture': _detect_gpu_arch(),
    }


def build_rows(merged: dict, expected_backends: list[str], model_name: str) -> list[dict]:
    meta = _run_metadata(model_name)
    tags = ",".join(["kvcache_transfer_bench"] + expected_backends)

    by_backend: dict[str, list[dict]] = {b: [] for b in expected_backends}
    for r in merged.get("results", []):
        backend = r.get("backend", "")
        if backend in by_backend:
            by_backend[backend].append(r)

    rows = []
    for backend in expected_backends:
        results = sorted(by_backend[backend], key=lambda r: r["transfer_size"])
        if not results:
            row = {'status': 'FAILURE', 'performance': '', 'tags': tags,
                   'metric': f"no results collected (backend={backend})"}
            row.update(meta)
            rows.append(row)
            continue
        for r in results:
            row = {
                'status': 'SUCCESS',
                'performance': f"{r['throughput_gbs']:.4f}",
                'tags': tags,
                'metric': f"GB/s (backend={backend} size={_bytes_to_label(r['transfer_size'])})",
            }
            row.update(meta)
            rows.append(row)
    return rows


def save_perf_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PERF_CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in PERF_CSV_FIELDNAMES})
    print(f"Saved {len(rows)} row(s) to perf.csv: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Directory containing results_merged.json")
    parser.add_argument("--output", required=True, help="Output perf.csv path")
    parser.add_argument("--model-name", default="pyt_kvcache_transfer_bench_all")
    parser.add_argument("--backends", default="nixl,mori,mooncake",
                        help="Comma-separated backends that were requested for this run")
    args = parser.parse_args()

    merged_path = Path(args.input_dir) / "results_merged.json"
    if not merged_path.exists():
        print(f"Error: {merged_path} not found (did merge_results.py run?)", file=sys.stderr)
        sys.exit(1)

    with open(merged_path) as f:
        merged = json.load(f)

    expected_backends = [b.strip().lower() for b in args.backends.split(",") if b.strip()]
    rows = build_rows(merged, expected_backends, args.model_name)
    save_perf_csv(rows, Path(args.output))


if __name__ == "__main__":
    main()
