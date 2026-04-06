"""
Shared utilities for KV cache benchmark backends.

Provides common functions used across mooncake, rixl, and mori backends
to avoid code duplication.
"""

import json
import socket as sock
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def resolve_hostname(hostname: str) -> str:
    """Resolve hostname to IP address. Returns hostname unchanged on failure."""
    try:
        return sock.gethostbyname(hostname)
    except sock.gaierror:
        return hostname


def generate_test_sizes(start_size: int, end_size: int) -> List[int]:
    """Generate sizes from start_size to end_size (inclusive), doubling each time."""
    if start_size > end_size:
        return [start_size]  # single size if range reversed
    sizes = []
    size = start_size
    while size <= end_size:
        sizes.append(size)
        if size == end_size:
            break
        size *= 2
    return sizes


def append_result_to_file(results_path: Path, metadata: Dict[str, Any], result_entry: Dict[str, Any]) -> None:
    """Append a single result entry to the results JSON file.
    Creates the file with metadata on first call; appends on subsequent calls.
    """
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
        data["results"].append(result_entry)
    else:
        data = {"metadata": metadata, "results": [result_entry]}
    with open(results_path, "w") as f:
        json.dump(data, f, indent=2)


def collect_version_info(
    backend_name: str,
    version_tries: List[Callable[[], Optional[str]]],
    pytorch_version: Optional[str] = None,
    rocm_version: Optional[str] = None,
) -> Dict[str, str]:
    """Collect version info for benchmark metadata.
    version_tries: list of callables that return version string (e.g. from importlib.metadata).
    """
    backend_version = "unknown"
    for _try in version_tries:
        try:
            v = _try()
            if v:
                backend_version = v
                break
        except Exception:
            pass

    if pytorch_version is None:
        try:
            import torch
            pytorch_version = torch.__version__
        except Exception:
            pytorch_version = "unknown"

    if rocm_version is None or rocm_version == "unknown":
        try:
            import torch
            rocm_version = getattr(torch.version, "hip", None) or "unknown"
        except Exception:
            rocm_version = "unknown"
        if rocm_version == "unknown":
            for _path in ("/opt/rocm/.info/version", "/opt/rocm/version"):
                try:
                    rocm_version = open(_path).read().strip()
                    break
                except Exception:
                    pass

    return {
        "version_info": {
            backend_name: backend_version,
            "pytorch": pytorch_version,
            "rocm": rocm_version,
        }
    }


def add_common_bench_args(parser, default_shared: str = "shared", include_append: bool = False) -> None:
    """Add common benchmark arguments to an argparse parser."""
    parser.add_argument("--start_size", type=int, required=True, help="Starting buffer size in bytes")
    parser.add_argument("--end_size", type=int, required=True, help="Ending buffer size in bytes")
    parser.add_argument("--target_node", type=str, default=None,
                        help="Target node hostname (default: TARGET_IP env or current hostname)")
    parser.add_argument("--initiator_node", type=str, default=None,
                        help="Initiator node hostname (informational only)")
    parser.add_argument("--shared_folder", type=str, default=default_shared,
                        help="Shared folder for coordination")
    parser.add_argument(
        "--sync-port",
        type=int,
        default=9999,
        help="TCP port for inter-node benchmark sync (default: 9999)",
    )
    if include_append:
        parser.add_argument("--append_results", action="store_true",
                            help="Append results to existing file instead of overwriting")
