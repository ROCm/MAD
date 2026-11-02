#################################################################################
#
# MIT License
#
# Copyright (c) Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################

import os
import sys
import csv
import glob
import json
import yaml
import psutil
import shutil
import signal
import argparse
import itertools
import subprocess
import time
from typing import List, Dict

# Import atom-specific accuracy runner
from accuracy_runner_atom import run_accuracy_atom

SUPPORTED_LIST_ARGS = ['model', 'tp', 'inp', 'out', 'bs', 'num_prompts', 'max_concurrency']
CSV_HEADER = [
    "hf_pipeline_tag",
    "model",
    "benchmark",
    "tp",
    "inp",
    "out",
    "kv_cache_dtype",
    "num_prompts",
    "max_concurrency",
    "bs",
    "cmd",
    "performance",
    "metric",
    "unit",
]


# ---------------------------------------------------------------------------
# Perf overrides: auto-fill of a sibling perf.yaml into --config.
#
# perf.yaml entries are keyed by a space-separated `model` list and carry only
# `extra_args` and/or `env`. Before expansion, each base config entry is
# matched against perf entries by model-token intersection; the first matching
# perf entry's extra_args and env are merged into the base entry as a
# *gap-fill*: anything already present in the base is left untouched, and only
# keys missing from the base are added from perf. Other base fields
# (benchmark, tp, inp, out, ...) are not touched at all — perf is a perf-
# tuning sheet, not a run-shape sheet.
# ---------------------------------------------------------------------------

def _tokens(model_field):
    """Split a `model:` value into a set of repo tokens.

    The base `config.yaml` and `perf.yaml` both allow space-separated repos
    on a single `model:` line; tokenize on whitespace so we can intersect.
    """
    if model_field is None:
        return set()
    return {t.strip() for t in str(model_field).split() if t.strip()}


def _fill_missing(base, overrides):
    """Return a copy of `base` with keys from `overrides` added only when missing.

    Base wins on key collisions: any key already present in `base` keeps its
    value, regardless of what `overrides` says. Both `extra_args` and `env`
    in this schema are flat scalar dicts, so a one-level operation is
    sufficient. Inputs are not mutated.
    """
    out = dict(base or {})
    for k, v in (overrides or {}).items():
        if k not in out:
            out[k] = v
    return out


PARALLELISM_KEYS = ('tp', 'ep', 'dp', 'pp')


def find_parallelism_override(overrides, cfg):
    """Return the first override entry whose tp/ep/dp/pp filters all match cfg, or None."""
    for ov in overrides:
        if all(str(cfg.get(k)) == str(ov[k]) for k in PARALLELISM_KEYS if k in ov):
            return ov
    return None


def deep_merge_evaluations(base_evals, model_evals):
    """Index-based merge: model evals override into corresponding base evals."""
    merged = []
    for i, model_eval in enumerate(model_evals):
        if i < len(base_evals):
            m = dict(base_evals[i])
            m.update(model_eval)
            merged.append(m)
        else:
            merged.append(model_eval)
    return merged


def load_perf_entries(perf_path):
    """Load perf.yaml into a normalized list of override entries.

    Returns an empty list when the file is missing or empty; each entry is
    `{'models': set[str], 'extra_args': dict, 'env': dict}`.
    """
    if not perf_path or not os.path.exists(perf_path):
        return []
    with open(perf_path, 'r') as f:
        raw = yaml.safe_load(f) or []
    entries = []
    for e in raw:
        entries.append({
            'models': _tokens(e.get('model')),
            'extra_args': e.get('extra_args', {}) or {},
            'bench_args': e.get('bench_args', {}) or {},
            'env': e.get('env', {}) or {},
            'bench_serving': e.get('bench_serving', {}) or {},
        })
    return entries


def load_acc_entries(acc_path):
    """Load acc.yaml with task-level and model-level entries.

    Returns (task_defaults, model_entries) where:
    - task_defaults: dict[hf_pipeline_tag -> entry dict with 'evaluations']
    - model_entries: list of model-specific entries with 'models' set and 'raw' dict

    Task-level entries have 'hf_pipeline_tag' but no 'model' field.
    Model-level entries have 'model' field.
    """
    if not acc_path or not os.path.exists(acc_path):
        return {}, []
    with open(acc_path, 'r') as f:
        raw = yaml.safe_load(f) or []

    task_defaults = {}
    model_entries = []

    for e in raw:
        if 'model' not in e and 'hf_pipeline_tag' in e:
            # Task-level default entry
            task_defaults[e['hf_pipeline_tag']] = e
        elif 'model' in e:
            # Model-level entry
            model_entries.append({
                'models': _tokens(e.get('model')),
                'extra_args': e.get('extra_args', {}) or {},
                'bench_args': e.get('bench_args', {}) or {},
                'env': e.get('env', {}) or {},
                'raw': e  # Keep full entry for evaluations and other fields
            })

    return task_defaults, model_entries


def apply_perf_overrides(configs, perf_entries):
    """Mutate `configs` in place, gap-filling matching perf overrides.

    For each base entry, the first perf entry whose model tokens intersect
    the base entry's model tokens wins. Only keys missing from the base
    entry's `extra_args` / `env` are added from perf — anything already set
    in the base is preserved. Returns a per-entry summary list suitable for
    logging, distinguishing keys that were added vs. skipped (already
    present in base).
    """
    summary = []
    for cfg in configs:
        base_tokens = _tokens(cfg.get('model'))
        matched = next((p for p in perf_entries if p['models'] & base_tokens), None)
        if not matched:
            continue

        base_extra = cfg.get('extra_args') or {}
        base_bench = cfg.get('bench_args') or {}
        base_env   = cfg.get('env') or {}
        cfg['extra_args'] = _fill_missing(base_extra, matched['extra_args'])
        cfg['bench_args'] = _fill_missing(base_bench, matched.get('bench_args') or {})
        cfg['env']        = _fill_missing(base_env,   matched['env'])

        # gap-fill per-model bench_serving knobs from perf.yaml (base wins)
        if matched.get('bench_serving'):
            cfg['bench_serving'] = _fill_missing(cfg.get('bench_serving') or {}, matched['bench_serving'])

        summary.append({
            'model': cfg.get('model'),
            'matched_perf_models': sorted(matched['models']),
            'extra_args_added':   sorted(k for k in matched['extra_args'] if k not in base_extra),
            'extra_args_skipped': sorted(k for k in matched['extra_args'] if k in base_extra),
            'bench_args_added':   sorted(k for k in (matched.get('bench_args') or {}) if k not in base_bench),
            'bench_args_skipped': sorted(k for k in (matched.get('bench_args') or {}) if k in base_bench),
            'env_added':          sorted(k for k in matched['env'] if k not in base_env),
            'env_skipped':        sorted(k for k in matched['env'] if k in base_env),
        })
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description='Run ATOM benchmark')
    parser.add_argument('--config',
                            type=str,
                            help='config yaml file',
                            required=True,
                        )
    parser.add_argument('--model',
                            type=str,
                            help='select model from config',
                            required=False,
                            default=None,
                        )
    parser.add_argument('--benchmark',
                            type=str,
                            help='select benchmark from config',
                            required=False,
                            default=None,
                        )
    parser.add_argument('--tp',
                            type=str,
                            help='select tensor parallel size from config',
                            required=False,
                            default=None,
                        )
    parser.add_argument('--inp',
                            type=str,
                            help='select input size from config',
                            required=False,
                            default=None,
                        )
    parser.add_argument('--out',
                            type=str,
                            help='select output size from config',
                            required=False,
                            default=None,
                        )
    parser.add_argument('--bs',
                            type=str,
                            help='select batch size from config',
                            required=False,
                            default=None,
                        )
    parser.add_argument('--num_prompts',
                            type=str,
                            help='select num prompts from config',
                            required=False,
                            default=None,
                        )
    parser.add_argument('--max_concurrency',
                            type=str,
                            help='select max concurrency from config',
                            required=False,
                            default=None,
                        )
    parser.add_argument('--perf',
                            type=str,
                            help='Path to perf.yaml (default: <dir of --config>/perf.yaml)',
                            required=False,
                            default=None,
                        )
    parser.add_argument('--perf-output',
                            type=str,
                            help='Where to write the merged config '
                                 '(default: <dir of --config>/perf_config.yaml). '
                                 'The original --config file is never modified.',
                            required=False,
                            default=None,
                        )
    parser.add_argument('--no-perf-merge',
                            action='store_true',
                            help='Disable auto-merge of sibling perf.yaml into --config',
                        )
    parser.add_argument('--acc',
                            type=str,
                            help='Path to acc.yaml (default: <dir of --config>/acc.yaml)',
                            required=False,
                            default=None,
                        )
    parser.add_argument('--acc-output',
                            type=str,
                            help='Where to write the merged accuracy config '
                                 '(default: <dir of --config>/acc_config.yaml). '
                                 'The original --config file is never modified.',
                            required=False,
                            default=None,
                        )
    parser.add_argument('--profile',
                            action='store_true',
                            help='Kernel-profiling run: serve under rtl trace + roctx markers and aggregate a kernel payload',
                        )
    args = parser.parse_args()
    return args

def dict_to_args(args_dict: Dict) -> str:
    """Convert argument dictionary to command-line string"""
    args_list = []
    for key, value in args_dict.items():
        if value is True:
            args_list.append(key)
        elif value is not False and value is not None:
            args_list.append(f"{key} {value}")
    return " ".join(args_list)

def dict_to_env(env_dict: Dict) -> str:
    """Convert environment dictionary to env var string"""
    return " ".join(f"{k}={v}" for k, v in env_dict.items())

def expand_configs(args, configs: List[Dict]):
    # Apply architecture specific overrides to config
    cfgs = []
    arch = os.environ.get('MAD_SYSTEM_GPU_ARCHITECTURE', 'unknown')
    for config in configs:
        cfg = config.copy()
        # pop all architecture specific overrides from config
        arch_overrides = cfg.pop('arch_overrides', {})
        if arch_override := arch_overrides.get(arch, {}):
            print(f"Detected {arch} architecture, applying override {arch_override} to config {cfg}")
            cfg.update(arch_override)
        cfgs.append(cfg)

    # Expand combinations from SUPPORTED_LIST_ARGS
    print(f"Expanding configs for the following keys: {SUPPORTED_LIST_ARGS} into individual configs")
    config_list = []
    for cfg in cfgs:
        # split config into common args and list args
        common_cfgs = {k: v for k, v in cfg.items() if k not in SUPPORTED_LIST_ARGS}
        list_cfgs = {k: str(v).split(' ') for k, v in cfg.items() if k in SUPPORTED_LIST_ARGS}
        # expand list args into one dict per combination
        expanded_cfgs = [dict(zip(list_cfgs.keys(), x)) for x in itertools.product(*list_cfgs.values())]
        for expanded_cfg in expanded_cfgs:
            config_list.append({**common_cfgs, **expanded_cfg})

    # filter config list according to command line args if specified
    filtered_configs = config_list
    for arg_name in SUPPORTED_LIST_ARGS:
        if arg_val := getattr(args, arg_name):
            print(f"Filtering configs by {arg_name}={arg_val}")
            filtered_configs = [cfg for cfg in filtered_configs if cfg.get(arg_name, None) == arg_val]

    # Benchmark field logic:
    # - If config lacks 'benchmark' field: use CLI args.benchmark (basic.yaml case)
    # - If config has 'benchmark: all' and CLI specifies one: normalize to CLI benchmark
    #   (treat "all" as "supports both, run whichever is requested")
    # - Otherwise: keep config's benchmark field
    for cfg in filtered_configs:
        if "benchmark" not in cfg:
            cfg["benchmark"] = args.benchmark
        elif cfg["benchmark"] == "all" and args.benchmark and args.benchmark != "all":
            # Normalize "all" to the requested benchmark
            cfg["benchmark"] = args.benchmark

    # filter configs by benchmark if CLI specifies one
    if args.benchmark and args.benchmark != "all":
        print(f"Filtering configs by benchmark={args.benchmark}")
        filtered_configs = [cfg for cfg in filtered_configs if cfg["benchmark"] == args.benchmark]

    return filtered_configs

def adjust_glm_batched_tokens(config):
    """Adjust max-num-batched-tokens for GLM-5.x models with large ISL.

    GLM-5.x models with MLA (Multi-head Latent Attention) require special handling
    for long context. This function calculates the appropriate max-num-batched-tokens
    and attn-prefill-chunk-size based on the input sequence length.

    Calculation: next power of 2 after the largest power of 2 less than inp.
    - For inp=100000: 2^17 = 131072 (128K)
    - For inp=50000: 2^16 = 65536 (64K)
    - For inp=10000: 2^14 = 16384 (16K, default)

    Only applies to models with 'GLM-5' in the model name.
    """
    if 'GLM-5' not in config.get('model', ''):
        return

    inp = config.get('inp')
    if not inp:
        return

    inp = int(inp)
    default_max_batched = 16384

    import math
    # Calculate next power of 2 after largest power of 2 less than inp
    calculated = 2 ** (int(math.log2(inp)) + 1)
    max_batched = max(calculated, default_max_batched)

    # Only adjust if different from default
    if max_batched != default_max_batched:
        if 'extra_args' not in config:
            config['extra_args'] = {}

        print(f"INFO: GLM-5 model with inp={inp}, setting max-num-batched-tokens={max_batched}")
        config['extra_args']['--max-num-batched-tokens'] = max_batched
        # Also set MLA-specific chunk size for GLM models
        config['extra_args']['--attn-prefill-chunk-size'] = max_batched

def _wait_for_server(server, port=8000, timeout=5400, poll_interval=30):
    """Wait until the server's /v1/models endpoint responds.

    Returns True once the endpoint is reachable. Returns False if the server
    process exits before becoming ready (e.g. an OOM during initialization) or
    if the timeout elapses -- instead of blocking on a dead endpoint for the
    full timeout, which previously made a crashed server look like a hang.
    """
    deadline = time.time() + timeout
    url = f"http://localhost:{port}/v1/models"
    while time.time() < deadline:
        if server.poll() is not None:
            print(
                f"Server process exited with code {server.returncode} "
                "before becoming ready",
                flush=True,
            )
            return False
        probe = subprocess.run(
            f"curl -s {url}",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if probe.returncode == 0:
            return True
        time.sleep(poll_interval)
    print("Timed out waiting for server to start", flush=True)
    return False

def run_serving(model, config):
    # num_prompts = multiplier * max_concurrency; multiplier from perf.yaml bench_serving (default 10), popped so it stays out of the CSV row
    multiplier = int((config.pop("bench_serving", {}) or {}).get("num_prompts_multiplier") or 10)
    if not config.get("num_prompts"):
        config["num_prompts"] = str(multiplier * int(config["max_concurrency"]))
    output_json = (
        f"{config['model']}_serving_{config['tp']}_{config['inp']}_{config['out']}_{config['num_prompts']}_{config['max_concurrency']}.json"
    )
    server_cmd = (
        "python -m atom.entrypoints.openai_server "
        f"--model {model} "
        f"-tp {config['tp']} "
    )

    # Add kv_cache_dtype if specified
    if 'kv_cache_dtype' in config and config['kv_cache_dtype']:
        server_cmd += f"--kv_cache_dtype {config['kv_cache_dtype']} "

    # Get env and extra args from config (keep as dicts)
    env_dict = config.pop('env', {})
    extra_args_dict = config.pop('extra_args', {})

    # Convert dicts to command line strings
    env_str = dict_to_env(env_dict) if env_dict else ""
    extra_args_str = dict_to_args(extra_args_dict) if extra_args_dict else ""

    server_cmd = f"{env_str} {server_cmd} {extra_args_str}".strip()
    config["cmd"] = server_cmd

    # start server
    print(server_cmd)
    server = subprocess.Popen(server_cmd, shell=True)

    try:
        # wait for server to start; fail fast if the process dies (e.g. OOM
        # during init) instead of blocking on a dead endpoint for 30 minutes
        if not _wait_for_server(server):
            print("Server failed to start")
            return [config]
        else:
            print(f"Server at {server.pid} contacted successfully", flush=True)

        # run benchmark
        bench_cmd = (
            "python -m atom.benchmarks.benchmark_serving "
            f"--model {model} "
            f"--backend vllm "
            f"--base-url http://localhost:8000 "
            f"--percentile-metrics ttft,tpot,itl,e2el "
            f"--dataset-name random "
            f"--ignore-eos "
            f"--request-rate inf "
            f"--max-concurrency {config['max_concurrency']} "
            f"--num-prompts {config['num_prompts']} "
            f"--random-input-len {config['inp']} "
            f"--random-output-len {config['out']} "
            f"--save-result "
            f"--result-dir ./ "
            f"--result-filename {output_json}"
        )

        # Add --trust-remote-code if it was in extra_args (from recipe)
        if '--trust-remote-code' in extra_args_dict:
            bench_cmd += " --trust-remote-code"
        bench_args = config.pop('bench_args', {})
        bench_args_str = ""
        for k, v in bench_args.items():
            if isinstance(v, bool):
                bench_args_str += f"{k} "
            else:
                bench_args_str += f"{k} {v} "
        bench_cmd = f"{bench_cmd} {bench_args_str}".strip()

        config["cmd"] = f"{server_cmd};{bench_cmd}"
        print(bench_cmd)
        subprocess.run(bench_cmd, shell=True, check=True)

        # parse output json
        results = []
        with open(output_json, "r", newline="", encoding="utf-8") as f:
            output = json.load(f)
            if "total_token_throughput" in output:
                metrics = {
                    "throughput_tot": str(output["total_token_throughput"]),
                    "throughput_gen": str(output["output_throughput"]),
                    "median_ttft": str(output["median_ttft_ms"]),
                    "median_tpot": str(output["median_tpot_ms"]),
                    "median_itl": str(output["median_itl_ms"]),
                    "median_e2el": str(output["median_e2el_ms"]),
                }
                for metric, perf in metrics.items():
                    if "throughput" in metric:
                        unit = "tok/sec"
                    else:
                        unit = "ms"
                    result = {
                        "performance": perf,
                        "metric": metric,
                        "unit": unit,
                        **config
                    }
                    results.append(result)

        return results

    finally:
        # kill server and children
        parent = psutil.Process(server.pid)
        for child in parent.children(recursive=True):
            child.send_signal(signal.SIGINT)
        server.send_signal(signal.SIGINT)
        _ = server.communicate()
        del server

def run_serving_profile(model, config):
    """Single-pass profiling for the ATOM serving benchmark (Component 5, kernel breakdown).

    Serves `atom.entrypoints.openai_server` under rocm-trace-lite (`rtl trace --mode full`) with roctx
    prefill/decode/mixed phase markers (scripts/atom/profiling/), producing rtl_trace.db (GPU kernels +
    per-pass roctx markers), then aggregates it into ./profile/kernel_summary_payload.json for the
    kernel-summary aggregator.
    """
    config.pop("bench_serving", None)  # perf.yaml-only knob; keep out of CSV row
    if not config.get("num_prompts"):
        config["num_prompts"] = str(min(5, 10 * int(config["max_concurrency"])))
    output_json = (
        f"{config['model']}_serving_profile_{config['tp']}_{config['inp']}_{config['out']}_{config['num_prompts']}_{config['max_concurrency']}.json"
    )
    profile_dir = config.get("profile_dir", f"profile_{config['model']}_{config['tp']}")
    os.makedirs(profile_dir, exist_ok=True)
    rtl_trace_path = os.path.join(profile_dir, "rtl_trace.db")

    server_cmd = (
        "python -m atom.entrypoints.openai_server "
        f"--model {model} "
        f"-tp {config['tp']} "
    )
    if 'kv_cache_dtype' in config and config['kv_cache_dtype']:
        server_cmd += f"--kv_cache_dtype {config['kv_cache_dtype']} "

    env_dict = config.pop('env', {})
    extra_args_dict = config.pop('extra_args', {})
    env_str = dict_to_env(env_dict) if env_dict else ""
    extra_args_str = dict_to_args(extra_args_dict) if extra_args_dict else ""

    # roctx phase markers: PYTHONPATH puts sitecustomize.py on the path so the patch reaches ATOM's
    # spawned engine-core/worker processes (spawn start method); ATOM_ROCTX_PHASE_MARKERS arms it.
    profiling_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiling")
    roctx_env = f"PYTHONPATH={profiling_dir}${{PYTHONPATH:+:$PYTHONPATH}} ATOM_ROCTX_PHASE_MARKERS=1"
    rtl_env = f"{roctx_env} rtl trace --mode full -o {rtl_trace_path}"
    server_cmd = f"{env_str} {rtl_env} {server_cmd} {extra_args_str}".strip()
    config["cmd"] = server_cmd

    print("=" * 60)
    print("  PROFILE: rocm-trace-lite (--mode full) + roctx phase markers")
    print("=" * 60)
    print(server_cmd)
    # Redirect the server's stdout+stderr to a file rather than inheriting the console: ATOM's aiter
    # backend prints one line per shape during CUDA-graph capture, which otherwise bloats the CI log
    # past the Actions blob-store limit (making it an unreadable BlobNotFound). On failure we print the
    # tail so the real traceback is retrievable; the full log is kept under profile_dir (--keep-model-dir).
    server_log_path = os.path.join(profile_dir, "server_profile.log")
    _server_log = open(server_log_path, "w")

    def _tail_server_log(n=200):
        try:
            _server_log.flush()
            with open(server_log_path, "r", errors="replace") as _f:
                _tail = _f.readlines()[-n:]
            print(f"----- last {len(_tail)} lines of {server_log_path} (server stdout+stderr) -----")
            print("".join(_tail))
            print("----- end server log tail -----", flush=True)
        except Exception as _e:
            print(f"(could not read server log {server_log_path}: {_e})")

    server = subprocess.Popen(server_cmd, shell=True, stdout=_server_log, stderr=subprocess.STDOUT)
    try:
        status = subprocess.run(
            "timeout 5400 bash -c 'until curl -s http://localhost:8000/v1/models; do sleep 30; done' || exit 1",
            shell=True,
        )
        if status.returncode != 0:
            print("Server failed to start for profiling run")
            _tail_server_log()
            return [config]
        print(f"Server at {server.pid} ready for profiling", flush=True)

        bench_cmd = (
            "python -m atom.benchmarks.benchmark_serving "
            f"--model {model} "
            f"--backend vllm "
            f"--base-url http://localhost:8000 "
            f"--percentile-metrics ttft,tpot,itl,e2el "
            f"--dataset-name random "
            f"--ignore-eos "
            f"--request-rate inf "
            f"--max-concurrency {config['max_concurrency']} "
            f"--num-prompts {config['num_prompts']} "
            f"--random-input-len {config['inp']} "
            f"--random-output-len {config['out']} "
            f"--save-result "
            f"--result-dir ./ "
            f"--result-filename {output_json}"
        )
        if '--trust-remote-code' in extra_args_dict:
            bench_cmd += " --trust-remote-code"
        print(bench_cmd)
        try:
            subprocess.run(bench_cmd, shell=True, check=True)
        except subprocess.CalledProcessError:
            print("Benchmark failed for profiling run")
            _tail_server_log()
            raise
    finally:
        # SIGINT (not SIGKILL) so rtl's signal handler finalizes rtl_trace.db before exit.
        try:
            parent = psutil.Process(server.pid)
            for child in parent.children(recursive=True):
                child.send_signal(signal.SIGINT)
            server.send_signal(signal.SIGINT)
            server.communicate(timeout=180)
        except Exception:
            try:
                server.send_signal(signal.SIGKILL)
                server.communicate()
            except Exception:
                pass
        del server

    if not os.path.exists(rtl_trace_path):
        print(f"WARNING: RTL trace not found at {rtl_trace_path}")
        return [config]
    print(f"RTL trace: {rtl_trace_path}")

    os.makedirs("profile", exist_ok=True)
    payload_path = os.path.join("profile", "kernel_summary_payload.json")
    # Shared, framework-agnostic curated aggregator. The MAD repo is present at runtime
    # (madengine MODEL_DIR), so reference it by repo-relative path.
    scripts_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    aggregator = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiling", "kernel_summary_payload.py")
    try:
        subprocess.run(
            ["python3", aggregator, rtl_trace_path, "-o", payload_path, "--full",
             "--tp", str(config["tp"]), "--precision", str(config.get("dtype", "auto")),
             "--isl", str(config["inp"]), "--osl", str(config["out"])],
            check=True,
        )
        print(f"Payload: {payload_path}")
    except Exception as e:
        print(f"WARNING: payload generation failed: {e}")

    result = {"performance": "N/A", "metric": "profile", "unit": "trace", **config}
    result["cmd"] = server_cmd
    return [result]

def run_accuracy(model, config):
    """Run accuracy evaluation using atom-specific accuracy framework.

    Delegates to accuracy_runner_atom which handles atom-specific server
    management and evaluation logic.
    """
    def _kill_atom_server(server):
        """Kill atom server and children."""
        parent = psutil.Process(server.pid)
        for child in parent.children(recursive=True):
            child.send_signal(signal.SIGINT)
        server.send_signal(signal.SIGINT)
        _ = server.communicate()
        del server

    return run_accuracy_atom(model, config, _wait_for_server, _kill_atom_server)

def main():
    args = parse_args()

    # Load base config
    with open(args.config, 'r') as f:
        print(f"Loading configs from {args.config}")
        configs = yaml.safe_load(f) or []

    # Perf merge: auto-detect a sibling perf.yaml, deep-merge matching
    # extra_args + env into each base entry in memory, and materialize the
    # merged result to a sibling perf_config.yaml. The original --config file
    # is never modified; the in-memory merged configs are what actually run.
    if not args.no_perf_merge:
        config_dir = os.path.dirname(os.path.abspath(args.config))
        perf_path = args.perf or os.path.join(config_dir, 'perf.yaml')
        perf_output = args.perf_output or os.path.join(config_dir, 'perf_config.yaml')

        if os.path.abspath(perf_output) == os.path.abspath(args.config):
            raise ValueError(
                f"--perf-output must differ from --config to keep the source untouched; "
                f"both resolved to {perf_output}"
            )

        perf_entries = load_perf_entries(perf_path)
        if perf_entries:
            summary = apply_perf_overrides(configs, perf_entries)
            if summary:
                print(f"Applied perf overrides from {perf_path} (base wins; perf fills gaps):")
                for s in summary:
                    print(
                        f"  model={s['model']} <- perf {s['matched_perf_models']}\n"
                        f"    extra_args added:   {s['extra_args_added']}\n"
                        f"    extra_args skipped: {s['extra_args_skipped']} (already set in base)\n"
                        f"    env added:          {s['env_added']}\n"
                        f"    env skipped:        {s['env_skipped']} (already set in base)"
                    )
                with open(perf_output, 'w') as f:
                    yaml.safe_dump(configs, f, sort_keys=False, default_flow_style=False)
                print(
                    f"Wrote merged config to {perf_output} "
                    f"(original {args.config} left untouched)"
                )
            else:
                print(
                    f"No perf entries matched any model in {args.config}; "
                    f"not writing {perf_output}"
                )
        else:
            print(f"No perf.yaml found at {perf_path} (or empty); skipping perf merge")

    # Accuracy merge: two-level merge from acc.yaml (task defaults + model overrides).
    # For atom-accuracy tag, always merge (tag implies accuracy, benchmark field irrelevant)
    # Only skip for benchmark=="all" (which does its own inline merge)
    if args.benchmark != "all":
        config_dir = os.path.dirname(os.path.abspath(args.config))
        acc_path = args.acc or os.path.join(config_dir, 'acc.yaml')
        acc_output = args.acc_output or os.path.join(config_dir, 'acc_config.yaml')

        if os.path.abspath(acc_output) == os.path.abspath(args.config):
            raise ValueError(
                f"--acc-output must differ from --config to keep the source untouched; "
                f"both resolved to {acc_output}"
            )

        task_defaults, model_entries = load_acc_entries(acc_path)
        if task_defaults or model_entries:
            # Apply two-level merge to each config
            for cfg in configs:
                pipeline_tag = cfg.get('hf_pipeline_tag', 'text-generation')

                # Level 1: Apply task-level defaults based on hf_pipeline_tag
                if pipeline_tag in task_defaults:
                    task_entry = task_defaults[pipeline_tag]
                    if 'evaluations' not in cfg and 'evaluations' in task_entry:
                        cfg['evaluations'] = task_entry['evaluations']
                        print(f"  Applied task-level defaults for {cfg.get('model')}: hf_pipeline_tag={pipeline_tag}")

                # Level 2: Apply model-specific overrides
                base_tokens = _tokens(cfg.get('model'))
                matched = next((p for p in model_entries if p['models'] & base_tokens), None)

                if matched:
                    base_extra = cfg.get('extra_args') or {}
                    base_bench = cfg.get('bench_args') or {}
                    base_env = cfg.get('env') or {}

                    cfg['extra_args'] = _fill_missing(base_extra, matched['extra_args'])
                    cfg['bench_args'] = _fill_missing(base_bench, matched['bench_args'])
                    cfg['env'] = _fill_missing(base_env, matched['env'])

                    if 'evaluations' in matched['raw']:
                        cfg['evaluations'] = deep_merge_evaluations(
                            cfg.get('evaluations', []),
                            matched['raw']['evaluations']
                        )

                    # Level 3: Apply parallelism-specific overrides (tp/ep/dp/pp)
                    parallelism_overrides = matched['raw'].get('overrides', [])
                    if parallelism_overrides:
                        ov = find_parallelism_override(parallelism_overrides, cfg)
                        if ov:
                            if 'evaluations' in ov:
                                cfg['evaluations'] = deep_merge_evaluations(
                                    cfg.get('evaluations', []),
                                    ov['evaluations']
                                )
                            ov_extra = ov.get('extra_args', {})
                            if ov_extra:
                                cfg['extra_args'] = _fill_missing(cfg.get('extra_args') or {}, ov_extra)
                            ov_env = ov.get('env', {})
                            if ov_env:
                                cfg['env'] = _fill_missing(cfg.get('env') or {}, ov_env)
                            print(
                                f"  Applied parallelism override "
                                f"({', '.join(f'{k}={ov[k]}' for k in PARALLELISM_KEYS if k in ov)})"
                            )

                    for k, v in matched['raw'].items():
                        if k not in ['model', 'extra_args', 'bench_args', 'env', 'evaluations', 'overrides']:
                            if k not in cfg:
                                cfg[k] = v

                    print(f"  Applied model-specific overrides for {cfg.get('model')}")

            with open(acc_output, 'w') as f:
                yaml.safe_dump(configs, f, sort_keys=False, default_flow_style=False)
            print(
                f"Wrote merged accuracy config to {acc_output} "
                f"(original {args.config} left untouched)"
            )
        else:
            print(f"No acc.yaml found at {acc_path} (or empty); skipping accuracy merge")

    # Expand and filter configs
    configs = expand_configs(args, configs)
    print(f"Running configs: ", *configs, sep='\n')

    # Iterate over configs
    for config in configs:
        model = config['model']
        # Use model name for logging
        config['model'] = os.path.basename(model)

        # Write header to csv
        OUTPUT_CSV = "perf_" + os.path.basename(model) + ".csv"
        header_write = 0 if os.path.exists(OUTPUT_CSV) else 1
        with open(OUTPUT_CSV, "a+", newline="") as outf:
            writer = csv.DictWriter(outf, delimiter=",", fieldnames=CSV_HEADER, extrasaction="ignore")
            if header_write:
                writer.writeheader()
                outf.flush()

            # Use huggingface token if present
            if MAD_SECRETS_HFTOKEN := os.environ.get('MAD_SECRETS_HFTOKEN'):
                os.environ['HF_TOKEN'] = MAD_SECRETS_HFTOKEN
            else:
                print("Warning: MAD_SECRETS_HFTOKEN is not set. If a gated model is used, please set MAD_SECRETS_HFTOKEN=<your-huggingface-token>")
            # Use CHECK_LOCAL_DATA env var to control whether to check for local data
            # By default, this is set to False and can be enabled through madengine additional_context
            CHECK_LOCAL_DATA = os.environ.get('CHECK_LOCAL_DATA', 'false').lower() == 'true'

            # Use dataprovider if present for model weights
            if CHECK_LOCAL_DATA and (MAD_DATAHOME := os.environ.get('MAD_DATAHOME')) and os.path.exists(os.path.join(MAD_DATAHOME, model)):
                model = os.path.join(MAD_DATAHOME, model)
                print("Found MAD_DATAHOME updating model path.")
            elif (MAD_DATAHOME := os.environ.get('MAD_DATAHOME')) and os.path.isfile(os.path.join(MAD_DATAHOME, 'config.json')):
                # MAD_DATAHOME points directly at a model checkpoint (e.g. a mounted /model_weights)
                model = MAD_DATAHOME
                print(f"Found model checkpoint at MAD_DATAHOME={MAD_DATAHOME}")
            elif config.get('extra_args', {}).get('--load_dummy', False):
                print("Found --load_dummy in config, using dummy weights for benchmarking")
            else:
                # Explicitly download model before running benchmarks for easier debugging.
                # NOTE: pass each exclude pattern with its own --exclude flag and avoid
                # shell=True; otherwise the hf CLI treats the extra patterns as positional
                # filenames and silently downloads 0 files.
                download_command = [
                    "hf", "download", model,
                    "--exclude", "original/*",
                    "--exclude", "*.tf",
                    "--exclude", "*.onnx",
                    "--exclude", "*.flax",
                    "--exclude", "*.rust",
                ]
                # Don't trust the CLI's exit code: the typer/click versions in this venv
                # leak a clean Exit(0) up as an unhandled exception, so `hf` can return
                # exit status 1 even after a fully successful download. Verify the
                # download via huggingface_hub's cache lookup helper, which resolves the
                # snapshot path through the hub's own cache layout/ref handling rather
                # than us hard-coding refs/main + models--<repo>/snapshots/.
                #
                # We also capture hf's stdout/stderr instead of inheriting them, so the
                # spurious typer/click traceback (plus the tqdm progress bars on stderr)
                # don't land in the log on the happy path. On real failure we re-emit
                # both streams for diagnosis before raising.
                from huggingface_hub import try_to_load_from_cache

                result = subprocess.run(
                    download_command, check=False, capture_output=True, text=True
                )
                cached_config = try_to_load_from_cache(
                    repo_id=model, filename="config.json"
                )
                # try_to_load_from_cache returns either a str path, None (not cached),
                # or a sentinel marking "known not to exist". Collapse all non-path
                # outcomes (and stale entries that no longer exist on disk) into one
                # failure branch.
                if not (
                    isinstance(cached_config, str) and os.path.exists(cached_config)
                ):
                    sys.stdout.write(result.stdout or "")
                    sys.stderr.write(result.stderr or "")
                    raise RuntimeError(
                        f"hf download {model} failed (exit={result.returncode}); "
                        f"config.json for {model} not found in the HF cache"
                    )
                snapshot_path = os.path.dirname(cached_config)
                # On the happy path, keep hf's stdout (the "✓ Downloaded" line and the
                # cache path) but drop stderr, which on this venv contains the spurious
                # typer/click traceback and the tqdm progress bars.
                sys.stdout.write(result.stdout or "")
                if result.returncode != 0:
                    print(
                        f"hf exited with code {result.returncode} but {snapshot_path} "
                        f"looks complete; treating as success (known typer/click Exit "
                        f"propagation issue in this venv)."
                    )
                print(f"Downloaded {model} to {snapshot_path}")

            # Adjust max-num-batched-tokens for GLM-5.x models if needed
            adjust_glm_batched_tokens(config)

            # Keep dict forms for potential acc.yaml merge (for benchmark: all)
            env_vars = config.get("env", {})
            extra_args = config.get("extra_args", {})

            config["env_dict"] = env_vars  # Preserve dict for acc.yaml merge
            config["extra_args_dict"] = extra_args  # Preserve dict for acc.yaml merge

            # run benchmark
            results = []
            benchmark = config["benchmark"]
            profile = bool(config.pop('profile', False)) or getattr(args, 'profile', False)
            if benchmark == "all":
                # Run serving first, then accuracy (similar to vllm)
                print(f"\n=== Running serving benchmark for {model} ===")
                serving_config = dict(config)
                serving_config["benchmark"] = "serving"
                serving_results = run_serving_profile(model, serving_config) if profile else run_serving(model, serving_config)

                # Write serving results
                for result in serving_results:
                    writer.writerow(result)

                # Run accuracy with acc.yaml merge
                print(f"\n=== Running accuracy benchmark for {model} ===")
                acc_config = dict(config)
                acc_config["benchmark"] = "accuracy"

                # Load and merge acc.yaml for accuracy run
                config_dir = os.path.dirname(os.path.abspath(args.config))
                acc_path = args.acc or os.path.join(config_dir, 'acc.yaml')

                if os.path.exists(acc_path):
                    task_defaults, model_entries = load_acc_entries(acc_path)
                    if task_defaults or model_entries:
                        full_model = model
                        pipeline_tag = acc_config.get('hf_pipeline_tag', 'text-generation')

                        # Level 1: Apply task-level defaults
                        if pipeline_tag in task_defaults:
                            task_entry = task_defaults[pipeline_tag]
                            if 'evaluations' not in acc_config and 'evaluations' in task_entry:
                                acc_config['evaluations'] = task_entry['evaluations']
                                print(f"  Applied task-level defaults for hf_pipeline_tag={pipeline_tag}")

                        # Level 2: Apply model-specific overrides
                        base_tokens = _tokens(full_model)
                        matched = next((p for p in model_entries if p['models'] & base_tokens), None)
                        if matched:
                            base_extra = acc_config.get('extra_args_dict', {})
                            base_env = acc_config.get('env_dict', {})
                            acc_config['extra_args_dict'] = _fill_missing(base_extra, matched['extra_args'])
                            acc_config['env_dict'] = _fill_missing(base_env, matched['env'])
                            if 'evaluations' in matched['raw']:
                                acc_config['evaluations'] = deep_merge_evaluations(
                                    acc_config.get('evaluations', []),
                                    matched['raw']['evaluations']
                                )

                            # Apply parallelism-specific overrides
                            parallelism_overrides = matched['raw'].get('overrides', [])
                            if parallelism_overrides:
                                ov = find_parallelism_override(parallelism_overrides, acc_config)
                                if ov:
                                    if 'evaluations' in ov:
                                        acc_config['evaluations'] = deep_merge_evaluations(
                                            acc_config.get('evaluations', []),
                                            ov['evaluations']
                                        )
                                    ov_extra = ov.get('extra_args', {})
                                    if ov_extra:
                                        acc_config['extra_args_dict'] = _fill_missing(
                                            acc_config.get('extra_args_dict', {}), ov_extra
                                        )
                                    ov_env = ov.get('env', {})
                                    if ov_env:
                                        acc_config['env_dict'] = _fill_missing(
                                            acc_config.get('env_dict', {}), ov_env
                                        )
                                    print(
                                        f"  Applied parallelism override "
                                        f"({', '.join(f'{k}={ov[k]}' for k in PARALLELISM_KEYS if k in ov)})"
                                    )

                            print(f"  Applied model-specific overrides from acc.yaml")

                    # Copy merged dicts back to non-suffixed keys for run_accuracy to use
                    acc_config['extra_args'] = acc_config.get('extra_args_dict', {})
                    acc_config['env'] = acc_config.get('env_dict', {})

                results = run_accuracy(model, acc_config)
            elif benchmark == "serving":
                results = run_serving_profile(model, config) if profile else run_serving(model, config)
            elif benchmark == "accuracy":
                results = run_accuracy(model, config)
            else:
                raise ValueError(f"Unknown benchmark: {benchmark}")

            # Write results to csv
            for result in results:
                writer.writerow(result)


if __name__ == "__main__":
    main()
