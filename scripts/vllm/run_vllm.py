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
import time
import yaml
import psutil
import shutil
import signal
import argparse
import itertools
import subprocess
import shlex
from typing import List, Dict

# Import vllm-specific accuracy runner
from accuracy_runner_vllm import run_accuracy_vllm

SUPPORTED_LIST_ARGS = ['model', 'tp', 'inp', 'out', 'bs', 'num_prompts', 'max_concurrency']
CSV_HEADER = [
    "hf_pipeline_tag",
    "model",
    "benchmark",
    "tp",
    "inp",
    "out",
    "dtype",
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


def _format_cli_arg(k, v):
    """Serialize one (key, value) pair to a CLI fragment.

    vLLM mixes two argument styles on the same command line:

    - Standard argparse (`--foo`, `--foo-bar`): bool True is a bare flag,
      bool False is omitted entirely, and non-bool values are passed as a
      separate token (`--foo value`).
    - Compilation-config / DSL shortcuts (`-cc.x.y`, `-O.level`, `-tp`-style
      single-dash keys, etc.): the parser splits on `=`, so booleans **must**
      be passed as `key=value` (e.g. `-cc.foo=True`). Passing `-cc.foo`
      bare causes pydantic to try to consume the *next* argv token as the
      value, which produces opaque bool-parse errors when the next token
      happens to be another flag.

    Heuristic: double-dash keys take the argparse form; everything else
    (single-dash, dotted, etc.) takes the `key=value` form.
    """
    is_argparse_style = k.startswith('--')
    if isinstance(v, bool):
        if is_argparse_style:
            return f" {k}" if v else ""
        return f" {k}={v}"
    # Shell-quote non-bool values: these command strings are executed with
    # shell=True, so values containing spaces / JSON / quotes (e.g.
    # --compilation-config '{"cudagraph_num_of_warmups": 3}') must be quoted
    # to survive shell tokenization.
    v_str = shlex.quote(str(v))
    if is_argparse_style:
        return f" {k} {v_str}"
    return f" {k}={v_str}"


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
    `{'models': set[str], 'extra_args': dict, 'bench_args': dict, 'env': dict}`.
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
        })
    return entries


def load_acc_entries(acc_path):
    """Load acc.yaml with task-level and model-level entries.

    Returns (task_defaults, model_entries) tuple:
    - task_defaults: dict keyed by hf_pipeline_tag
    - model_entries: list of model-specific overrides
    """
    if not acc_path or not os.path.exists(acc_path):
        return {}, []

    with open(acc_path, 'r') as f:
        raw = yaml.safe_load(f) or []

    task_defaults = {}
    model_entries = []

    for e in raw:
        if 'model' not in e and 'hf_pipeline_tag' in e:
            # Task-level default (no model field)
            task_defaults[e['hf_pipeline_tag']] = e
        elif 'model' in e:
            # Model-level entry
            model_entries.append({
                'models': _tokens(e.get('model')),
                'extra_args': e.get('extra_args', {}) or {},
                'bench_args': e.get('bench_args', {}) or {},
                'env': e.get('env', {}) or {},
                'raw': e,  # Keep raw entry for evaluations and other fields
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
        cfg['bench_args'] = _fill_missing(base_bench, matched['bench_args'])
        cfg['env']        = _fill_missing(base_env,   matched['env'])

        summary.append({
            'model': cfg.get('model'),
            'matched_perf_models': sorted(matched['models']),
            'extra_args_added':   sorted(k for k in matched['extra_args'] if k not in base_extra),
            'extra_args_skipped': sorted(k for k in matched['extra_args'] if k in base_extra),
            'bench_args_added':   sorted(k for k in matched['bench_args'] if k not in base_bench),
            'bench_args_skipped': sorted(k for k in matched['bench_args'] if k in base_bench),
            'env_added':          sorted(k for k in matched['env'] if k not in base_env),
            'env_skipped':        sorted(k for k in matched['env'] if k in base_env),
        })
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description='Run VLLM benchmark')
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
                            help='Profile workload',
                        )
    args = parser.parse_args()
    return args

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

def run_latency(model, config):
    output_json = (
        f"{config['model']}_latency_{config['tp']}_{config['inp']}_{config['out']}_{config['bs']}.json"
    )
    cmd = (
        "vllm bench latency "
        f"--model {model} "
        f"--dtype {config['dtype']} "
        f"-tp {config['tp']} "
        f"--input-len {config['inp']} "
        f"--output-len {config['out']} "
        f"--batch-size {config['bs']} "
        f"--num-iters-warmup 3 --num-iters 5 "
        f"--no-enable-prefix-caching "
        f"--trust-remote-code "
        f"--output-json {output_json} "
    )
    # pop env and extra args from config
    env = config.pop('env', "")
    extra_args = config.pop('extra_args', "")
    cmd = f"{env} {cmd} {extra_args}".strip()
    config["cmd"] = cmd
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)

    # Parse output json
    results = []
    with open(output_json, "r", newline="", encoding="utf-8") as f:
        output = json.load(f)
        if "avg_latency" in output: 
            result = {
                "performance": output["avg_latency"],
                "metric": "latency",
                "unit": "ms",
                **config
            }
            results.append(result)

    return results

def run_throughput(model, config):
    output_json = (
        f"{config['model']}_throughput_{config['tp']}_{config['inp']}_{config['out']}_{config['num_prompts']}.json"
    )
    cmd = (
        "vllm bench throughput "
        f"--model {model} "
        f"--dtype {config['dtype']} "
        f"-tp {config['tp']} "
        f"--input-len {config['inp']} "
        f"--output-len {config['out']} "
        f"--num-prompts {config['num_prompts']} "
        f"--no-enable-prefix-caching "
        f"--trust-remote-code "
        f"--output-json {output_json} "
    )
    # pop env and extra args from config
    env = config.pop('env', "")
    extra_args = config.pop('extra_args', "")
    cmd = f"{env} {cmd} {extra_args}".strip()
    config["cmd"] = cmd
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)

    # Parse output json
    results = []
    with open(output_json, "r", newline="", encoding="utf-8") as f:
        output = json.load(f)
        if "tokens_per_second" in output: 
            elapsed_time = output["elapsed_time"]
            throughput_gen = str(
                int(int(config["num_prompts"]) * int(config["out"]) / elapsed_time)
            )
            metrics = {
                "throughput_tot": str(output["tokens_per_second"]),
                "throughput_gen": throughput_gen,
            }
            for metric, perf in metrics.items():
                result = {
                    "performance": perf,
                    "metric": metric,
                    "unit": "tok/sec",
                    **config
                }
                results.append(result)

    return results


# ---------------------------------------------------------------------------
# Serving helpers (shared by run_serving and run_serving_profile)
# ---------------------------------------------------------------------------

def _kill_server(server):
    """Kill a vllm serve subprocess and all its children."""
    try:
        parent = psutil.Process(server.pid)
        for child in parent.children(recursive=True):
            child.send_signal(signal.SIGINT)
        server.send_signal(signal.SIGINT)
        _ = server.communicate()
    except (psutil.NoSuchProcess, ProcessLookupError):
        pass
    del server


def _wait_for_server(server, timeout=1800, poll_interval=15):
    """Wait for the vllm server to be ready. Returns True on success.

    Polls http://localhost:8000/v1/models AND the serve subprocess: if `server`
    exits/dies before becoming ready, returns False immediately instead of blocking
    for the full timeout. A stalled model load (e.g. an HF-Hub download stall) would
    otherwise burn the entire timeout (up to hours) and leave the run with no payload
    (the `payload_not_found` failure mode on large-model profiling runs).
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if server.poll() is not None:
            print(f"Server process exited (code {server.returncode}) before becoming "
                  f"ready — aborting wait", flush=True)
            return False
        try:
            code = subprocess.run(
                "curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/v1/models",
                shell=True, capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            if code == "200":
                return True
        except subprocess.TimeoutExpired:
            pass
        time.sleep(poll_interval)
    print(f"Server did not become ready within {timeout}s", flush=True)
    return False


def _build_server_cmd(model, config, env="", extra_args="", extra_flags=""):
    """Build a vllm serve command string."""
    cmd = (
        "vllm serve "
        f"{model} "
        f"--dtype {config['dtype']} "
        f"-tp {config['tp']} "
        f"--no-enable-prefix-caching "
        f"--trust-remote-code "
        f"--disable-uvicorn-access-log "
    )
    if extra_flags:
        cmd += f"{extra_flags} "
    return f"{env} {cmd} {extra_args}".strip()


def _build_bench_serve_cmd(model, config, bench_args=None, extra_flags=""):
    """Build a vllm bench serve command string."""
    cmd = (
        "vllm bench serve "
        f"--model {model} "
        f"--percentile-metrics ttft,tpot,itl,e2el "
        f"--dataset-name random "
        f"--ignore-eos "
        f"--temperature 0 "
        f"--trust-remote-code "
        f"--max-concurrency {config['max_concurrency']} "
        f"--num-prompts {config['num_prompts']} "
        f"--random-input-len {config['inp']} "
        f"--random-output-len {config['out']} "
        f"--request-rate inf "
    )
    if extra_flags:
        cmd += f"{extra_flags} "
    if bench_args:
        for k, v in bench_args.items():
            if isinstance(v, bool):
                cmd += f"{k} "
            else:
                cmd += f"{k} {v} "
    return cmd.strip()


def run_serving(model, config):
    # by default use num_prompts = 10 * max_concurrency if not specified
    if not config.get("num_prompts"):
        config["num_prompts"] = str(10 * int(config["max_concurrency"]))
    output_json = (
        f"{config['model']}_serving_{config['tp']}_{config['inp']}_{config['out']}_{config['num_prompts']}_{config['max_concurrency']}.json"
    )
    # pop env and extra args from config
    env = config.pop('env', "")
    extra_args = config.pop('extra_args', "")
    server_cmd = _build_server_cmd(model, config, env, extra_args)
    config["cmd"] = server_cmd

    # start server
    print(server_cmd)
    server = subprocess.Popen(server_cmd, shell=True)

    try:
        # wait for server to start; timeout after 30 minutes
        status = subprocess.run(
            "timeout 5400 bash -c 'until curl -s http://localhost:8000/v1/models; do sleep 30; done' || exit 1",
            shell=True
        )
        if status.returncode != 0:
            print("Server failed to start")
            return [config]
        else:
            print(f"Server at {server.pid} contacted successfully", flush=True)

        # run benchmark
        bench_args = config.pop('bench_args', {})
        bench_cmd = _build_bench_serve_cmd(
            model, config, bench_args,
            extra_flags=f"--save-result --result-filename {output_json}",
        )
        
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
        _kill_server(server)


def run_serving_profile(model, config):
    """Single-pass profiling for the serving benchmark.

    Runs `vllm serve` under rocm-trace-lite (`rtl trace --mode full`) with roctx prefill/decode/mixed
    phase markers (scripts/vllm/profiling/), producing rtl_trace.db (GPU kernels + per-pass markers),
    then aggregates it into ./profile/kernel_summary_payload.json for the client-perf-hub
    uploader. The earlier torch.compile / PyTorch-profiler pass is gone: the per-pass
    prefill/decode/mixed annotations now live directly in the rtl trace as roctx markers.
    """
    if not config.get("num_prompts"):
        config["num_prompts"] = str(min(5, 10 * int(config["max_concurrency"])))

    profile_dir = config.get("profile_dir", f"profile_{config['model']}_{config['tp']}")
    os.makedirs(profile_dir, exist_ok=True)

    env = config.pop('env', "")
    extra_args = config.pop('extra_args', "")
    bench_args = config.pop('bench_args', {})

    print("=" * 60)
    print("  PROFILE: rocm-trace-lite (--mode full) + roctx phase markers")
    print("=" * 60)

    rtl_trace_path = os.path.join(profile_dir, "rtl_trace.db")
    # roctx phase markers: PYTHONPATH puts sitecustomize.py on the path so the patch reaches vLLM's
    # spawned workers; VLLM_ROCTX_PHASE_MARKERS arms it. Both are inherited by the `vllm serve` that
    # `rtl trace` launches. See scripts/vllm/profiling/.
    profiling_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiling")
    roctx_env = f"PYTHONPATH={profiling_dir}${{PYTHONPATH:+:$PYTHONPATH}} VLLM_ROCTX_PHASE_MARKERS=1"
    # Capture-stability (perf-preserving): during CUDA-graph capture a slow Triton JIT on one TP rank can
    # stall the capture collective long enough for the NCCL watchdog (ProcessGroupNCCL::HeartbeatMonitor)
    # to kill a worker — observed on the CI runner as "Worker proc VllmWorker-0 died unexpectedly" ~mid
    # capture. Raise the heartbeat timeout (default 480s) so a slow compile can't trip it. Env-only, zero
    # runtime cost, and keeps every captured graph (unlike enforce_eager / trimming capture sizes).
    watchdog_env = "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800 TORCH_NCCL_ENABLE_MONITORING=1"
    rtl_env = f"{watchdog_env} {roctx_env} {env} rtl trace --mode full -o {rtl_trace_path}"
    # Compile warm-up: extra eager warm-up passes before capture push shape-specialized Triton compiles
    # *out of* the lockstep capture window (the actual trigger of the stall above). Keeps all capture
    # sizes → no perf loss, only a slightly longer startup. Skipped if the config already sets its own
    # --compilation-config (e.g. a PIECEWISE override) to avoid a duplicate flag.
    warmup_flags = (
        "" if "--compilation-config" in (extra_args or "")
        else "--compilation-config '{\"cudagraph_num_of_warmups\": 3}'"
    )
    server_cmd = _build_server_cmd(model, config, rtl_env, extra_args, extra_flags=warmup_flags)

    print(server_cmd)
    server = subprocess.Popen(server_cmd, shell=True)
    try:
        if not _wait_for_server(server, timeout=5400):
            print("Server failed to start for profiling run")
            return [config]
        print(f"Server at {server.pid} ready for profiling", flush=True)

        bench_cmd = _build_bench_serve_cmd(model, config, bench_args)
        print(bench_cmd)
        subprocess.run(bench_cmd, shell=True, check=True)
    finally:
        _kill_server(server)

    if not os.path.exists(rtl_trace_path):
        print(f"WARNING: RTL trace not found at {rtl_trace_path}")
        return [config]
    print(f"RTL trace: {rtl_trace_path}")

    # Aggregate the rtl trace into the client-perf-hub payload (Kernels[] + Config). Written under
    # ./profile/ so it matches the uploader's `kernel_summary_payload*.json` glob and survives
    # --keep-model-dir.
    os.makedirs("profile", exist_ok=True)
    payload_path = os.path.join("profile", "kernel_summary_payload.json")
    aggregator = os.path.join(profiling_dir, "kernel_summary_payload.py")
    try:
        subprocess.run(
            ["python3", aggregator, rtl_trace_path, "-o", payload_path, "--full",
             "--tp", str(config["tp"]), "--precision", str(config.get("dtype", "auto")),
             "--isl", str(config["inp"]), "--osl", str(config["out"])],
            check=True)
        print(f"Payload: {payload_path}")
    except Exception as e:
        print(f"WARNING: payload generation failed: {e}")

    result = {"performance": "N/A", "metric": "profile", "unit": "trace", **config}
    result["cmd"] = server_cmd
    return [result]


def run_accuracy(model, config):
    """Run accuracy evaluation using vllm-specific accuracy framework.

    Delegates to accuracy_runner_vllm which handles vllm-specific server
    management and evaluation logic.
    """
    return run_accuracy_vllm(model, config, _kill_server)

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

    # NOTE: acc.yaml merge is now deferred to benchmark execution time.
    # For benchmark: serving → only perf.yaml is applied (already done above)
    # For benchmark: accuracy or all → acc.yaml will be applied during execution

    # Expand and filter configs
    configs = expand_configs(args, configs)
    print(f"Running configs: ", *configs, sep='\n')

    # Track which models have had accuracy run (to avoid duplicates when benchmark: all)
    accuracy_completed = set()

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

            # Preserve original repo ID for acc.yaml token matching (before local path substitution)
            original_repo_id = model

            # Use dataprovider if present for model weights
            if CHECK_LOCAL_DATA and (MAD_DATAHOME := os.environ.get('MAD_DATAHOME')) and os.path.exists(os.path.join(MAD_DATAHOME, model)):
                model = os.path.join(MAD_DATAHOME, model)
                print("Found MAD_DATAHOME")
            elif (MAD_DATAHOME := os.environ.get('MAD_DATAHOME')) and os.path.isfile(os.path.join(MAD_DATAHOME, 'config.json')):
                # MAD_DATAHOME points directly at a model checkpoint (e.g. a mounted /model_weights)
                model = MAD_DATAHOME
                print(f"Found model checkpoint at MAD_DATAHOME={MAD_DATAHOME}")
            elif config.get('extra_args', {}).get('--load-format', None) == 'dummy':
                print("Found --load-format dummy in config, using dummy weights for benchmarking")
            else:
                # Explicitly pre-download the model so `vllm serve` loads from a COMPLETE
                # local cache. Otherwise a transient HF-Hub stall during model load hangs
                # the server (the root cause of intermittent large-model profiling failures).
                # NOTE: the old `hf download {model} --exclude <globs>` CLI form silently
                # fetched 0 files on newer hf CLIs (the globs were parsed as explicit
                # filenames), so nothing was actually pre-staged. Use the Python API with
                # bounded retries instead (respects HF_HOME / HF_TOKEN / HF_HUB_ENABLE_HF_TRANSFER).
                print("Explicitly pre-downloading model from HF Hub")
                _ignore = ["original/*", "*.tf", "*.onnx", "*.flax", "*.rust"]
                # Enable the Rust-accelerated HF downloader (hf_transfer) to avoid the multi-hour
                # stalls observed on very large (1T) checkpoints with the default Python downloader.
                # Best-effort: install it if missing (upstream images ship without it) and only set
                # the env flag once it is available, so the download never breaks when it isn't.
                try:
                    import hf_transfer  # noqa: F401
                    _have_hft = True
                except Exception:
                    _r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "hf_transfer"],
                                        check=False, timeout=300)
                    _have_hft = (_r.returncode == 0)
                if _have_hft:
                    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
                    print("hf_transfer enabled (accelerated HF download)")
                else:
                    print("hf_transfer unavailable; using default HF downloader")
                # Run snapshot_download in a subprocess with a hard wall-clock timeout. A silently
                # stalled HF connection makes the in-process call hang indefinitely (observed: a ~12h CI
                # hang stuck at 71/185 files) — a plain try/except retry can NOT catch that because no
                # exception is ever raised. subprocess.run(timeout=...) SIGKILLs the stalled child, and
                # snapshot_download resumes from the partial cache on the next attempt, so progress is
                # monotonic across retries. Per-attempt timeout is tunable via PREDOWNLOAD_TIMEOUT_SEC.
                _dl_script = (
                    "from huggingface_hub import snapshot_download; "
                    f"snapshot_download({model!r}, ignore_patterns={_ignore!r}, max_workers=16)"
                )
                _dl_timeout = int(os.environ.get("PREDOWNLOAD_TIMEOUT_SEC", "3600"))
                _dl_ok = False
                for _attempt in range(1, 6):
                    try:
                        subprocess.run([sys.executable, "-c", _dl_script],
                                       check=True, timeout=_dl_timeout)
                        _dl_ok = True
                        break
                    except subprocess.TimeoutExpired:
                        print(f"HF pre-download attempt {_attempt}/5 stalled >{_dl_timeout}s — killed; "
                              f"retrying (resumes from partial cache)...", flush=True)
                    except subprocess.CalledProcessError as e:
                        print(f"HF pre-download attempt {_attempt}/5 failed (exit {e.returncode}); "
                              f"retrying...", flush=True)
                        time.sleep(min(30 * _attempt, 120))
                if not _dl_ok:
                    raise RuntimeError(
                        f"Model pre-download did not complete after 5 attempts "
                        f"({_dl_timeout}s wall-clock each)")
            
            # concatenate env vars and extra args into the corresponding strings
            # Keep dict forms for potential acc.yaml merge (for benchmark: all)
            env_vars = config.get("env", {})
            extra_args = config.get("extra_args", {})
            bench_args = config.get("bench_args", {})

            config["env_dict"] = env_vars  # Preserve dict for acc.yaml merge
            config["extra_args_dict"] = extra_args  # Preserve dict for acc.yaml merge
            config["bench_args_dict"] = bench_args  # Preserve dict for acc.yaml merge

            env_vars_str = " ".join(f"{k}={v}" for k, v in env_vars.items())
            extra_args_str = ""
            for k, v in extra_args.items():
                extra_args_str += _format_cli_arg(k, v)
            config["env"] = env_vars_str
            config["extra_args"] = extra_args_str
            
            # run benchmark
            results = []
            benchmark = config["benchmark"]

            # Handle "all" benchmark: run serving first, then accuracy with acc.yaml merge
            if benchmark == "all":
                # Run serving benchmark with current config (already has perf.yaml merged)
                print(f"\n=== Running serving benchmark for {model} ===")
                serving_config = dict(config)
                serving_config["benchmark"] = "serving"
                if not args.profile:
                    serving_results = run_serving(model, serving_config)
                else:
                    serving_results = run_serving_profile(model, serving_config)

                # Write serving results
                for result in serving_results:
                    writer.writerow(result)

                # Run accuracy only once per model (skip if already done)
                if model not in accuracy_completed:
                    # Now merge with acc.yaml and run accuracy
                    print(f"\n=== Running accuracy benchmark for {model} ===")
                    # Load acc.yaml and apply overrides using two-level merge
                    config_dir = os.path.dirname(os.path.abspath(args.config))
                    acc_path = args.acc or os.path.join(config_dir, 'acc.yaml')

                    if os.path.exists(acc_path):
                        task_defaults, model_entries = load_acc_entries(acc_path)

                        if task_defaults or model_entries:
                            # Create a copy of config for accuracy
                            acc_config = dict(config)
                            # Use original repo ID for token matching (not local path)
                            full_model = original_repo_id
                            pipeline_tag = acc_config.get('hf_pipeline_tag', 'text-generation')

                            # Level 1: Apply task-level defaults based on hf_pipeline_tag
                            if pipeline_tag in task_defaults:
                                task_entry = task_defaults[pipeline_tag]
                                if 'evaluations' not in acc_config and 'evaluations' in task_entry:
                                    acc_config['evaluations'] = task_entry['evaluations']
                                    print(f"  Applied task-level defaults for hf_pipeline_tag={pipeline_tag}")

                            # Level 2: Apply model-specific overrides
                            base_tokens = _tokens(full_model)
                            matched = next((p for p in model_entries if p['models'] & base_tokens), None)

                            if matched:
                                # Apply model overrides
                                base_extra = acc_config.get('extra_args_dict', {})
                                base_bench = acc_config.get('bench_args_dict', {})
                                base_env = acc_config.get('env_dict', {})

                                acc_config['extra_args_dict'] = _fill_missing(base_extra, matched['extra_args'])
                                acc_config['bench_args_dict'] = _fill_missing(base_bench, matched['bench_args'])
                                acc_config['env_dict'] = _fill_missing(base_env, matched['env'])

                                if 'evaluations' in matched['raw']:
                                    acc_config['evaluations'] = deep_merge_evaluations(
                                        acc_config.get('evaluations', []),
                                        matched['raw']['evaluations']
                                    )

                                # Level 3: Apply parallelism-specific overrides
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

                                for k, v in matched['raw'].items():
                                    if k not in ['model', 'extra_args', 'bench_args', 'env', 'evaluations', 'overrides']:
                                        if k not in acc_config:
                                            acc_config[k] = v

                                # Merge bench_args into config for accuracy runner
                                for k, v in acc_config['bench_args_dict'].items():
                                    if k not in acc_config:
                                        acc_config[k] = v

                                # Rebuild env and extra_args strings
                                env_vars_str = " ".join(f"{k}={v}" for k, v in acc_config['env_dict'].items())
                                extra_args_str = ""
                                for k, v in acc_config['extra_args_dict'].items():
                                    extra_args_str += _format_cli_arg(k, v)
                                acc_config["env"] = env_vars_str
                                acc_config["extra_args"] = extra_args_str

                                print(f"  Applied model-specific overrides for {full_model}")
                            else:
                                print(f"  No model-specific entry, using task-level defaults only")

                            acc_config["benchmark"] = "accuracy"
                            accuracy_results = run_accuracy(model, acc_config)

                            # Write accuracy results
                            for result in accuracy_results:
                                writer.writerow(result)

                            # Mark this model as having accuracy completed
                            accuracy_completed.add(model)
                        else:
                            print(f"  Warning: No acc.yaml entries found, skipping accuracy")
                    else:
                        print(f"  Warning: acc.yaml not found at {acc_path}, skipping accuracy")
                else:
                    print(f"  Skipping accuracy for {model} (already completed)")

            elif benchmark == "latency":
                results = run_latency(model, config)
                for result in results:
                    writer.writerow(result)
            elif benchmark == "throughput":
                results = run_throughput(model, config)
                for result in results:
                    writer.writerow(result)
            elif benchmark == "serving":
                if not args.profile:
                    results = run_serving(model, config)
                else:
                    results = run_serving_profile(model, config)
                for result in results:
                    writer.writerow(result)
            elif benchmark == "accuracy":
                # Merge with acc.yaml before running accuracy
                print(f"\n=== Running accuracy benchmark for {model} ===")
                config_dir = os.path.dirname(os.path.abspath(args.config))
                acc_path = args.acc or os.path.join(config_dir, 'acc.yaml')

                if os.path.exists(acc_path):
                    task_defaults, model_entries = load_acc_entries(acc_path)

                    if not task_defaults and not model_entries:
                        print(f"  Warning: acc.yaml at {acc_path} is empty, running accuracy with base config only")
                    else:
                        # Two-level merge: 1) task defaults, 2) model overrides
                        # Use original repo ID for token matching (not local path)
                        full_model = original_repo_id
                        pipeline_tag = config.get('hf_pipeline_tag', 'text-generation')

                        # Level 1: Apply task-level defaults based on hf_pipeline_tag
                        if pipeline_tag in task_defaults:
                            task_entry = task_defaults[pipeline_tag]
                            # Merge evaluations from task defaults
                            if 'evaluations' not in config and 'evaluations' in task_entry:
                                config['evaluations'] = task_entry['evaluations']
                                print(f"  Applied task-level defaults for hf_pipeline_tag={pipeline_tag}")

                        # Level 2: Apply model-specific overrides
                        base_tokens = _tokens(full_model)
                        matched = next((p for p in model_entries if p['models'] & base_tokens), None)

                        if matched:
                            # Apply model overrides (model wins over task defaults)
                            base_extra = config.get('extra_args_dict', {})
                            base_bench = config.get('bench_args_dict', {})
                            base_env = config.get('env_dict', {})

                            config['extra_args_dict'] = _fill_missing(base_extra, matched['extra_args'])
                            config['bench_args_dict'] = _fill_missing(base_bench, matched['bench_args'])
                            config['env_dict'] = _fill_missing(base_env, matched['env'])

                            if 'evaluations' in matched['raw']:
                                config['evaluations'] = deep_merge_evaluations(
                                    config.get('evaluations', []),
                                    matched['raw']['evaluations']
                                )

                            # Level 3: Apply parallelism-specific overrides
                            parallelism_overrides = matched['raw'].get('overrides', [])
                            if parallelism_overrides:
                                ov = find_parallelism_override(parallelism_overrides, config)
                                if ov:
                                    if 'evaluations' in ov:
                                        config['evaluations'] = deep_merge_evaluations(
                                            config.get('evaluations', []),
                                            ov['evaluations']
                                        )
                                    ov_extra = ov.get('extra_args', {})
                                    if ov_extra:
                                        config['extra_args_dict'] = _fill_missing(
                                            config.get('extra_args_dict', {}), ov_extra
                                        )
                                    ov_env = ov.get('env', {})
                                    if ov_env:
                                        config['env_dict'] = _fill_missing(
                                            config.get('env_dict', {}), ov_env
                                        )
                                    print(
                                        f"  Applied parallelism override "
                                        f"({', '.join(f'{k}={ov[k]}' for k in PARALLELISM_KEYS if k in ov)})"
                                    )

                            for k, v in matched['raw'].items():
                                if k not in ['model', 'extra_args', 'bench_args', 'env', 'evaluations', 'overrides']:
                                    if k not in config:
                                        config[k] = v

                            # Merge bench_args into config for accuracy runner
                            for k, v in config['bench_args_dict'].items():
                                if k not in config:
                                    config[k] = v

                            # Rebuild env and extra_args strings
                            env_vars_str = " ".join(f"{k}={v}" for k, v in config['env_dict'].items())
                            extra_args_str = ""
                            for k, v in config['extra_args_dict'].items():
                                extra_args_str += _format_cli_arg(k, v)
                            config["env"] = env_vars_str
                            config["extra_args"] = extra_args_str

                            print(f"  Applied model-specific overrides for {full_model}")
                        else:
                            print(f"  No model-specific entry, using task-level defaults only")
                else:
                    print(f"  Warning: acc.yaml not found at {acc_path}, running accuracy with base config only")

                results = run_accuracy(model, config)
                for result in results:
                    writer.writerow(result)
            else:
                raise ValueError(f"Unknown benchmark: {benchmark}")


if __name__ == "__main__":
    main()
