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
"""Benchmark orchestrator for vLLM-Omni diffusion models (text-to-image / video).

Diffusion models are served by vLLM-Omni (`vllm serve <model> --omni`, OpenAI
image-generation API) and benchmarked with the omni diffusion serving bench
(`benchmarks/diffusion/diffusion_benchmark_serving.py`). Neither exists on the
plain vLLM image, hence a dedicated engine rather than a branch in scripts/vllm.
"""

import os
import sys
import csv
import json
import time
import yaml
import psutil
import signal
import argparse
import itertools
import subprocess
from typing import List, Dict

from accuracy_runner_vllm_omni import run_accuracy_vllm_omni

SUPPORTED_LIST_ARGS = ['model', 'tp', 'num_prompts', 'max_concurrency']

CSV_HEADER = [
    "hf_pipeline_tag",
    "model",
    "benchmark",
    "tp",
    "dtype",
    "num_prompts",
    "max_concurrency",
    "concurrency",
    "num_images",
    "num_inference_steps",
    "height",
    "width",
    "cmd",
    "performance",
    "metric",
    "unit",
]

# hf_pipeline_tag -> diffusion_benchmark_serving.py `--task`
HF_PIPELINE_TO_TASK = {
    "text-to-image": "t2i",
    "image-to-image": "i2i",
    "text-to-video": "t2v",
    "image-to-video": "i2v",
}

SERVER_PORT = 8000


# ---------------------------------------------------------------------------
# Perf overrides: auto-fill of a sibling perf.yaml into --config.
#
# perf.yaml entries are keyed by a space-separated `model` list and carry only
# `extra_args` / `env` (plus `docker_base`, which is consumed by the CI build
# layer, not here). Merge is a gap-fill: the base config always wins.
# ---------------------------------------------------------------------------

def _tokens(model_field):
    """Split a `model:` value into a set of repo tokens."""
    if model_field is None:
        return set()
    return {t.strip() for t in str(model_field).split() if t.strip()}


def _format_cli_arg(k, v):
    """Serialize one (key, value) pair to a CLI fragment."""
    if isinstance(v, bool):
        return f"{k} " if v else ""
    return f"{k} {v} "


def _fill_missing(base, overrides):
    """Return `base` with any keys it lacks filled in from `overrides`."""
    out = dict(base or {})
    for k, v in (overrides or {}).items():
        if k not in out:
            out[k] = v
    return out


def load_perf_entries(perf_path):
    """Load perf.yaml into a normalized list of override entries."""
    if not perf_path or not os.path.exists(perf_path):
        return []
    with open(perf_path, 'r') as f:
        raw = yaml.safe_load(f) or []
    return [{
        'models': _tokens(e.get('model')),
        'extra_args': e.get('extra_args', {}) or {},
        'env': e.get('env', {}) or {},
    } for e in raw]


def apply_perf_overrides(configs, perf_entries):
    """Mutate `configs` in place, gap-filling matching perf overrides."""
    summary = []
    for cfg in configs:
        base_tokens = _tokens(cfg.get('model'))
        matched = next((p for p in perf_entries if p['models'] & base_tokens), None)
        if not matched:
            continue
        base_extra = cfg.get('extra_args') or {}
        base_env = cfg.get('env') or {}
        cfg['extra_args'] = _fill_missing(base_extra, matched['extra_args'])
        cfg['env'] = _fill_missing(base_env, matched['env'])
        summary.append({
            'model': cfg.get('model'),
            'matched_perf_models': sorted(matched['models']),
            'extra_args_added': sorted(k for k in matched['extra_args'] if k not in base_extra),
            'env_added': sorted(k for k in matched['env'] if k not in base_env),
        })
    return summary


def load_acc_entries(acc_path):
    """Load acc.yaml with task-level and model-level entries."""
    if not acc_path or not os.path.exists(acc_path):
        return {}, []
    with open(acc_path, 'r') as f:
        raw = yaml.safe_load(f) or []
    task_defaults = {}
    model_entries = []
    for e in raw:
        if 'model' not in e and 'hf_pipeline_tag' in e:
            task_defaults[e['hf_pipeline_tag']] = e
        elif 'model' in e:
            model_entries.append({
                'models': _tokens(e.get('model')),
                'extra_args': e.get('extra_args', {}) or {},
                'env': e.get('env', {}) or {},
                'raw': e,
            })
    return task_defaults, model_entries


def parse_args():
    parser = argparse.ArgumentParser(description='Run vLLM-Omni diffusion benchmark')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the benchmark config yaml')
    parser.add_argument('--model', type=str, default=None,
                        help='Model repo to run (filters the config)')
    parser.add_argument('--benchmark', type=str, default='all',
                        help='Benchmark to run; only "serving" is supported for diffusion')
    parser.add_argument('--tp', type=str, default=None)
    parser.add_argument('--num_prompts', type=str, default=None)
    parser.add_argument('--max_concurrency', type=str, default=None)
    parser.add_argument('--perf', type=str, default=None,
                        help='Path to perf.yaml (defaults to a sibling of --config)')
    parser.add_argument('--perf-output', dest='perf_output', type=str, default=None)
    parser.add_argument('--no-perf-merge', dest='no_perf_merge', action='store_true')
    parser.add_argument('--acc', type=str, default=None,
                        help='Path to acc.yaml (defaults to a sibling of --config)')
    parser.add_argument('--acc-output', dest='acc_output', type=str, default=None)
    return parser.parse_args()


def expand_configs(args, configs: List[Dict]):
    """Apply arch overrides, expand list-valued keys, then filter by CLI args."""
    cfgs = []
    arch = os.environ.get('MAD_SYSTEM_GPU_ARCHITECTURE', 'unknown')
    for config in configs:
        cfg = config.copy()
        arch_overrides = cfg.pop('arch_overrides', {})
        if arch_override := arch_overrides.get(arch, {}):
            print(f"Detected {arch} architecture, applying override {arch_override}")
            cfg.update(arch_override)
        cfgs.append(cfg)

    config_list = []
    for cfg in cfgs:
        common_cfgs = {k: v for k, v in cfg.items() if k not in SUPPORTED_LIST_ARGS}
        list_cfgs = {k: str(v).split(' ') for k, v in cfg.items() if k in SUPPORTED_LIST_ARGS}
        expanded = [dict(zip(list_cfgs.keys(), x)) for x in itertools.product(*list_cfgs.values())]
        for e in expanded:
            config_list.append({**common_cfgs, **e})

    filtered = config_list
    for arg_name in SUPPORTED_LIST_ARGS:
        if arg_val := getattr(args, arg_name, None):
            print(f"Filtering configs by {arg_name}={arg_val}")
            filtered = [cfg for cfg in filtered if cfg.get(arg_name, None) == arg_val]

    if args.benchmark and args.benchmark != "all":
        print(f"Filtering configs by benchmark={args.benchmark}")
        filtered = [cfg for cfg in filtered if cfg.get('benchmark', None) == args.benchmark]

    return filtered


def _kill_server(server, timeout=120):
    """Kill the vllm serve subprocess and all its children.

    Teardown is bounded: the omni diffusion server shuts down N tensor-parallel
    workers and can wedge (it already logs "Terminating diffusion worker ...
    after timeout" during a normal exit). An unbounded communicate() would hang
    the whole job until the madengine timeout, so escalate SIGINT -> SIGKILL.
    """
    try:
        parent = psutil.Process(server.pid)
        for child in parent.children(recursive=True):
            child.send_signal(signal.SIGINT)
        server.send_signal(signal.SIGINT)
        try:
            _ = server.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"Server did not exit within {timeout}s of SIGINT — killing", flush=True)
            try:
                parent = psutil.Process(server.pid)
                for child in parent.children(recursive=True):
                    child.kill()
            except (psutil.NoSuchProcess, ProcessLookupError):
                pass
            server.kill()
            try:
                _ = server.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                print("Server still alive after SIGKILL; continuing", flush=True)
    except (psutil.NoSuchProcess, ProcessLookupError):
        pass
    del server


def _wait_for_server(server, timeout=1800, poll_interval=15):
    """Poll /v1/models until ready. Returns False early if the server dies."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if server.poll() is not None:
            print(f"Server process exited (code {server.returncode}) before becoming "
                  f"ready — aborting wait", flush=True)
            return False
        try:
            code = subprocess.run(
                f"curl -s -o /dev/null -w '%{{http_code}}' http://localhost:{SERVER_PORT}/v1/models",
                shell=True, capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            if code == "200":
                return True
        except subprocess.TimeoutExpired:
            pass
        time.sleep(poll_interval)
    print(f"Server did not become ready within {timeout}s", flush=True)
    return False


def find_diffusion_bench():
    """Locate the vLLM-Omni diffusion serving bench (known path, then fallback search)."""
    known = "/app/vllm-omni/benchmarks/diffusion/diffusion_benchmark_serving.py"
    if os.path.isfile(known):
        return known
    for root in ("/app", "/workspace", "/usr/local/lib"):
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            if "diffusion_benchmark_serving.py" in files:
                return os.path.join(dirpath, "diffusion_benchmark_serving.py")
    raise FileNotFoundError(
        "diffusion_benchmark_serving.py not found — is this a vllm-omni image? "
        "Expected docker_base vllm/vllm-omni-rocm:* (see configs/perf.yaml)."
    )


def build_server_cmd(model, config, env="", extra_args=""):
    """Build a `vllm serve --omni` command (minimal; tuning via extra_args)."""
    cmd = (
        "vllm serve "
        f"{model} "
        f"--omni "
        f"--dtype {config['dtype']} "
        f"-tp {config['tp']} "
        f"--host 0.0.0.0 "
        f"--port {SERVER_PORT} "
    )
    return f"{env} {cmd} {extra_args}".strip()


def run_image_serving(model, config):
    """Start `vllm serve --omni`, run the omni diffusion bench, parse latency/throughput."""
    hf_pipeline_tag = config.get("hf_pipeline_tag", "text-to-image")
    task = HF_PIPELINE_TO_TASK.get(hf_pipeline_tag)
    if task is None:
        raise ValueError(f"Unsupported hf_pipeline_tag: {hf_pipeline_tag}")

    width = config.get("width", 1024)
    height = config.get("height", 1024)
    # The omni diffusion bench has no images-per-request flag (only --num-input-images,
    # which is i2i *input*), so >1 output per prompt cannot be requested. Fail loudly
    # rather than silently benchmarking 1 image while reporting num_images>1.
    num_images = int(config.get("num_images", 1))
    if num_images != 1:
        raise ValueError(
            f"num_images={num_images} is not supported: the vLLM-Omni diffusion bench "
            f"cannot set images-per-request. Use num_images: 1."
        )
    num_inference_steps = config.get("num_inference_steps")  # None -> bench default (50)
    max_concurrency = str(config.get("max_concurrency") or 1)
    config["max_concurrency"] = max_concurrency
    if not config.get("num_prompts"):
        config["num_prompts"] = str(5 * int(max_concurrency))
    num_prompts = config["num_prompts"]
    dataset = config.get("dataset", "vbench")
    warmup_requests = config.get("warmup_requests", 2)

    output_json = (
        f"{config['model']}_image_{hf_pipeline_tag}_{height}x{width}"
        f"_{num_prompts}_{max_concurrency}.json"
    )

    env = config.pop("env", "")
    extra_args = config.pop("extra_args", "")
    server_cmd = build_server_cmd(model, config, env, extra_args)
    config["cmd"] = server_cmd

    print(server_cmd)
    server = subprocess.Popen(server_cmd, shell=True)

    try:
        if not _wait_for_server(server):
            # Fail the run rather than returning a metric-less row: a blank CSV
            # row exits 0 and reads downstream as a successful benchmark.
            raise RuntimeError(
                f"Omni diffusion server failed to start for {model} "
                f"({height}x{width}, tp={config['tp']})")
        print(f"Omni server at {server.pid} contacted successfully", flush=True)

        bench = find_diffusion_bench()
        bench_cmd = (
            f"python3 {bench} "
            f"--base-url http://localhost:{SERVER_PORT} "
            f"--model {model} "
            f"--task {task} "
            f"--dataset {dataset} "
            f"--num-prompts {num_prompts} "
            f"--max-concurrency {max_concurrency} "
            f"--width {width} "
            f"--height {height} "
            f"--warmup-requests {warmup_requests} "
            f"--output-file {output_json}"
        )
        if num_inference_steps is not None:
            bench_cmd += f" --num-inference-steps {num_inference_steps}"

        config["cmd"] = f"{server_cmd};{bench_cmd}"
        print(bench_cmd)
        subprocess.run(bench_cmd, shell=True, check=True)

        metrics_map = {
            "latency_mean": ("latency_mean", "s"),
            "latency_median": ("latency_median", "s"),
            "latency_p99": ("latency_p99", "s"),
            "throughput_qps": ("throughput_qps", "req/s"),
        }
        common = {
            "hf_pipeline_tag": hf_pipeline_tag,
            "height": height,
            "width": width,
            "num_images": num_images,
            "num_inference_steps": num_inference_steps if num_inference_steps is not None else "",
            "concurrency": max_concurrency,
        }
        results = []
        with open(output_json, "r", newline="", encoding="utf-8") as f:
            output = json.load(f)
            for json_key, (metric, unit) in metrics_map.items():
                if json_key not in output:
                    continue
                results.append({
                    "performance": str(output[json_key]),
                    "metric": metric,
                    "unit": unit,
                    **config,
                    **common,
                })

            # The omni bench reports only request throughput. Derive outputs/s
            # (images/s) so the dashboard's output-throughput field is populated
            # instead of defaulting to 0; with num_images pinned to 1 this equals
            # throughput_qps.
            if "throughput_qps" in output:
                results.append({
                    "performance": str(float(output["throughput_qps"]) * num_images),
                    "metric": "output_throughput_ops",
                    "unit": "outputs/s",
                    **config,
                    **common,
                })
        return results

    finally:
        _kill_server(server)


def predownload_model(model):
    """Pre-download weights so `vllm serve` loads from a COMPLETE local cache.

    A transient HF-Hub stall during model load otherwise hangs the server. The
    download runs in a subprocess with a hard wall-clock timeout because a
    silently stalled connection raises no exception; SIGKILL + retry resumes
    from the partial cache, so progress is monotonic across attempts.
    """
    print("Explicitly pre-downloading model from HF Hub")
    ignore = ["original/*", "*.tf", "*.onnx", "*.flax", "*.rust"]
    script = (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download({model!r}, ignore_patterns={ignore!r}, max_workers=8)"
    )
    timeout = int(os.environ.get("PREDOWNLOAD_TIMEOUT_SEC", "3600"))
    for attempt in range(1, 6):
        try:
            subprocess.run([sys.executable, "-c", script], check=True, timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            print(f"HF pre-download attempt {attempt}/5 stalled >{timeout}s — killed; "
                  f"retrying (resumes from partial cache)...", flush=True)
        except subprocess.CalledProcessError as e:
            print(f"HF pre-download attempt {attempt}/5 failed (exit {e.returncode}); "
                  f"retrying...", flush=True)
            time.sleep(min(30 * attempt, 120))
    raise RuntimeError(
        f"Model pre-download did not complete after 5 attempts ({timeout}s each)")


def main():
    args = parse_args()

    with open(args.config, 'r') as f:
        print(f"Loading configs from {args.config}")
        configs = yaml.safe_load(f) or []

    # Perf merge: auto-detect a sibling perf.yaml, gap-fill extra_args/env into
    # each base entry in memory, and materialize the result to perf_config.yaml.
    # The original --config file is never modified.
    if not args.no_perf_merge:
        config_dir = os.path.dirname(os.path.abspath(args.config))
        perf_path = args.perf or os.path.join(config_dir, 'perf.yaml')
        perf_output = args.perf_output or os.path.join(config_dir, 'perf_config.yaml')
        if os.path.abspath(perf_output) == os.path.abspath(args.config):
            raise ValueError(
                f"--perf-output must differ from --config to keep the source untouched; "
                f"both resolved to {perf_output}")
        perf_entries = load_perf_entries(perf_path)
        if perf_entries:
            summary = apply_perf_overrides(configs, perf_entries)
            if summary:
                print(f"Applied perf overrides from {perf_path} (base wins; perf fills gaps):")
                for s in summary:
                    print(f"  model={s['model']} <- perf {s['matched_perf_models']}\n"
                          f"    extra_args added: {s['extra_args_added']}\n"
                          f"    env added:        {s['env_added']}")
                with open(perf_output, 'w') as f:
                    yaml.safe_dump(configs, f, sort_keys=False, default_flow_style=False)
                print(f"Wrote merged config to {perf_output}")
        else:
            print(f"No perf.yaml found at {perf_path} (or empty); skipping perf merge")

    # Accuracy merge: two-level merge from acc.yaml (task defaults + model overrides)
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
        for cfg in configs:
            pipeline_tag = cfg.get('hf_pipeline_tag', 'text-to-image')
            if pipeline_tag in task_defaults:
                task_entry = task_defaults[pipeline_tag]
                if 'evaluations' not in cfg and 'evaluations' in task_entry:
                    cfg['evaluations'] = task_entry['evaluations']
            base_tokens = _tokens(cfg.get('model'))
            matched = next((p for p in model_entries if p['models'] & base_tokens), None)
            if matched:
                base_extra = cfg.get('extra_args') or {}
                base_env = cfg.get('env') or {}
                cfg['extra_args'] = _fill_missing(base_extra, matched['extra_args'])
                cfg['env'] = _fill_missing(base_env, matched['env'])
                if 'evaluations' in matched['raw']:
                    cfg['evaluations'] = matched['raw']['evaluations']
                print(f"  Applied acc overrides for model={cfg.get('model')} <- {sorted(matched['models'])}")
        with open(acc_output, 'w') as f:
            yaml.safe_dump(configs, f, sort_keys=False, default_flow_style=False)
        print(f"Wrote merged accuracy config to {acc_output}")

    configs = expand_configs(args, configs)
    print("Running configs: ", *configs, sep='\n')

    for config in configs:
        model = config['model']
        config['model'] = os.path.basename(model)

        OUTPUT_CSV = "perf_" + os.path.basename(model) + ".csv"
        header_write = 0 if os.path.exists(OUTPUT_CSV) else 1
        with open(OUTPUT_CSV, "a+", newline="") as outf:
            writer = csv.DictWriter(outf, delimiter=",", fieldnames=CSV_HEADER,
                                    extrasaction="ignore")
            if header_write:
                writer.writeheader()
                outf.flush()

            if MAD_SECRETS_HFTOKEN := os.environ.get('MAD_SECRETS_HFTOKEN'):
                os.environ['HF_TOKEN'] = MAD_SECRETS_HFTOKEN
            else:
                print("Warning: MAD_SECRETS_HFTOKEN is not set. FLUX models are gated; "
                      "set MAD_SECRETS_HFTOKEN=<your-huggingface-token>")

            CHECK_LOCAL_DATA = os.environ.get('CHECK_LOCAL_DATA', 'false').lower() == 'true'
            if CHECK_LOCAL_DATA and (MAD_DATAHOME := os.environ.get('MAD_DATAHOME')) \
                    and os.path.exists(os.path.join(MAD_DATAHOME, model)):
                model = os.path.join(MAD_DATAHOME, model)
                print("Found MAD_DATAHOME")
            else:
                predownload_model(model)

            env_vars = config.get("env", {})
            extra_args = config.get("extra_args", {})
            config["env"] = " ".join(f"{k}={v}" for k, v in env_vars.items())
            config["extra_args"] = "".join(_format_cli_arg(k, v) for k, v in extra_args.items())

            benchmark = config.get("benchmark", "serving")
            if benchmark == "serving":
                results = run_image_serving(model, config)
            elif benchmark == "accuracy":
                results = run_accuracy_vllm_omni(
                    model, config,
                    wait_for_server_fn=_wait_for_server,
                    kill_server_fn=_kill_server,
                    build_server_cmd_fn=build_server_cmd,
                    server_port=SERVER_PORT,
                )
            else:
                raise ValueError(
                    f"vllm-omni diffusion supports benchmark=serving|accuracy (got {benchmark})")

            for result in results:
                writer.writerow(result)

            if benchmark == "accuracy":
                failed = [r for r in results if r.get("performance") == "fail"]
                if failed:
                    metrics = [r.get("metric", "unknown") for r in failed]
                    print(f"\nACCURACY CHECK FAILED: {', '.join(metrics)} below threshold")
                    sys.exit(1)


if __name__ == "__main__":
    main()
