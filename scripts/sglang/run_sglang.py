#################################################################################
#
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
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
"""Config-driven online serving benchmark for SGLang.

Structured after scripts/vllm/run_vllm.py, which solves the same problem for
vLLM, and emits the same perf CSV schema so MAD's multiple_results ingestion is
shared. The existing scripts/sglang/sglang_benchmark_report.sh stays as-is; it
drives the offline bench_one_batch / bench_offline_throughput path.
"""

import os
import sys
import csv
import json
import yaml
import psutil
import signal
import argparse
import itertools
import subprocess
from typing import List, Dict

SUPPORTED_LIST_ARGS = ['model', 'tp', 'inp', 'out', 'num_prompts', 'max_concurrency']
CSV_HEADER = [
    "model",
    "benchmark",
    "variant",
    "tp",
    "inp",
    "out",
    "dtype",
    "num_prompts",
    "max_concurrency",
    "cmd",
    "performance",
    "metric",
    "unit",
]

HOST = "127.0.0.1"
PORT = 30000
# Kimi K3 is a ~1.5 TB checkpoint loaded over TP8; 30 minutes (what run_vllm.py
# allows) is not enough. Overridable for smaller models.
SERVER_START_TIMEOUT = int(os.environ.get("SGLANG_SERVER_START_TIMEOUT", 5400))


def parse_args():
    parser = argparse.ArgumentParser(description='Run SGLang serving benchmark')
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
    parser.add_argument('--variant',
                            type=str,
                            help='select variant from config',
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

    # filter configs by benchmark
    if args.benchmark and args.benchmark != "all":
        print(f"Filtering configs by benchmark={args.benchmark}")
        filtered_configs = [cfg for cfg in filtered_configs if cfg["benchmark"] == args.benchmark]

    # filter configs by variant; variants share a model, so this is the only way
    # to select between recipes such as K3 nospec and K3 dspark
    if args.variant and args.variant != "all":
        print(f"Filtering configs by variant={args.variant}")
        filtered_configs = [cfg for cfg in filtered_configs if cfg.get("variant", None) == args.variant]

    return filtered_configs


def read_last_json_line(path: str):
    """SGLang appends one JSON object per run to --output-file, so the result of
    this run is the last non-empty line (vLLM writes a plain JSON document)."""
    with open(path, "r", newline="", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    if not lines:
        raise Exception(f"No benchmark results found in {path}")
    return json.loads(lines[-1])


def run_serving(model, config):
    # by default use num_prompts = 10 * max_concurrency if not specified
    if not config.get("num_prompts"):
        config["num_prompts"] = str(10 * int(config["max_concurrency"]))
    server_cmd = (
        "sglang serve "
        f"--model-path {model} "
        f"--dtype {config['dtype']} "
        f"--tp-size {config['tp']} "
        f"--trust-remote-code "
        f"--host {HOST} "
        f"--port {PORT} "
    )
    # pop env and extra args from config
    env = config.pop('env', "")
    extra_args = config.pop('extra_args', "")
    server_cmd = f"{env} {server_cmd} {extra_args}".strip()
    config["cmd"] = server_cmd

    # start server
    print(server_cmd, flush=True)
    server = subprocess.Popen(server_cmd, shell=True)
    results = []

    try:
        # wait for the server to become ready. /health only returns 200 once the
        # server leaves the Starting state, whereas /v1/models answers earlier.
        status = subprocess.run(
            f"timeout {SERVER_START_TIMEOUT} bash -c "
            f"'until curl -sf http://{HOST}:{PORT}/health; do sleep 30; done' || exit 1",
            shell=True
        )
        if status.returncode != 0:
            raise Exception("Server failed to start")
        else:
            print(f"Server at {server.pid} contacted successfully", flush=True)

        # run serving benchmark
        output_json = (
            f"{config['model']}_{config['variant']}_serving_{config['tp']}_{config['inp']}_"
            f"{config['out']}_{config['num_prompts']}_{config['max_concurrency']}.jsonl"
        )
        bench_cmd = (
            "python3 -m sglang.benchmark.serving "
            f"--backend sglang "
            f"--host {HOST} "
            f"--port {PORT} "
            f"--model {model} "
            f"--dataset-name random "
            f"--random-input-len {config['inp']} "
            f"--random-output-len {config['out']} "
            f"--random-range-ratio 1.0 "
            f"--max-concurrency {config['max_concurrency']} "
            f"--num-prompts {config['num_prompts']} "
            f"--output-file {output_json}"
        )
        config["cmd"] = f"{server_cmd};{bench_cmd}"
        print(bench_cmd, flush=True)
        subprocess.run(bench_cmd, shell=True, check=True)

        # parse output jsonl
        output = read_last_json_line(output_json)
        if "total_throughput" in output:
            metrics = {
                "throughput_tot": str(output["total_throughput"]),
                "throughput_gen": str(output["output_throughput"]),
                "median_ttft": str(output["median_ttft_ms"]),
                "median_tpot": str(output["median_tpot_ms"]),
                "median_itl": str(output["median_itl_ms"]),
                # SGLang names this median_e2e_latency_ms, not vLLM's median_e2el_ms
                "median_e2el": str(output["median_e2e_latency_ms"]),
            }
            # only reported under speculative decoding
            if output.get("accept_length"):
                metrics["accept_length"] = str(output["accept_length"])
            for metric, perf in metrics.items():
                if "throughput" in metric:
                    unit = "tok/sec"
                elif metric == "accept_length":
                    unit = "tokens"
                else:
                    unit = "ms"
                result = {
                    "performance": perf,
                    "metric": metric,
                    "unit": unit,
                    **config
                }
                results.append(result)

    finally:
        # kill server and children
        parent = psutil.Process(server.pid)
        for child in parent.children(recursive=True):
            child.send_signal(signal.SIGINT)
        server.send_signal(signal.SIGINT)
        _ = server.communicate()
        del server

    return results


def main():
    args = parse_args()

    # Load, expand and filter configs
    with open(args.config, 'r') as f:
        print(f"Loading configs from {args.config}")
        configs = yaml.safe_load(f)
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
            writer = csv.DictWriter(outf, delimiter=",", fieldnames=CSV_HEADER)
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
            elif config.get('extra_args', {}).get('--load-format', None) == 'dummy':
                print("Found --load-format dummy in config, using dummy weights for benchmarking")
            elif os.path.isabs(model):
                print(f"Using local model path: {model}")
            else:
                # Explicitly pre-download the model so the server loads from a COMPLETE local
                # cache. The old `hf download {model} --exclude <globs>` CLI form silently fetched
                # 0 files on newer hf CLIs (the globs were parsed as explicit filenames), so nothing
                # was pre-staged and the full 1T download happened at server launch, blowing past the
                # startup wait. Use the Python API with a hard per-attempt timeout + bounded retries,
                # accelerated by hf_transfer (best-effort install; only enabled when importable).
                print("Explicitly pre-downloading model from HF Hub")
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
                _ignore = ["original/*", "*.tf", "*.onnx", "*.flax", "*.rust"]
                _dl_script = (
                    "from huggingface_hub import snapshot_download; "
                    f"snapshot_download({model!r}, ignore_patterns={_ignore!r}, max_workers=16)"
                )
                _dl_timeout = int(os.environ.get("PREDOWNLOAD_TIMEOUT_SEC", "3600"))
                _dl_ok = False
                for _attempt in range(1, 6):
                    try:
                        subprocess.run([sys.executable, "-c", _dl_script], check=True, timeout=_dl_timeout)
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
            env_vars = config.get("env", {})
            extra_args = config.get("extra_args", {})
            env_vars_str = " ".join(f"{k}={v}" for k, v in env_vars.items())
            extra_args_str = ""
            for k, v in extra_args.items():
                if isinstance(v, bool):
                    extra_args_str += f" {k}"
                else:
                    extra_args_str += f" {k} {v}"
            config["env"] = env_vars_str
            config["extra_args"] = extra_args_str

            # run benchmark
            results = []
            benchmark = config["benchmark"]
            if benchmark == "serving":
                results = run_serving(model, config)
            else:
                raise ValueError(f"Unknown benchmark: {benchmark}")

            # Write results to csv
            for result in results:
                writer.writerow(result)


if __name__ == "__main__":
    main()
