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
from typing import List, Dict

SUPPORTED_LIST_ARGS = ['model', 'tp', 'inp', 'out', 'bs', 'num_prompts', 'max_concurrency']
CSV_HEADER = [
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

def run_serving(model, config):
    # by default use num_prompts = 10 * max_concurrency if not specified
    if not config.get("num_prompts"):
        config["num_prompts"] = str(10 * int(config["max_concurrency"]))
    output_json = (
        f"{config['model']}_serving_{config['tp']}_{config['inp']}_{config['out']}_{config['num_prompts']}_{config['max_concurrency']}.json"
    )
    server_cmd = (
        "vllm serve "
        f"{model} "
        f"--dtype {config['dtype']} "
        f"-tp {config['tp']} "
        f"--trust-remote-code "
        f" --swap-space 16 "
        f" --disable-uvicorn-access-log"
    )
    # pop env and extra args from config
    env = config.pop('env', "")
    extra_args = config.pop('extra_args', "")
    server_cmd = f"{env} {server_cmd} {extra_args}".strip()
    config["cmd"] = server_cmd

    # start server
    print(server_cmd)
    server = subprocess.Popen(server_cmd, shell=True)

    try:
        # wait for server to start; timeout after 30 minutes
        status = subprocess.run(
            "timeout 1800 bash -c 'until curl -s http://localhost:8000/v1/models; do sleep 30; done' || exit 1",
            shell=True
        )
        if status.returncode != 0:
            print("Server failed to start")
            return [config]
        else:
            print(f"Server at {server.pid} contacted successfully", flush=True)

        # run benchmark
        bench_cmd = (
            "vllm bench serve "
            f"--model {model} "
            f"--percentile-metrics tpot,itl,e2el "
            f"--dataset-name random "
            f"--ignore-eos "
            f"--max-concurrency {config['max_concurrency']} "
            f"--num-prompts {config['num_prompts']} "
            f"--random-input-len {config['inp']} "
            f"--random-output-len {config['out']} "
            f"--trust-remote-code "
            f"--save-result "
            f"--result-filename {output_json}"
        )
        bench_args = config.pop('bench_args', {})
        bench_args_str = ""
        for k, v in bench_args.items():
            if isinstance(v, bool):
                bench_args_str += f"--{k} "
            else:
                bench_args_str += f"--{k} {v} "
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

def run_accuracy(model, config):
    output_json = (
        f"{config['model']}_accuracy_{config['tp']}.json"
    )
    server_cmd = (
        "vllm serve "
        f"{model} "
        f"--dtype {config['dtype']} "
        f"-tp {config['tp']} "
        f"--trust-remote-code "
        f" --swap-space 16 "
        f" --disable-uvicorn-access-log"
    )
    # pop env and extra args from config
    env = config.pop('env', "")
    extra_args = config.pop('extra_args', "")
    server_cmd = f"{env} {server_cmd} {extra_args}".strip()
    config["cmd"] = server_cmd

    # start server
    print(server_cmd)
    server = subprocess.Popen(server_cmd, shell=True)

    try:
        # wait for server to start; timeout after 30 minutes
        status = subprocess.run(
            "timeout 1800 bash -c 'until curl -s http://localhost:8000/v1/models; do sleep 30; done' || exit 1",
            shell=True
        )
        if status.returncode != 0:
            print("Server failed to start")
            return [config]
        else:
            print(f"Server at {server.pid} contacted successfully", flush=True)

        # run benchmark
        model_args = {
            "model": model,
            "max_gen_toks": 2048,
            "num_concurrent": 256,
            "max_retries": 10,
            "base_url": "http://localhost:8000/v1/completions"
        }
        model_args = ",".join([f"{k}={v}" for k, v in model_args.items()])
        bench_cmd = (
            "lm_eval "
            "--model local-completions "
            f"--model_args {model_args} "
            f"--tasks gsm8k "
            f"--limit 250 "
            f"--output_path ./tmp "
        )
        bench_args = config.pop('bench_args', {})
        bench_args_str = ""
        for k, v in bench_args.items():
            if isinstance(v, bool):
                bench_args_str += f"--{k} "
            else:
                bench_args_str += f"--{k} {v} "
        bench_cmd = f"{bench_cmd} {bench_args_str}".strip()
        config["cmd"] = f"{server_cmd};{bench_cmd}"
        print(bench_cmd)
        subprocess.run(bench_cmd, shell=True, check=True)

        # find output file and move into output_json
        output_files = glob.glob("./tmp/*/*.json")
        if len(output_files) == 0:
            print("No output files found")
            return [config]
        elif len(output_files) > 1:
            print(f"Multiple output files found: {output_files}")
            return [config]
        else:
            output_file = output_files[0]
            shutil.move(output_file, output_json)
            shutil.rmtree("./tmp")

        # parse output json
        results = []
        
        with open(output_json, "r", newline="", encoding="utf-8") as f:
            output = json.load(f)
            if "results" in output and "gsm8k" in output["results"]:
                gsm8k_results = output["results"]["gsm8k"]
                if "exact_match,flexible-extract" in gsm8k_results:
                    result = {
                        "performance": gsm8k_results["exact_match,flexible-extract"],
                        "metric": "exact_match,flexible-extract",
                        "unit": "percent",
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
            # Use dataprovider if present for model weights
            if MAD_DATAHOME := os.environ.get('MAD_DATAHOME'):
                model = MAD_DATAHOME
            elif config.get('extra_args', {}).get('load-format', None) == 'dummy':
                print("Found --load-format dummy in config, using dummy weights for benchmarking")
            else:
                # Explicitly download model before running benchmarks for easier debugging
                download_command=f"hf download {model} --exclude \"original/*\" \"*.tf\" \"*.onnx\" \"*.flax\" \"*.rust\""
                subprocess.run(download_command, shell=True, check=True)
            
            # concatenate env vars and extra args into the corresponding strings
            env_vars = config.get("env", {})
            extra_args = config.get("extra_args", {})
            env_vars_str = " ".join(f"{k}={v}" for k, v in env_vars.items())
            extra_args_str = ""
            for k, v in extra_args.items():
                if isinstance(v, bool):
                    extra_args_str += f" --{k}"
                else:
                    extra_args_str += f" --{k} {v}"
            config["env"] = env_vars_str
            config["extra_args"] = extra_args_str
            
            # run benchmark
            results = []
            benchmark = config["benchmark"]
            if benchmark == "latency":
                results = run_latency(model, config)
            elif benchmark == "throughput":
                results = run_throughput(model, config)
            elif benchmark == "serving":
                results = run_serving(model, config)
            elif benchmark == "accuracy":
                results = run_accuracy(model, config)
            else:
                raise ValueError(f"Unknown benchmark: {benchmark}")
            
            # Write results to csv
            for result in results:
                writer.writerow(result)


if __name__ == "__main__":
    main()