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
import re
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
import sys
import subprocess
import concurrent.futures
from typing import List, Dict

# Import sglang-specific accuracy runner
from accuracy_runner_sglang import run_accuracy_sglang

SUPPORTED_HF_PIPELINE_TAGS = {
    "text-generation",
    "text-to-image",
    "image-to-image",
    "text-to-video",
    "image-to-video",
}
SUPPORTED_LIST_ARGS = ['model', 'tp', 'inp', 'out', 'bs', 'num_prompts', 'max_concurrency', 'concurrency', 'num_frames']
FIXED_ENV_VARS = [
    "SGLANG_USE_AITER=1",
]
# Kimi K3 is a ~1.5 TB checkpoint loaded over TP8; 30 minutes is not enough.
SERVER_START_TIMEOUT = int(os.environ.get("SGLANG_SERVER_START_TIMEOUT", 5400))
CSV_HEADER = [
    "model",
    "benchmark",
    "hf_pipeline_tag",
    "tp",
    "inp",
    "out",
    "dtype",
    "quantization",
    "kv_cache_dtype",
    "num_prompts",
    "max_concurrency",
    "concurrency",
    "bs",
    "num_images",
    "num_frames",
    "num_inference_steps",
    "guidance_scale",
    "height",
    "width",
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

def wait_for_server(server, health_url, timeout=1800, poll_interval=30, fail_on_http_error=False):
    """Poll health_url until ready; return False early if the server exits.

    With fail_on_http_error, an HTTP error status (e.g. 503 while weights are
    still loading) counts as not ready instead of just a successful connection.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if server.poll() is not None:
            print(f"Server process exited early with code {server.returncode} "
                  f"before becoming ready at {health_url}")
            return False
        curl_flags = "-sf" if fail_on_http_error else "-s"
        if subprocess.run(f"curl {curl_flags} {health_url}", shell=True).returncode == 0:
            return True
        time.sleep(poll_interval)
    print(f"Timed out after {timeout}s waiting for server at {health_url}")
    return False


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


PARALLELISM_KEYS = ('tp', 'ep', 'dp', 'pp')


def find_parallelism_override(overrides, cfg):
    """Return the first override entry whose tp/ep/dp/pp filters all match cfg, or None."""
    for ov in overrides:
        if all(str(cfg.get(k)) == str(ov[k]) for k in PARALLELISM_KEYS if k in ov):
            return ov
    return None


def _build_acc_config(model, config, args):
    """Build an accuracy config by merging acc.yaml into a copy of *config*.

    Applies the two-level merge (task defaults then model overrides) and
    rebuilds env / extra_args strings. Used by the ``benchmark == "all"``
    paths so the merge logic is not duplicated per pipeline tag.
    """
    acc_config = dict(config)
    acc_config["benchmark"] = "accuracy"

    config_dir = os.path.dirname(os.path.abspath(args.config))
    acc_path = args.acc or os.path.join(config_dir, 'acc.yaml')

    if os.path.exists(acc_path):
        task_defaults, model_entries = load_acc_entries(acc_path)
        if task_defaults or model_entries:
            pipeline_tag = acc_config.get('hf_pipeline_tag', 'text-generation')

            if pipeline_tag in task_defaults:
                task_entry = task_defaults[pipeline_tag]
                if 'evaluations' not in acc_config and 'evaluations' in task_entry:
                    acc_config['evaluations'] = task_entry['evaluations']
                    print(f"  Applied task-level defaults for hf_pipeline_tag={pipeline_tag}")

            base_tokens = _tokens(model)
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

                env_vars_str = " ".join(FIXED_ENV_VARS + [f"{k}={v}" for k, v in acc_config['env_dict'].items()])
                extra_args_str = ""
                for k, v in acc_config['extra_args_dict'].items():
                    if isinstance(v, bool):
                        extra_args_str += f" {k}"
                    else:
                        extra_args_str += f" {k} {v}"
                acc_config["env"] = env_vars_str
                acc_config["extra_args"] = extra_args_str

                print(f"  Applied model-specific overrides from acc.yaml")

    return acc_config


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
            'env': e.get('env', {}) or {},
            'bench_args': e.get('bench_args', {}) or {},
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
            task_defaults[e['hf_pipeline_tag']] = e
        elif 'model' in e:
            model_entries.append({
                'models': _tokens(e.get('model')),
                'extra_args': e.get('extra_args', {}) or {},
                'env': e.get('env', {}) or {},
                'bench_args': e.get('bench_args', {}) or {},
                'raw': e,
            })

    return task_defaults, model_entries


def resolve_tokenizer_arg(model):
    """Return a `--tokenizer-path <file> ` fragment for models that ship only a
    tekken `tokenizer.tok.json` (no HF `tokenizer.json`); '' otherwise.

    SGLang loads the correct TiktokenTokenizer -- and thus produces decodable,
    non-empty streamed text (so bench_serving can record TTFT/ITL) -- only when
    the tokenizer path points at the tekken file. This is keyed on the tokenizer
    files the model ships, so it is model-agnostic.
    """
    def _pick(d):
        # tokenizer.json present -> standard HF tokenizer, no flag needed (definitive)
        if os.path.isfile(os.path.join(d, "tokenizer.json")):
            return "", True
        tok = os.path.join(d, "tokenizer.tok.json")
        if os.path.isfile(tok):
            return f"--tokenizer-path {tok} ", True
        return "", False  # inconclusive: neither tokenizer file is present locally
    if os.path.isdir(model):
        return _pick(model)[0]
    cache = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    for snap in sorted(glob.glob(os.path.join(
            cache, "models--" + model.replace("/", "--"), "snapshots", "*"))):
        arg, definitive = _pick(snap)
        if definitive:
            return arg
    # Not resolvable from the local cache -- e.g. on the first config the hub
    # pre-download fetches weights/config by explicit filename and skips the
    # tokenizer, so the tekken file is not cached yet. Ask the Hub and fetch just
    # that tiny file so --tokenizer-path is deterministic from the very first run.
    try:
        from huggingface_hub import HfApi, hf_hub_download
        files = set(HfApi().list_repo_files(model))
        if "tokenizer.tok.json" in files and "tokenizer.json" not in files:
            return f"--tokenizer-path {hf_hub_download(model, 'tokenizer.tok.json')} "
    except Exception as e:
        print(f"resolve_tokenizer_arg: hub fallback failed for {model}: {e}")
    return ""


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
        base_env   = cfg.get('env') or {}
        base_bench = cfg.get('bench_args') or {}
        cfg['extra_args'] = _fill_missing(base_extra, matched['extra_args'])
        cfg['env']        = _fill_missing(base_env,   matched['env'])
        cfg['bench_args'] = _fill_missing(base_bench, matched.get('bench_args', {}))

        summary.append({
            'model': cfg.get('model'),
            'matched_perf_models': sorted(matched['models']),
            'extra_args_added':   sorted(k for k in matched['extra_args'] if k not in base_extra),
            'extra_args_skipped': sorted(k for k in matched['extra_args'] if k in base_extra),
            'env_added':          sorted(k for k in matched['env'] if k not in base_env),
            'env_skipped':        sorted(k for k in matched['env'] if k in base_env),
        })
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description='Run SGLang benchmark')
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
    parser.add_argument('--concurrency',
                            type=str,
                            help='select concurrency from config (diffusion models)',
                            required=False,
                            default=None,
                        )
    parser.add_argument('--num_frames',
                            type=str,
                            help='select num frames from config (video generation)',
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
                            help='Kernel-profiling run: serve under rtl trace + roctx markers and '
                                 'emit profile/kernel_summary_payload.json (text-generation serving).',
                        )
    parser.add_argument('--variant',
                            type=str,
                            help='select variant from config',
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

    # filter configs by variant; variants share a model, so this is the only way
    # to select between recipes such as K3 nospec and K3 dspark
    if args.variant and args.variant != "all":
        print(f"Filtering configs by variant={args.variant}")
        filtered_configs = [cfg for cfg in filtered_configs if cfg.get("variant", None) == args.variant]

    return filtered_configs

def resolve_parallelism(config):
    """Resolve image/video parallelism settings into SGLang CLI args.

    `tp` is the tensor-parallel degree (--tp-size). The total device count
    (--num-gpus) is derived as tp_size * ulysses_degree * cfg_parallel_degree,
    so configs declare parallel degrees rather than a raw GPU count.
    `enable_cfg_parallel` is a boolean that contributes a factor of 2 when set.

    `tp` is left in `config` (it is a CSV column); `ulysses_degree` and
    `enable_cfg_parallel` are popped. Returns (num_gpus, parallel_args_str).
    """
    tp_size = int(config.get('tp', 1))
    ulysses_degree = config.pop('ulysses_degree', None)
    enable_cfg_parallel = config.pop('enable_cfg_parallel', None)

    ulysses_factor = int(ulysses_degree) if ulysses_degree is not None else 1
    cfg_factor = 2 if enable_cfg_parallel else 1
    num_gpus = tp_size * ulysses_factor * cfg_factor

    MAX_GPUS = 8
    if num_gpus > MAX_GPUS:
        raise ValueError(
            f"Requested num_gpus={num_gpus} exceeds the maximum of {MAX_GPUS} "
            f"(tp_size={tp_size} * ulysses_degree={ulysses_factor} * cfg={cfg_factor}). "
            f"Reduce tp, ulysses_degree, or disable enable_cfg_parallel."
        )

    parts = [f"--num-gpus {num_gpus}", f"--tp-size {tp_size}"]
    if ulysses_degree is not None:
        parts.append(f"--ulysses-degree {ulysses_degree}")
    if enable_cfg_parallel:
        parts.append("--enable-cfg-parallel true")
    return num_gpus, " ".join(parts)


def run_latency(model, config):
    output_json = (
        f"{config['model']}_latency_{config['tp']}_{config['inp']}_{config['out']}_{config['bs']}.json"
    )
    cmd = (
        "python -m sglang.bench_one_batch "
        f"--model-path {model} "
        f"--dtype {config['dtype']} "
        + (f"--quantization {config['quantization']} " if config.get('quantization') else "")
        + (f"--kv-cache-dtype {config['kv_cache_dtype']} " if config.get('kv_cache_dtype') else "")
        + f"--tensor-parallel-size {config['tp']} "
        f"--batch-size {config['bs']} "
        f"--input-len {config['inp']} "
        f"--output-len {config['out']} "
        f"--result-filename {output_json} "
    )
    # pop env and extra args from config
    env = config.pop('env', "")
    extra_args = config.pop('extra_args', "")
    cmd = f"{env} {cmd} {extra_args}".strip()
    config["cmd"] = cmd
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)

    # Parse output jsonl (bench_one_batch writes one JSON object per line)
    results = []
    with open(output_json, "r", newline="", encoding="utf-8") as f:
        output = json.loads(f.readline())
        if "total_latency" in output:
            result = {
                "performance": str(output["total_latency"] * 1000),
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
        "python -m sglang.bench_offline_throughput "
        f"--model-path {model} "
        f"--dtype {config['dtype']} "
        + (f"--quantization {config['quantization']} " if config.get('quantization') else "")
        + (f"--kv-cache-dtype {config['kv_cache_dtype']} " if config.get('kv_cache_dtype') else "")
        + f"--tensor-parallel-size {config['tp']} "
        f"--dataset-name random "
        f"--num-prompts {config['num_prompts']} "
        f"--random-input-len {config['inp']} "
        f"--random-output-len {config['out']} "
        f"--result-filename {output_json} "
    )
    # pop env and extra args from config
    env = config.pop('env', "")
    extra_args = config.pop('extra_args', "")
    cmd = f"{env} {cmd} {extra_args}".strip()
    config["cmd"] = cmd
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)

    # Parse output jsonl (bench_offline_throughput writes one JSON object per line)
    results = []
    with open(output_json, "r", newline="", encoding="utf-8") as f:
        output = json.loads(f.readline())
        if "total_throughput" in output:
            metrics = {
                "throughput_tot": str(output["total_throughput"]),
                "throughput_gen": str(output["output_throughput"]),
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
    variant_suffix = f"_{config['variant']}" if config.get('variant') else ""
    output_json = (
        f"{config['model']}_serving{variant_suffix}_{config['tp']}_{config['inp']}_{config['out']}_{config['num_prompts']}_{config['max_concurrency']}.json"
    )
    server_cmd = (
        "python -m sglang.launch_server "
        f"--model-path {model} "
        + resolve_tokenizer_arg(model)
        + f"--dtype {config['dtype']} "
        + (f"--quantization {config['quantization']} " if config.get('quantization') else "")
        + (f"--kv-cache-dtype {config['kv_cache_dtype']} " if config.get('kv_cache_dtype') else "")
        + f"--tensor-parallel-size {config['tp']} "
        f"--host 0.0.0.0 "
        f"--port 8000 "
        f"--disable-radix-cache "
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
        # wait for server to start; timeout via SGLANG_SERVER_START_TIMEOUT (default 90 minutes)
        if not wait_for_server(server, "http://localhost:8000/health",
                               timeout=SERVER_START_TIMEOUT, fail_on_http_error=True):
            print("Server failed to start")
            return [config]
        print(f"Server at {server.pid} contacted successfully", flush=True)

        # run benchmark
        bench_cmd = (
            "python -m sglang.bench_serving "
            f"--backend sglang "
            f"--base-url http://localhost:8000 "
            f"--model {model} "
            f"--dataset-name random "
            f"--request-rate inf "
            f"--random-range-ratio {config.get('random_range_ratio', 0.8)} "
            f"--max-concurrency {config['max_concurrency']} "
            f"--num-prompts {config['num_prompts']} "
            f"--random-input-len {config['inp']} "
            f"--random-output-len {config['out']} "
            f"--output-file {output_json}"
        )
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

        # parse output jsonl (bench_serving appends one JSON object per run)
        results = []
        output = read_last_json_line(output_json)
        if "total_throughput" in output:
            metrics = {
                "throughput_tot": str(output["total_throughput"]),
                "throughput_gen": str(output["output_throughput"]),
                "median_tpot": str(output["median_tpot_ms"]),
                "median_itl": str(output["median_itl_ms"]),
                "median_e2el": str(output["median_e2e_latency_ms"]),
                "median_ttft": str(output["median_ttft_ms"]),
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

        return results

    finally:
        # kill server and children
        parent = psutil.Process(server.pid)
        for child in parent.children(recursive=True):
            child.send_signal(signal.SIGKILL)
        server.send_signal(signal.SIGKILL)
        _ = server.communicate()
        del server

def run_serving_profile(model, config):
    """Single-pass profiling for the serving benchmark (Component 5, kernel breakdown).

    Runs `sglang.launch_server` under rocm-trace-lite (`rtl trace --mode full`) with roctx
    prefill/decode/mixed phase markers (scripts/sglang/profiling/), producing rtl_trace.db (GPU
    kernels + per-pass roctx markers), then aggregates it into ./profile/kernel_summary_payload.json
    for the client-perf-hub uploader. Mirrors the vLLM producer's run_serving_profile.
    """
    if not config.get("num_prompts"):
        config["num_prompts"] = str(min(5, 10 * int(config["max_concurrency"])))
    profile_dir = config.get("profile_dir", f"profile_{config['model']}_{config['tp']}")
    os.makedirs(profile_dir, exist_ok=True)
    rtl_trace_path = os.path.join(profile_dir, "rtl_trace.db")

    server_cmd = (
        "python -m sglang.launch_server "
        f"--model-path {model} "
        f"--dtype {config['dtype']} "
        + (f"--quantization {config['quantization']} " if config.get('quantization') else "")
        + (f"--kv-cache-dtype {config['kv_cache_dtype']} " if config.get('kv_cache_dtype') else "")
        + f"--tensor-parallel-size {config['tp']} "
        f"--host 0.0.0.0 "
        f"--port 8000 "
        f"--disable-radix-cache "
    )
    env = config.pop('env', "")
    extra_args = config.pop('extra_args', "")

    # roctx phase markers: PYTHONPATH puts sitecustomize.py on the path so the patch reaches SGLang's
    # spawned scheduler/TP workers; SGLANG_ROCTX_PHASE_MARKERS arms it. Both are inherited by the
    # `sglang.launch_server` that `rtl trace` launches. See scripts/sglang/profiling/.
    profiling_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiling")
    roctx_env = f"PYTHONPATH={profiling_dir}${{PYTHONPATH:+:$PYTHONPATH}} SGLANG_ROCTX_PHASE_MARKERS=1"
    rtl_env = f"{roctx_env} rtl trace --mode full -o {rtl_trace_path}"
    server_cmd = f"{env} {rtl_env} {server_cmd} {extra_args}".strip()
    config["cmd"] = server_cmd

    print("=" * 60)
    print("  PROFILE: rocm-trace-lite (--mode full) + roctx phase markers")
    print("=" * 60)
    print(server_cmd)
    server = subprocess.Popen(server_cmd, shell=True)
    try:
        if not wait_for_server(server, "http://localhost:8000/v1/models", timeout=5400):
            print("Server failed to start for profiling run")
            return [config]
        print(f"Server at {server.pid} ready for profiling", flush=True)

        bench_cmd = (
            "python -m sglang.bench_serving "
            f"--backend sglang "
            f"--base-url http://localhost:8000 "
            f"--model {model} "
            f"--dataset-name random "
            f"--request-rate inf "
            f"--random-range-ratio {config.get('random_range_ratio', 0.8)} "
            f"--max-concurrency {config['max_concurrency']} "
            f"--num-prompts {config['num_prompts']} "
            f"--random-input-len {config['inp']} "
            f"--random-output-len {config['out']}"
        )
        print(bench_cmd)
        subprocess.run(bench_cmd, shell=True, check=True)
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

    # Aggregate rtl trace -> client-perf-hub payload under ./profile/ (matches uploader glob).
    os.makedirs("profile", exist_ok=True)
    payload_path = os.path.join("profile", "kernel_summary_payload.json")
    # Shared, framework-agnostic curated aggregator (single source of truth = the vLLM
    # producer, which includes #380 pass-curation). The whole MAD-private repo is present
    # at runtime (madengine MODEL_DIR), so reference it by repo-relative path rather than
    # keeping a per-engine copy (which drifts and misses curation).
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
    """Run accuracy evaluation using sglang-specific accuracy framework.

    Delegates to accuracy_runner_sglang which handles sglang-specific server
    management and evaluation logic.
    """
    def _kill_sglang_server(server):
        """Kill sglang server and children."""
        parent = psutil.Process(server.pid)
        for child in parent.children(recursive=True):
            child.send_signal(signal.SIGKILL)
        server.send_signal(signal.SIGKILL)
        _ = server.communicate()
        del server

    return run_accuracy_sglang(model, config, wait_for_server, _kill_sglang_server)



def run_video_serving(model, config, hf_pipeline_tag):
    port = 30000
    num_gpus, parallel_args = resolve_parallelism(config)
    height = config.pop('height', None)
    width = config.pop('width', None)
    num_frames = config.pop('num_frames', None)
    num_inference_steps = config.pop('num_inference_steps', None)
    config.pop('guidance_scale', None)
    fps = config.pop('fps', None)
    save_output = bool(config.pop('save_output', False))
    max_concurrency = config.pop('concurrency', None) or config.pop('max_concurrency', '1')
    if not config.get("num_prompts"):
        config["num_prompts"] = str(2 * int(max_concurrency))
    num_prompts = config["num_prompts"]
    task = hf_pipeline_tag
    dataset = config.pop('dataset', 'vbench')

    env = config.pop('env', "")
    extra_args = config.pop('extra_args', "")
    extra_args_server = re.sub(r'--save-output\s*(true|false)?\s*', '', extra_args).strip()

    output_json = (
        f"{config['model']}_video_gen_{hf_pipeline_tag}_{num_frames or 'default'}frames"
        f"_{num_prompts}_{max_concurrency}.json"
    )

    if save_output:
        output_path_arg = f'--output-path {config["model"]}_video_gen_{hf_pipeline_tag}_outputs'
    else:
        output_path_arg = '--output-path ""'

    server_cmd = (
        f"sglang serve "
        f"--model-path {model} "
        f"{parallel_args} "
        f"--host 0.0.0.0 "
        f"--port {port} "
        f"{output_path_arg} "
        f"{extra_args_server}"
    ).strip()
    server_cmd = f"{env} {server_cmd}".strip()
    config["cmd"] = server_cmd

    print(f"Starting video generation server: {server_cmd}")
    server = subprocess.Popen(server_cmd, shell=True)

    try:
        if not wait_for_server(server, f"http://localhost:{port}/health"):
            print("Video generation server failed to start")
            return [{"hf_pipeline_tag": hf_pipeline_tag, "height": height or "", "width": width or "",
                     "num_inference_steps": num_inference_steps or "",
                     "num_frames": num_frames or "", **config}]
        print(f"Server at {server.pid} contacted successfully", flush=True)

        bench_cmd = (
            f"python -m sglang.multimodal_gen.benchmarks.bench_serving "
            f"--port {port} "
            f"--host localhost "
            f"--model {model} "
            f"--dataset {dataset} "
            f"--num-prompts {num_prompts} "
            f"--max-concurrency {max_concurrency} "
            f"--request-rate inf "
            f"--task {task} "
            f"--warmup-requests 5 "
            f"--output-file {output_json}"
        )
        if height is not None:
            bench_cmd += f" --height {height}"
        if width is not None:
            bench_cmd += f" --width {width}"
        if num_frames is not None:
            bench_cmd += f" --num-frames {num_frames}"
        if num_inference_steps is not None:
            bench_cmd += f" --num-inference-steps {num_inference_steps}"
        if fps is not None:
            bench_cmd += f" --fps {fps}"

        bench_args = config.pop('bench_args', {})
        bench_args_str = ""
        for k, v in bench_args.items():
            if isinstance(v, bool):
                bench_args_str += f"{k} "
            else:
                bench_args_str += f"{k} {v} "
        bench_cmd = f"{bench_cmd} {bench_args_str}".strip()

        config["cmd"] = f"{server_cmd};{bench_cmd}"
        print(f"Running benchmark [{hf_pipeline_tag}]: {bench_cmd}")
        subprocess.run(bench_cmd, shell=True, check=True)

        with open(output_json, "r", encoding="utf-8") as f:
            output = json.load(f)

        results = []
        common_fields = {
            "hf_pipeline_tag": hf_pipeline_tag, "height": height or "", "width": width or "",
            "num_inference_steps": num_inference_steps or "",
            "concurrency": max_concurrency or "",
            "num_frames": num_frames or "", **config
        }
        metrics_map = {
            "latency_median": ("latency_median", "s"),
            "latency_mean": ("latency_mean", "s"),
            "throughput_qps": ("throughput_qps", "req/s"),
            "output_throughput_ops": ("output_throughput_ops", "outputs/s"),
        }
        for metric_name, (json_key, unit) in metrics_map.items():
            value = output.get(json_key, 0)
            results.append({
                "performance": str(value), "metric": metric_name, "unit": unit,
                **common_fields
            })

        return results

    finally:
        parent = psutil.Process(server.pid)
        for child in parent.children(recursive=True):
            child.send_signal(signal.SIGKILL)
        server.send_signal(signal.SIGKILL)
        _ = server.communicate()
        del server


def run_image_serving(model, config, hf_pipeline_tag):
    port = 30000
    num_gpus, parallel_args = resolve_parallelism(config)
    num_images = int(config.get('num_images', 1))
    height = config.pop('height', 1024)
    width = config.pop('width', 1024)
    num_inference_steps = config.pop('num_inference_steps', None)
    config.pop('guidance_scale', None)
    save_output = bool(config.pop('save_output', True))
    perf_dump_path = config.pop('perf_dump_path', None)
    max_concurrency = config.pop('concurrency', None) or config.pop('max_concurrency', '1')
    if not config.get("num_prompts"):
        config["num_prompts"] = str(2 * int(max_concurrency))
    num_prompts = config["num_prompts"]
    task = hf_pipeline_tag
    dataset = config.pop('dataset', 'vbench')

    env = config.pop('env', "")
    extra_args = config.pop('extra_args', "")
    extra_args_server = re.sub(r'--save-output\s*(true|false)?\s*', '', extra_args).strip()

    output_json = f"{config['model']}_image_gen_{hf_pipeline_tag}_{height}x{width}_{num_prompts}_{max_concurrency}.json"

    if save_output:
        output_path_arg = f'--output-path {config["model"]}_image_gen_{hf_pipeline_tag}_{height}x{width}_outputs'
    else:
        output_path_arg = '--output-path ""'

    server_cmd = (
        f"sglang serve "
        f"--model-path {model} "
        f"{parallel_args} "
        f"--host 0.0.0.0 "
        f"--port {port} "
        f"{output_path_arg} "
        f"{extra_args_server}"
    ).strip()
    server_cmd = f"{env} {server_cmd}".strip()
    config["cmd"] = server_cmd

    print(f"Starting image generation server: {server_cmd}")
    server = subprocess.Popen(server_cmd, shell=True)

    try:
        if not wait_for_server(server, f"http://localhost:{port}/health"):
            print("Image generation server failed to start")
            return [{"hf_pipeline_tag": hf_pipeline_tag, "height": height, "width": width,
                     "num_inference_steps": num_inference_steps or "",
                     "num_images": num_images, **config}]
        print(f"Server at {server.pid} contacted successfully", flush=True)

        bench_cmd = (
            f"python -m sglang.multimodal_gen.benchmarks.bench_serving "
            f"--port {port} "
            f"--host localhost "
            f"--model {model} "
            f"--dataset {dataset} "
            f"--num-prompts {num_prompts} "
            f"--max-concurrency {max_concurrency} "
            f"--request-rate inf "
            f"--task {task} "
            f"--width {width} "
            f"--height {height} "
            f"--num-outputs-per-prompt {num_images} "
            f"--warmup-requests 5 "
            f"--output-file {output_json}"
        )
        if num_inference_steps is not None:
            bench_cmd += f" --num-inference-steps {num_inference_steps}"

        bench_args = config.pop('bench_args', {})
        if perf_dump_path:
            bench_args["--perf-dump-path"] = perf_dump_path
        bench_args_str = ""
        for k, v in bench_args.items():
            if isinstance(v, bool):
                bench_args_str += f"{k} "
            else:
                bench_args_str += f"{k} {v} "
        bench_cmd = f"{bench_cmd} {bench_args_str}".strip()

        config["cmd"] = f"{server_cmd};{bench_cmd}"
        print(f"Running benchmark [{hf_pipeline_tag}]: {bench_cmd}")
        subprocess.run(bench_cmd, shell=True, check=True)

        with open(output_json, "r", encoding="utf-8") as f:
            output = json.load(f)

        results = []
        common_fields = {
            "hf_pipeline_tag": hf_pipeline_tag, "height": height, "width": width,
            "num_inference_steps": num_inference_steps or "",
            "concurrency": max_concurrency or "",
            "num_images": num_images, **config
        }
        metrics_map = {
            "latency_median": ("latency_median", "s"),
            "latency_mean": ("latency_mean", "s"),
            "throughput_qps": ("throughput_qps", "req/s"),
            "output_throughput_ops": ("output_throughput_ops", "outputs/s"),
        }
        for metric_name, (json_key, unit) in metrics_map.items():
            value = output.get(json_key, 0)
            results.append({
                "performance": str(value), "metric": metric_name, "unit": unit,
                **common_fields
            })

        return results

    finally:
        parent = psutil.Process(server.pid)
        for child in parent.children(recursive=True):
            child.send_signal(signal.SIGKILL)
        server.send_signal(signal.SIGKILL)
        _ = server.communicate()
        del server


def run_image_throughput(model, config, hf_pipeline_tag):
    num_gpus, parallel_args = resolve_parallelism(config)
    height = config.pop('height', 1024)
    width = config.pop('width', 1024)
    num_inference_steps = config.pop('num_inference_steps', None)
    guidance_scale = config.pop('guidance_scale', None)
    num_prompts = config.get('num_prompts', '10')
    dataset = config.pop('dataset', 'random')

    env = config.pop('env', "")
    extra_args = config.pop('extra_args', "")

    output_json = (
        f"{config['model']}_image_throughput_{hf_pipeline_tag}_{height}x{width}"
        f"_{num_prompts}.json"
    )

    cmd = (
        f"python -m sglang.multimodal_gen.benchmarks.bench_offline_throughput "
        f"--model-path {model} "
        f"{parallel_args} "
        f"--width {width} "
        f"--height {height} "
        f"--num-prompts {num_prompts} "
        f"--dataset {dataset} "
        f"--output-file {output_json}"
    )
    if num_inference_steps is not None:
        cmd += f" --num-inference-steps {num_inference_steps}"
    if guidance_scale is not None:
        cmd += f" --guidance-scale {guidance_scale}"

    cmd = f"{env} {cmd} {extra_args}".strip()
    config["cmd"] = cmd

    print(f"Running image throughput benchmark [{hf_pipeline_tag}]: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

    results = []
    common_fields = {
        "hf_pipeline_tag": hf_pipeline_tag, "height": height, "width": width,
        "num_inference_steps": num_inference_steps or "",
        "guidance_scale": guidance_scale or "", **config
    }

    with open(output_json, "r", encoding="utf-8") as f:
        for line in f:
            output = json.loads(line)
            results_data = output.get("results", output)
            metrics_map = {
                "requests_per_second": ("requests_per_second", "req/s"),
                "frames_per_second": ("frames_per_second", "frames/s"),
                "megapixels_per_second": ("megapixels_per_second", "MP/s"),
            }
            for metric_name, (json_key, unit) in metrics_map.items():
                value = results_data.get(json_key, 0)
                results.append({
                    "performance": str(value), "metric": metric_name, "unit": unit,
                    **common_fields
                })
            break

    return results


def run_video_throughput(model, config, hf_pipeline_tag):
    num_gpus, parallel_args = resolve_parallelism(config)
    height = config.pop('height', None)
    width = config.pop('width', None)
    num_frames = config.pop('num_frames', None)
    num_inference_steps = config.pop('num_inference_steps', None)
    guidance_scale = config.pop('guidance_scale', None)
    fps = config.pop('fps', None)
    concurrency = config.pop('concurrency', None)
    num_prompts = config.get('num_prompts', '10')
    dataset = config.pop('dataset', 'random')

    env = config.pop('env', "")
    extra_args = config.pop('extra_args', "")

    output_json = (
        f"{config['model']}_video_throughput_{hf_pipeline_tag}_{num_frames or 'default'}frames"
        f"_{num_prompts}.json"
    )

    cmd = (
        f"python -m sglang.multimodal_gen.benchmarks.bench_offline_throughput "
        f"--model-path {model} "
        f"{parallel_args} "
        f"--num-prompts {num_prompts} "
        f"--dataset {dataset} "
        f"--output-file {output_json}"
    )
    if height is not None:
        cmd += f" --height {height}"
    if width is not None:
        cmd += f" --width {width}"
    if num_frames is not None:
        cmd += f" --num-frames {num_frames}"
    if num_inference_steps is not None:
        cmd += f" --num-inference-steps {num_inference_steps}"
    if guidance_scale is not None:
        cmd += f" --guidance-scale {guidance_scale}"
    if fps is not None:
        cmd += f" --fps {fps}"
    if concurrency is not None:
        cmd += f" --concurrency {concurrency}"

    cmd = f"{env} {cmd} {extra_args}".strip()
    config["cmd"] = cmd

    print(f"Running video throughput benchmark [{hf_pipeline_tag}]: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

    results = []
    common_fields = {
        "hf_pipeline_tag": hf_pipeline_tag, "height": height or "", "width": width or "",
        "num_inference_steps": num_inference_steps or "",
        "guidance_scale": guidance_scale or "",
        "concurrency": concurrency or "",
        "num_frames": num_frames or "", **config
    }

    with open(output_json, "r", encoding="utf-8") as f:
        for line in f:
            output = json.loads(line)
            results_data = output.get("results", output)
            metrics_map = {
                "requests_per_second": ("requests_per_second", "req/s"),
                "frames_per_second": ("frames_per_second", "frames/s"),
                "megapixels_per_second": ("megapixels_per_second", "MP/s"),
            }
            for metric_name, (json_key, unit) in metrics_map.items():
                value = results_data.get(json_key, 0)
                results.append({
                    "performance": str(value), "metric": metric_name, "unit": unit,
                    **common_fields
                })
            break

    return results


def run_image_profile(model, config, hf_pipeline_tag, profile):
    num_gpus, parallel_args = resolve_parallelism(config)
    height = config.pop('height', 1024)
    width = config.pop('width', 1024)
    num_inference_steps = config.pop('num_inference_steps', None)
    guidance_scale = config.pop('guidance_scale', None)
    config.pop('save_output', None)
    config.pop('dataset', None)
    config.pop('concurrency', None)
    config.pop('num_prompts', None)

    profile_opts = profile if isinstance(profile, dict) else {}
    profile_all_stages = profile_opts.get('profile_all_stages', True)
    num_profiled_timesteps = profile_opts.get('num_profiled_timesteps', 5)
    warmup_steps = profile_opts.get('warmup_steps', 5)
    profile_dir = profile_opts.get('output_dir', './profile_output/')
    perf_dump_path = profile_opts.get('perf_dump_path',
        f"{config['model']}_profile_{hf_pipeline_tag}_{height}x{width}_perf.json")

    env = config.pop('env', "")
    extra_args = config.pop('extra_args', "")

    cmd = (
        f"sglang generate "
        f"--model-path {model} "
        f"{parallel_args} "
        f"--width {width} "
        f"--height {height} "
        f"--warmup "
        f"--warmup-steps {warmup_steps} "
        f"--no-save-output "
        f"--perf-dump-path {perf_dump_path} "
        f"--prompt \"a professional photograph of a sunset over mountains\""
    )
    if profile:
        cmd += f" --profile --num-profiled-timesteps {num_profiled_timesteps}"
        if profile_all_stages:
            cmd += " --profile-all-stages"
    if num_inference_steps is not None:
        cmd += f" --num-inference-steps {num_inference_steps}"
    if guidance_scale is not None:
        cmd += f" --guidance-scale {guidance_scale}"

    profiler_env = f"SGLANG_DIFFUSION_TORCH_PROFILER_DIR={profile_dir} " if profile else ""
    cmd = f"{profiler_env}{env} {cmd} {extra_args}".strip()
    config["cmd"] = cmd

    mode = "profile" if profile else "latency"
    print(f"Running image {mode} [{hf_pipeline_tag}]: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

    results = []
    common_fields = {
        "hf_pipeline_tag": hf_pipeline_tag, "height": height, "width": width,
        "num_inference_steps": num_inference_steps or "", **config
    }

    if os.path.exists(perf_dump_path):
        with open(perf_dump_path, "r", encoding="utf-8") as f:
            output = json.load(f)
        total_ms = output.get("total_duration_ms", 0)
        results.append({
            "performance": str(total_ms), "metric": "total_time",
            "unit": "ms", **common_fields
        })
        denoise_ms = sum(
            step.get("duration_ms", 0)
            for step in output.get("denoise_steps_ms", [])
        )
        if denoise_ms:
            results.append({
                "performance": str(denoise_ms), "metric": "denoise_time",
                "unit": "ms", **common_fields
            })

    if profile:
        print(f"Profile traces saved to {profile_dir}")
    return results


def run_image_latency(model, config, hf_pipeline_tag, profile):
    num_gpus, parallel_args = resolve_parallelism(config)
    height = config.pop('height', 1024)
    width = config.pop('width', 1024)
    num_inference_steps = config.pop('num_inference_steps', None)
    guidance_scale = config.pop('guidance_scale', None)
    config.pop('save_output', None)
    config.pop('dataset', None)
    config.pop('concurrency', None)
    config.pop('num_prompts', None)

    profile_opts = profile if isinstance(profile, dict) else {}
    perf_dump_path = profile_opts.get('perf_dump_path',
        f"{config['model']}_latency_{hf_pipeline_tag}_{height}x{width}_perf.json")

    env = config.pop('env', "")
    extra_args = config.pop('extra_args', "")

    cmd = (
        f"sglang generate "
        f"--model-path {model} "
        f"{parallel_args} "
        f"--width {width} "
        f"--height {height} "
        f"--warmup "
        f"--no-save-output "
        f"--perf-dump-path {perf_dump_path} "
        f"--prompt \"a professional photograph of a sunset over mountains\""
    )
    if num_inference_steps is not None:
        cmd += f" --num-inference-steps {num_inference_steps}"
    if guidance_scale is not None:
        cmd += f" --guidance-scale {guidance_scale}"

    cmd = f"{env} {cmd} {extra_args}".strip()
    config["cmd"] = cmd

    print(f"Running image latency [{hf_pipeline_tag}]: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

    results = []
    common_fields = {
        "hf_pipeline_tag": hf_pipeline_tag, "height": height, "width": width,
        "num_inference_steps": num_inference_steps or "", **config
    }

    if os.path.exists(perf_dump_path):
        with open(perf_dump_path, "r", encoding="utf-8") as f:
            output = json.load(f)
        total_ms = output.get("total_duration_ms", 0)
        results.append({
            "performance": str(total_ms), "metric": "total_time",
            "unit": "ms", **common_fields
        })
        denoise_ms = sum(
            step.get("duration_ms", 0)
            for step in output.get("denoise_steps_ms", [])
        )
        if denoise_ms:
            results.append({
                "performance": str(denoise_ms), "metric": "denoise_time",
                "unit": "ms", **common_fields
            })

    return results


def run_video_profile(model, config, hf_pipeline_tag, profile):
    num_gpus, parallel_args = resolve_parallelism(config)
    height = config.pop('height', None)
    width = config.pop('width', None)
    num_frames = config.pop('num_frames', None)
    num_inference_steps = config.pop('num_inference_steps', None)
    guidance_scale = config.pop('guidance_scale', None)
    fps = config.pop('fps', None)
    config.pop('save_output', None)
    config.pop('dataset', None)
    config.pop('concurrency', None)
    config.pop('num_prompts', None)

    profile_opts = profile if isinstance(profile, dict) else {}
    profile_all_stages = profile_opts.get('profile_all_stages', True)
    num_profiled_timesteps = profile_opts.get('num_profiled_timesteps', 5)
    warmup_steps = profile_opts.get('warmup_steps', 5)
    profile_dir = profile_opts.get('output_dir', './profile_output/')
    perf_dump_path = profile_opts.get('perf_dump_path',
        f"{config['model']}_profile_{hf_pipeline_tag}_{num_frames or 'default'}frames_perf.json")

    env = config.pop('env', "")
    extra_args = config.pop('extra_args', "")

    cmd = (
        f"sglang generate "
        f"--model-path {model} "
        f"{parallel_args} "
        f"--warmup "
        f"--warmup-steps {warmup_steps} "
        f"--no-save-output "
        f"--perf-dump-path {perf_dump_path} "
        f"--prompt \"a professional photograph of a sunset over mountains\""
    )
    if profile:
        cmd += f" --profile --num-profiled-timesteps {num_profiled_timesteps}"
        if profile_all_stages:
            cmd += " --profile-all-stages"
    if height is not None:
        cmd += f" --height {height}"
    if width is not None:
        cmd += f" --width {width}"
    if num_frames is not None:
        cmd += f" --num-frames {num_frames}"
    if num_inference_steps is not None:
        cmd += f" --num-inference-steps {num_inference_steps}"
    if guidance_scale is not None:
        cmd += f" --guidance-scale {guidance_scale}"
    if fps is not None:
        cmd += f" --fps {fps}"

    profiler_env = f"SGLANG_DIFFUSION_TORCH_PROFILER_DIR={profile_dir} " if profile else ""
    cmd = f"{profiler_env}{env} {cmd} {extra_args}".strip()
    config["cmd"] = cmd

    mode = "profile" if profile else "latency"
    print(f"Running video {mode} [{hf_pipeline_tag}]: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

    results = []
    common_fields = {
        "hf_pipeline_tag": hf_pipeline_tag, "height": height or "", "width": width or "",
        "num_inference_steps": num_inference_steps or "",
        "num_frames": num_frames or "", **config
    }

    if os.path.exists(perf_dump_path):
        with open(perf_dump_path, "r", encoding="utf-8") as f:
            output = json.load(f)
        total_ms = output.get("total_duration_ms", 0)
        results.append({
            "performance": str(total_ms), "metric": "total_time",
            "unit": "ms", **common_fields
        })
        denoise_ms = sum(
            step.get("duration_ms", 0)
            for step in output.get("denoise_steps_ms", [])
        )
        if denoise_ms:
            results.append({
                "performance": str(denoise_ms), "metric": "denoise_time",
                "unit": "ms", **common_fields
            })

    if profile:
        print(f"Profile traces saved to {profile_dir}")
    return results


def run_video_latency(model, config, hf_pipeline_tag):
    num_gpus, parallel_args = resolve_parallelism(config)
    height = config.pop('height', None)
    width = config.pop('width', None)
    num_frames = config.pop('num_frames', None)
    num_inference_steps = config.pop('num_inference_steps', None)
    guidance_scale = config.pop('guidance_scale', None)
    fps = config.pop('fps', None)
    concurrency = config.pop('concurrency', None)
    num_prompts = config.get('num_prompts', '3')
    dataset = config.pop('dataset', 'random')

    env = config.pop('env', "")
    extra_args = config.pop('extra_args', "")

    output_json = (
        f"{config['model']}_video_latency_{hf_pipeline_tag}_{num_frames or 'default'}frames"
        f"_{num_prompts}.json"
    )

    cmd = (
        f"python -m sglang.multimodal_gen.benchmarks.bench_offline_throughput "
        f"--model-path {model} "
        f"{parallel_args} "
        f"--num-prompts {num_prompts} "
        f"--dataset {dataset} "
        f"--output-file {output_json}"
    )
    if height is not None:
        cmd += f" --height {height}"
    if width is not None:
        cmd += f" --width {width}"
    if num_frames is not None:
        cmd += f" --num-frames {num_frames}"
    if num_inference_steps is not None:
        cmd += f" --num-inference-steps {num_inference_steps}"
    if guidance_scale is not None:
        cmd += f" --guidance-scale {guidance_scale}"
    if fps is not None:
        cmd += f" --fps {fps}"
    if concurrency is not None:
        cmd += f" --concurrency {concurrency}"

    cmd = f"{env} {cmd} {extra_args}".strip()
    config["cmd"] = cmd

    print(f"Running video latency benchmark [{hf_pipeline_tag}]: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

    results = []
    common_fields = {
        "hf_pipeline_tag": hf_pipeline_tag, "height": height or "", "width": width or "",
        "num_inference_steps": num_inference_steps or "",
        "guidance_scale": guidance_scale or "",
        "concurrency": concurrency or "",
        "num_frames": num_frames or "", **config
    }

    with open(output_json, "r", encoding="utf-8") as f:
        for line in f:
            output = json.loads(line)
            results_data = output.get("results", output)
            value = results_data.get("latency_per_request_seconds", 0)
            results.append({
                "performance": str(value), "metric": "latency_per_request",
                "unit": "s", **common_fields
            })
            break

    return results


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
    # For sglang-accuracy tag, always merge (tag implies accuracy, benchmark field irrelevant)
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
            matched_any = False
            for cfg in configs:
                pipeline_tag = cfg.get('hf_pipeline_tag', 'text-generation')

                # Level 1: Apply task-level defaults
                if pipeline_tag in task_defaults:
                    task_entry = task_defaults[pipeline_tag]
                    if 'evaluations' not in cfg and 'evaluations' in task_entry:
                        cfg['evaluations'] = task_entry['evaluations']

                # Level 2: Apply model-specific overrides
                base_tokens = _tokens(cfg.get('model'))
                matched = next((p for p in model_entries if p['models'] & base_tokens), None)
                if matched:
                    matched_any = True
                    base_extra = cfg.get('extra_args') or {}
                    base_env = cfg.get('env') or {}
                    cfg['extra_args'] = _fill_missing(base_extra, matched['extra_args'])
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

                    print(
                        f"  Applied acc overrides for model={cfg.get('model')} "
                        f"<- {sorted(matched['models'])}"
                    )

            if matched_any:
                with open(acc_output, 'w') as f:
                    yaml.safe_dump(configs, f, sort_keys=False, default_flow_style=False)
                print(
                    f"Wrote merged accuracy config to {acc_output} "
                    f"(original {args.config} left untouched)"
                )
            else:
                print(
                    f"No acc entries matched any model in {args.config}; "
                    f"not writing {acc_output}"
                )
        else:
            print(f"No acc.yaml found at {acc_path} (or empty); skipping accuracy merge")

    # Expand and filter configs
    configs = expand_configs(args, configs)

    # Validate hf_pipeline_tag values
    for i, cfg in enumerate(configs):
        tag = cfg.get("hf_pipeline_tag", "text-generation")
        if tag not in SUPPORTED_HF_PIPELINE_TAGS:
            raise ValueError(
                f"Config entry {i} has unsupported hf_pipeline_tag '{tag}'. "
                f"Supported tags: {sorted(SUPPORTED_HF_PIPELINE_TAGS)}"
            )

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
            fieldnames = CSV_HEADER
            if config.get("variant"):
                # variants share a model (e.g. K3 nospec / dspark), so tag their rows
                fieldnames = CSV_HEADER[:2] + ["variant"] + CSV_HEADER[2:]
            writer = csv.DictWriter(outf, delimiter=",", fieldnames=fieldnames, extrasaction="ignore")
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
                _repos = [model]
                # A speculative-decoding config needs its draft checkpoint too
                _draft = config.get("extra_args", {}).get("--speculative-draft-model-path")
                if _draft and not os.path.isabs(str(_draft)):
                    _repos.append(str(_draft))
                _dl_script = (
                    "from huggingface_hub import snapshot_download\n"
                    f"for _repo in {_repos!r}:\n"
                    f"    snapshot_download(_repo, ignore_patterns={_ignore!r}, max_workers=16)"
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
            
            # Keep dict forms for potential acc.yaml merge (for benchmark: all)
            env_vars = config.get("env", {})
            extra_args = config.get("extra_args", {})

            config["env_dict"] = env_vars  # Preserve dict for acc.yaml merge
            config["extra_args_dict"] = extra_args  # Preserve dict for acc.yaml merge

            # profiling support: inject torch profiler env var
            profile = config.pop('profile', False) or args.profile
            if profile:
                env_vars["SGLANG_DIFFUSION_TORCH_PROFILER_DIR"] = "./profile_output/"
            env_vars_str = " ".join(FIXED_ENV_VARS + [f"{k}={v}" for k, v in env_vars.items()])
            benchmark = config["benchmark"]
            hf_pipeline_tag = config.get("hf_pipeline_tag", "text-generation")
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
            if hf_pipeline_tag in ("text-to-image", "image-to-image"):
                if profile:
                    results = run_image_profile(model, config, hf_pipeline_tag, profile=profile)
                elif benchmark == "all":
                    print(f"\n=== Running serving benchmark for {model} ===")
                    serving_config = dict(config)
                    serving_config["benchmark"] = "serving"
                    serving_results = run_image_serving(model, serving_config, hf_pipeline_tag)
                    for result in serving_results:
                        writer.writerow(result)

                    print(f"\n=== Running accuracy benchmark for {model} ===")
                    acc_config = _build_acc_config(model, config, args)
                    results = run_accuracy(model, acc_config)
                elif benchmark == "latency":
                    results = run_image_latency(model, config, hf_pipeline_tag, profile=profile)
                elif benchmark == "throughput":
                    results = run_image_throughput(model, config, hf_pipeline_tag)
                elif benchmark == "accuracy":
                    results = run_accuracy(model, config)
                else:
                    results = run_image_serving(model, config, hf_pipeline_tag)
            elif hf_pipeline_tag in ("text-to-video", "image-to-video"):
                if profile:
                    results = run_video_profile(model, config, hf_pipeline_tag, profile=profile)
                elif benchmark == "all":
                    print(f"\n=== Running serving benchmark for {model} ===")
                    serving_config = dict(config)
                    serving_config["benchmark"] = "serving"
                    serving_results = run_video_serving(model, serving_config, hf_pipeline_tag)
                    for result in serving_results:
                        writer.writerow(result)
                    print(f"\n=== Running accuracy benchmark for {model} ===")
                    acc_config = _build_acc_config(model, config, args)
                    results = run_accuracy(model, acc_config)
                elif benchmark == "throughput":
                    results = run_video_throughput(model, config, hf_pipeline_tag)
                elif benchmark == "latency":
                    results = run_video_latency(model, config, hf_pipeline_tag)
                elif benchmark == "accuracy":
                    results = run_accuracy(model, config)
                else:
                    results = run_video_serving(model, config, hf_pipeline_tag)
            elif hf_pipeline_tag == "text-generation":
                if benchmark == "all":
                    print(f"\n=== Running serving benchmark for {model} ===")
                    serving_config = dict(config)
                    serving_config["benchmark"] = "serving"
                    if profile:
                        serving_results = run_serving_profile(model, serving_config)
                    else:
                        serving_results = run_serving(model, serving_config)

                    for result in serving_results:
                        writer.writerow(result)

                    print(f"\n=== Running accuracy benchmark for {model} ===")
                    acc_config = _build_acc_config(model, config, args)
                    results = run_accuracy(model, acc_config)
                elif profile and benchmark == "serving":
                    results = run_serving_profile(model, config)
                elif benchmark == "latency":
                    results = run_latency(model, config)
                elif benchmark == "throughput":
                    results = run_throughput(model, config)
                elif benchmark == "serving":
                    results = run_serving(model, config)
                elif benchmark == "accuracy":
                    # Note: acc.yaml merge already happened at line 1705 for benchmark:all mode
                    # Config already has merged evaluations from task defaults + model overrides
                    results = run_accuracy(model, config)
                else:
                    raise ValueError(f"Unknown benchmark: {benchmark}")
            else:
                raise ValueError(f"Unknown hf_pipeline_tag: {hf_pipeline_tag}")

            # Write results to csv
            for result in results:
                writer.writerow(result)

            # Fail the run if any accuracy metric failed
            if benchmark in ("accuracy", "all"):
                failed = [r for r in results if r.get("performance") == "fail"]
                if failed:
                    metrics = [f"{r.get('metric', 'unknown')}({r.get('unit', '')})" for r in failed]
                    print(f"\nACCURACY FAILED: {', '.join(metrics)}")
                    sys.exit(1)


if __name__ == "__main__":
    main()
