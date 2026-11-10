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

"""
Atom-specific accuracy evaluation runner.

Supports multiple evaluation frameworks based on acc.yaml schema:
- lm_eval: Language model evaluation (default for text models)
- compute_clip_score: CLIP score for image/video models (future)
- vbench: Video generation benchmark (future)

Schema:
- Each acc.yaml entry contains: model (space-separated), evaluations list
- Each evaluation contains: framework field
- lm_eval supports: task, num_fewshot, apply_chat_template, base_url,
  lm_eval_model, num_concurrent, max_gen_toks, limit
- Default framework based on hf_pipeline_tag:
  text -> lm_eval, text-image -> compute_clip_score, text-video -> vbench
"""

import os
import glob
import json
import shutil
import subprocess
from typing import Dict, List, Any, Callable


def run_lm_eval(model: str, config: Dict[str, Any], eval_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run lm_eval framework evaluation.

    Args:
        model: Model path or name
        config: Main configuration dict
        eval_config: Evaluation-specific configuration

    Returns:
        List of result dicts containing performance metrics
    """
    # Extract lm_eval specific parameters
    task = eval_config.get("task", "gsm8k")
    num_fewshot = eval_config.get("num_fewshot", 3)
    apply_chat_template = eval_config.get("apply_chat_template", False)
    trust_remote_code = eval_config.get("trust_remote_code", False)
    num_concurrent = eval_config.get("num_concurrent", 64)
    max_gen_toks = eval_config.get("max_gen_toks", 2048)
    limit = eval_config.get("limit")
    lm_eval_model = eval_config.get("lm_eval_model", "local-completions")
    base_url = eval_config.get("base_url", "http://localhost:8000/v1/completions")

    # Build model_args
    model_args = {
        "model": model,
        "max_gen_toks": max_gen_toks,
        "num_concurrent": num_concurrent,
        "max_retries": 3,
        "base_url": base_url,
    }

    # Add tokenized_requests if using local-completions
    if lm_eval_model == "local-completions":
        model_args["tokenized_requests"] = False

    model_args_str = ",".join(f"{k}={v}" for k, v in model_args.items())

    # Build lm_eval command (use shell=True like vLLM for simplicity)
    bench_cmd = (
        "lm_eval "
        f"--model {lm_eval_model} "
        f"--model_args {model_args_str} "
        f"--tasks {task} "
        f"--num_fewshot {num_fewshot} "
        f"--output_path ./tmp "
    )

    # Add optional parameters
    if limit is not None:
        bench_cmd += f"--limit {limit} "
    if apply_chat_template:
        bench_cmd += "--apply_chat_template "
    if trust_remote_code:
        bench_cmd += "--trust_remote_code "

    print(f"Running lm_eval command: {bench_cmd}", flush=True)

    # Append bench_cmd to config['cmd'] for reproducibility (server_cmd;bench_cmd pattern)
    if 'cmd' in config:
        config['cmd'] = f"{config['cmd']};{bench_cmd}"
    else:
        config['cmd'] = bench_cmd

    # Run lm_eval
    subprocess.run(bench_cmd, shell=True, check=True)

    # Find output file (lm_eval creates a subdirectory inside ./tmp)
    output_files = glob.glob("./tmp/*/*.json")
    print(f"DEBUG: Found {len(output_files)} result files in ./tmp/*/*.json", flush=True)
    if len(output_files) == 0:
        print("ERROR: No output files found after lm_eval completed", flush=True)
        return [{**config, "performance": -1, "metric": "error", "unit": "no_output_file"}]
    elif len(output_files) > 1:
        print(f"ERROR: Multiple output files found: {output_files}", flush=True)
        return [{**config, "performance": -1, "metric": "error", "unit": "multiple_output_files"}]

    # Move output file and parse
    # Replace / with _ in model name for file path
    model_name = config['model'].replace('/', '_')
    output_json = f"{model_name}_accuracy_{task}_{config['tp']}.json"
    output_file = output_files[0]
    shutil.move(output_file, output_json)
    shutil.rmtree("./tmp")

    # Parse results
    results = []
    try:
        with open(output_json, 'r') as f:
            lm_eval_results = json.load(f)

            # Extract key metrics
            task_results = lm_eval_results.get("results", {}).get(task, {})
            print(f"DEBUG: task_results keys: {list(task_results.keys())}", flush=True)

            # Extract all numeric metrics, excluding stderr fields
            # lm_eval includes standard error metrics with _stderr suffix (e.g., exact_match_stderr)
            for metric_key, metric_value in task_results.items():
                if isinstance(metric_value, (int, float)) and not metric_key.endswith('_stderr'):
                    results.append({
                        "performance": metric_value,
                        "metric": metric_key,
                        "unit": "score",
                        **config  # Spread all config fields
                    })

        print(f"DEBUG: Created {len(results)} result entries", flush=True)
        if results:
            print(f"DEBUG: First result keys: {list(results[0].keys())}", flush=True)
            print(f"DEBUG: First result performance: {results[0].get('performance')}", flush=True)


    except Exception as e:
        print(f"Error parsing lm_eval results: {e}", flush=True)
        import traceback
        traceback.print_exc()
        # Clean up on error
        if os.path.exists("./tmp"):
            shutil.rmtree("./tmp")
        return [{**config, "performance": -1, "metric": "error", "unit": "parse_failed"}]

    if not results:
        print(f"ERROR: No results extracted from lm_eval output", flush=True)
        return [{**config, "performance": -1, "metric": "error", "unit": "no_metrics_extracted"}]

    return results


def run_compute_clip_score(model: str, config: Dict[str, Any], eval_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run compute_clip_score framework evaluation (not yet implemented for atom).

    Note: compute_clip_score is implemented in SGLang (see PR #62).
    For atom, consider using SGLang for image generation model evaluation.
    """
    print(f"ERROR: compute_clip_score framework not yet implemented for atom", flush=True)
    print("For image generation models, consider using SGLang which has CLIPScore support (PR #62)", flush=True)
    return [{**config, "performance": -1, "metric": "error", "unit": "compute_clip_score_not_implemented"}]


def run_vbench(model: str, config: Dict[str, Any], eval_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run vbench framework evaluation (not yet implemented for atom)."""
    print(f"ERROR: vbench framework not yet implemented for atom", flush=True)
    print("For video generation models, vbench evaluation is planned but not yet available", flush=True)
    return [{**config, "performance": -1, "metric": "error", "unit": "vbench_not_implemented"}]


def dict_to_env(env_dict: Dict[str, Any]) -> str:
    """Convert environment variable dict to command-line string."""
    if not env_dict:
        return ""
    return " ".join(f"{k}={v}" for k, v in env_dict.items())


def dict_to_args(args_dict: Dict[str, Any]) -> str:
    """Convert arguments dict to command-line string."""
    if not args_dict:
        return ""
    result = ""
    for k, v in args_dict.items():
        if isinstance(v, bool):
            if v:
                result += f"{k} "
        else:
            result += f"{k} {v} "
    return result.strip()


def run_accuracy_atom(
    model: str,
    config: Dict[str, Any],
    wait_for_server_fn: Callable,
    kill_server_fn: Callable
) -> List[Dict[str, Any]]:
    """Run accuracy evaluation for atom with atom-specific server management.

    Args:
        model: Model path or name
        config: Configuration dict
        wait_for_server_fn: Function to wait for server readiness
        kill_server_fn: Function to kill the server process

    Returns:
        List of result dicts containing performance metrics
    """
    # Build server command
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

    # Start server
    print(server_cmd, flush=True)
    server = subprocess.Popen(server_cmd, shell=True)

    try:
        # Wait for server to start
        server_ready = wait_for_server_fn(server)

        if server_ready:
            print(f"Server at {server.pid} contacted successfully", flush=True)
        else:
            print("ERROR: Server failed to start within timeout", flush=True)
            return [{**config, "performance": -1, "metric": "error", "unit": "server_start_timeout"}]

        # Get evaluations list from config
        evaluations = config.get("evaluations", [])
        if not evaluations:
            print("ERROR: No evaluations specified in config", flush=True)
            return [{**config, "performance": -1, "metric": "error", "unit": "no_evaluations"}]

        all_results = []

        # Run each evaluation
        for eval_config in evaluations:
            framework = eval_config.get("framework")

            # Framework must be explicitly specified in acc.yaml
            if not framework:
                print(f"ERROR: No 'framework' specified in evaluation config", flush=True)
                print("acc.yaml must explicitly specify framework (lm_eval, compute_clip_score, vbench)", flush=True)
                return [{**config, "performance": -1, "metric": "error", "unit": "no_framework"}]

            print(f"Running evaluation with framework: {framework}", flush=True)

            # Dispatch to appropriate framework
            if framework == "lm_eval":
                results = run_lm_eval(model, config, eval_config)
            elif framework == "compute_clip_score":
                results = run_compute_clip_score(model, config, eval_config)
            elif framework == "vbench":
                results = run_vbench(model, config, eval_config)
            else:
                print(f"ERROR: Unknown evaluation framework: {framework}", flush=True)
                print("Supported frameworks: lm_eval, compute_clip_score, vbench", flush=True)
                return [{**config, "performance": -1, "metric": "error", "unit": "unknown_framework"}]

            all_results.extend(results)

        return all_results

    finally:
        kill_server_fn(server)
