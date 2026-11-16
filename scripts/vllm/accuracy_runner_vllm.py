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
VLLM-specific accuracy evaluation runner.

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
from typing import Dict, List, Any


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
    # Support both new evaluations schema and old acc.yaml format
    task = eval_config.get("task", "gsm8k")

    # For apply_chat_template, check eval_config first, then fall back to top-level config
    apply_chat_template = eval_config.get("apply_chat_template")
    if apply_chat_template is None:
        apply_chat_template = config.get("apply_chat_template", False)
    trust_remote_code = eval_config.get("trust_remote_code", False)

    num_fewshot = eval_config.get("num_fewshot", config.get("num_fewshot", 3))
    num_concurrent = eval_config.get("num_concurrent", config.get("num_concurrent", 64))
    max_gen_toks = eval_config.get("max_gen_toks", config.get("max_gen_toks", 2048))
    limit = eval_config.get("limit", config.get("limit"))
    lm_eval_model = eval_config.get("lm_eval_model", config.get("lm_eval_model", "local-completions"))
    base_url = eval_config.get("base_url", config.get("base_url", "http://localhost:8000/v1/completions"))

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

    model_args_str = ",".join([f"{k}={v}" for k, v in model_args.items()])

    # Build base command
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

    # Append bench_cmd to config['cmd'] for reproducibility (server_cmd;bench_cmd pattern)
    if 'cmd' in config:
        config['cmd'] = f"{config['cmd']};{bench_cmd}"
    else:
        config['cmd'] = bench_cmd

    # Run lm_eval
    print(bench_cmd)
    try:
        subprocess.run(bench_cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: lm_eval failed with exit code {e.returncode}")
        # Return config row with error indicator so failure is visible in CSV
        return [{**config, "performance": -1, "metric": "error", "unit": "lm_eval_failed"}]

    # Find output file
    output_files = glob.glob("./tmp/*/*.json")
    if len(output_files) == 0:
        print("ERROR: No output files found after lm_eval completed")
        return [{**config, "performance": -1, "metric": "error", "unit": "no_output_file"}]
    elif len(output_files) > 1:
        print(f"ERROR: Multiple output files found: {output_files}")
        return [{**config, "performance": -1, "metric": "error", "unit": "multiple_output_files"}]

    # Replace / with _ in model name for file path
    model_name = config['model'].replace('/', '_')
    output_json = f"{model_name}_accuracy_{task}_{config['tp']}.json"
    output_file = output_files[0]
    shutil.move(output_file, output_json)
    shutil.rmtree("./tmp")

    # Parse results
    results = []
    with open(output_json, "r", newline="", encoding="utf-8") as f:
        output = json.load(f)

        if "results" in output and task in output["results"]:
            task_results = output["results"][task]

            # Extract all numeric metrics, excluding stderr fields
            # lm_eval includes standard error metrics with _stderr suffix (e.g., exact_match_stderr)
            for metric_key, metric_value in task_results.items():
                if isinstance(metric_value, (int, float)) and not metric_key.endswith('_stderr'):
                    results.append({
                        "performance": metric_value,
                        "metric": metric_key,
                        "unit": "score",
                        **config
                    })

    return results


def run_compute_clip_score(model: str, config: Dict[str, Any], eval_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run compute_clip_score framework evaluation (not yet implemented for vLLM).

    Note: compute_clip_score is implemented in SGLang (see PR #62).
    For vLLM, use vLLM-Omni for image generation models.
    """
    print(f"ERROR: compute_clip_score framework not yet implemented for vLLM")
    print("For image generation models, consider using SGLang which has CLIPScore support (PR #62)")
    print("Or use vLLM-Omni for text-to-image diffusion models")
    return [{**config, "performance": -1, "metric": "error", "unit": "compute_clip_score_not_implemented"}]


def run_vbench(model: str, config: Dict[str, Any], eval_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run vbench framework evaluation (not yet implemented for vLLM)."""
    print(f"ERROR: vbench framework not yet implemented for vLLM")
    print("For video generation models, vbench evaluation is planned but not yet available")
    return [{**config, "performance": -1, "metric": "error", "unit": "vbench_not_implemented"}]


def dict_to_env(env_input) -> str:
    """Convert environment variable dict to command-line string.

    Args:
        env_input: Either a dict to convert, or an already-converted string
    """
    if not env_input:
        return ""
    if isinstance(env_input, str):
        return env_input
    return " ".join(f"{k}={v}" for k, v in env_input.items())


def dict_to_args(args_input) -> str:
    """Convert arguments dict to command-line string.

    Args:
        args_input: Either a dict to convert, or an already-converted string
    """
    if not args_input:
        return ""
    if isinstance(args_input, str):
        return args_input
    result = ""
    for k, v in args_input.items():
        if isinstance(v, bool):
            if v:
                result += f"{k} "
        else:
            result += f"{k} {v} "
    return result.strip()


def run_accuracy_vllm(model: str, config: Dict[str, Any], _kill_server_fn) -> List[Dict[str, Any]]:
    """Run accuracy evaluation for vllm with support for multiple frameworks.

    Args:
        model: Model path or name
        config: Configuration dict
        _kill_server_fn: Function to kill the server process

    Returns:
        List of result dicts containing performance metrics
    """
    server_cmd = (
        "vllm serve "
        f"{model} "
        f"--dtype {config['dtype']} "
        f"-tp {config['tp']} "
        f"--no-enable-prefix-caching "
        f"--disable-uvicorn-access-log "
    )

    # Get env and extra args from config (may be dicts or already-converted strings)
    env_input = config.pop('env', {})
    extra_args_input = config.pop('extra_args', {})

    # Convert to command line strings (handles both dict and string inputs)
    env_str = dict_to_env(env_input)
    extra_args_str = dict_to_args(extra_args_input)

    server_cmd = f"{env_str} {server_cmd} {extra_args_str}".strip()
    config["cmd"] = server_cmd

    # start server
    print(server_cmd)
    server = subprocess.Popen(server_cmd, shell=True)

    try:
        # wait for server to start; timeout after 30 minutes (fail fast if the server exits)
        status = subprocess.run(
            f"timeout 5400 bash -c 'until curl -s http://localhost:8000/v1/models; do kill -0 {server.pid} 2>/dev/null || exit 1; sleep 30; done' || exit 1",
            shell=True,
        )
        if status.returncode != 0:
            print("ERROR: Server failed to start within timeout")
            return [{**config, "performance": -1, "metric": "error", "unit": "server_start_timeout"}]
        print(f"Server at {server.pid} contacted successfully", flush=True)

        # Get evaluations list from config
        evaluations = config.get("evaluations", [])

        # Evaluations must be specified in config (acc.yaml)
        if not evaluations:
            print(f"ERROR: No evaluations specified in config for model {config['model']}")
            print("acc.yaml must specify 'evaluations' list with 'framework' field")
            return [{**config, "performance": -1, "metric": "error", "unit": "no_evaluations"}]

        all_results = []

        # Run each evaluation
        for eval_config in evaluations:
            framework = eval_config.get("framework")
            if not framework:
                print(f"ERROR: No 'framework' specified in evaluation config")
                print("acc.yaml must explicitly specify framework (lm_eval, compute_clip_score, vbench)")
                return [{**config, "performance": -1, "metric": "error", "unit": "no_framework"}]

            print(f"Running {framework} evaluation for model {config['model']}")

            # Dispatch to appropriate framework
            if framework == "lm_eval":
                results = run_lm_eval(model, config, eval_config)
            elif framework == "compute_clip_score":
                results = run_compute_clip_score(model, config, eval_config)
            elif framework == "vbench":
                results = run_vbench(model, config, eval_config)
            else:
                print(f"ERROR: Unknown evaluation framework: {framework}")
                print("Supported frameworks: lm_eval, compute_clip_score, vbench")
                return [{**config, "performance": -1, "metric": "error", "unit": "unknown_framework"}]

            all_results.extend(results)

        return all_results

    finally:
        _kill_server_fn(server)
