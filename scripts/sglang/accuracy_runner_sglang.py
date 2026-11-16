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
SGLang-specific accuracy evaluation runner.

Supports multiple evaluation frameworks based on acc.yaml schema:
- lm_eval: Language model evaluation (for text-generation models)
- compute_clip_score: CLIP score for text-to-image and text-to-video models
- vbench: Video generation benchmark (future)

Schema:
- Each acc.yaml entry contains: model (space-separated), evaluations list
- Each evaluation contains: framework field
- lm_eval supports: task, num_fewshot, apply_chat_template, base_url,
  lm_eval_model, num_concurrent, max_gen_toks, limit
- compute_clip_score supports: prompt, height, width, num_inference_steps,
  guidance_scale, seed, clip_score_threshold, num_frames (video only)
- Default framework based on hf_pipeline_tag:
  text-generation -> lm_eval, text-to-image -> compute_clip_score,
  text-to-video -> compute_clip_score, video-generation -> vbench
"""

import os
import glob
import json
import shutil
import subprocess
import signal
import time
import psutil
from typing import Dict, List, Any

CLIP_SCORE_THRESHOLD = 0.25


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

    # Run lm_eval
    config["cmd"] = f"{config.get('cmd', '')};{bench_cmd}".strip(";")
    print(bench_cmd)
    subprocess.run(bench_cmd, shell=True, check=True)

    # Find output file
    output_files = glob.glob("./tmp/*/*.json")
    if len(output_files) == 0:
        print("ERROR: No output files found after lm_eval completed")
        return [{**config, "performance": "fail", "metric": "error", "unit": "no_output_file"}]
    elif len(output_files) > 1:
        print(f"ERROR: Multiple output files found: {output_files}")
        return [{**config, "performance": "fail", "metric": "error", "unit": "multiple_output_files"}]

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


def _compute_clip_score(image, prompt, clip_model, processor):
    """Compute CLIPScore (text-to-image) for a single image against a prompt."""
    import torch
    from PIL import Image as PILImage

    if not isinstance(image, PILImage.Image):
        image = PILImage.fromarray(image)

    inputs = processor(text=[prompt], images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = clip_model(**inputs)
        score = outputs.logits_per_image.item() / 100.0
    return score


def _load_clip_model():
    """Load CLIP model and processor (openai/clip-vit-base-patch32)."""
    from transformers import CLIPModel, CLIPProcessor

    clip_model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model = CLIPModel.from_pretrained(clip_model_name)
    clip_model.eval()
    return clip_model, processor


def run_compute_clip_score(model: str, config: Dict[str, Any], eval_config: Dict[str, Any], wait_for_server_fn, kill_server_fn) -> List[Dict[str, Any]]:
    """Run compute_clip_score framework evaluation for image or video models.

    For text-to-image: generates an image and computes CLIPScore.
    For text-to-video: generates a video, extracts 3 key frames, computes CLIPScore per frame.

    Args:
        model: Model path or name
        config: Main configuration dict
        eval_config: Evaluation-specific configuration
        wait_for_server_fn: Function to wait for server readiness
        kill_server_fn: Function to kill the server process

    Returns:
        List of result dicts containing CLIPScore metrics
    """
    hf_pipeline_tag = config.get("hf_pipeline_tag")

    if hf_pipeline_tag == "text-to-image":
        return _run_image_clip_accuracy(model, config, eval_config, wait_for_server_fn, kill_server_fn)
    elif hf_pipeline_tag == "text-to-video":
        return _run_video_clip_accuracy(model, config, eval_config, wait_for_server_fn, kill_server_fn)
    else:
        print(f"ERROR: compute_clip_score only supports text-to-image and text-to-video, got {hf_pipeline_tag}")
        return [{**config, "performance": "fail", "metric": "error", "unit": "unsupported_pipeline_tag"}]


def _run_image_clip_accuracy(model: str, config: Dict[str, Any], eval_config: Dict[str, Any], wait_for_server_fn, kill_server_fn) -> List[Dict[str, Any]]:
    """Run CLIPScore accuracy test for text-to-image models."""
    port = 30000
    tp = int(config.get("tp", 1))

    # Extract image generation parameters from eval_config
    height = eval_config.get("height", 1024)
    width = eval_config.get("width", 1024)
    num_inference_steps = eval_config.get("num_inference_steps", 30)
    guidance_scale = eval_config.get("guidance_scale")
    seed = eval_config.get("seed", 42)
    prompt = eval_config.get("prompt", "A professional photograph of an astronaut riding a horse on the moon")
    clip_threshold = eval_config.get("clip_score_threshold", CLIP_SCORE_THRESHOLD)

    # Get env and extra args
    env_input = config.pop('env', {})
    extra_args_input = config.pop('extra_args', {})
    env_str = dict_to_env(env_input)
    extra_args_str = dict_to_args(extra_args_input)

    # Resolve parallelism (ulysses_degree, enable_cfg_parallel)
    ulysses_degree = config.pop('ulysses_degree', None)
    enable_cfg_parallel = config.pop('enable_cfg_parallel', None)
    ulysses_factor = int(ulysses_degree) if ulysses_degree is not None else 1
    cfg_factor = 2 if enable_cfg_parallel else 1
    num_gpus = tp * ulysses_factor * cfg_factor

    parallel_parts = [f"--num-gpus {num_gpus}", f"--tp-size {tp}"]
    if ulysses_degree is not None:
        parallel_parts.append(f"--ulysses-degree {ulysses_degree}")
    if enable_cfg_parallel:
        parallel_parts.append("--enable-cfg-parallel true")
    parallel_args = " ".join(parallel_parts)

    server_cmd = (
        f"sglang serve "
        f"--model-path {model} "
        + (f"--quantization {config.get('quantization')} " if config.get('quantization') else "")
        + f"{parallel_args} "
        f"--host 0.0.0.0 "
        f"--port {port} "
        f"{extra_args_str}"
    ).strip()
    server_cmd = f"{env_str} {server_cmd}".strip()
    config["cmd"] = server_cmd

    print(f"Starting image server for CLIPScore accuracy: {server_cmd}")
    server = subprocess.Popen(server_cmd, shell=True)

    common_fields = {
        "hf_pipeline_tag": config.get("hf_pipeline_tag"),
        "height": height,
        "width": width,
        "num_inference_steps": num_inference_steps,
        **config,
    }

    try:
        if not wait_for_server_fn(server, f"http://localhost:{port}/health"):
            print("Image server failed to start")
            return [{"performance": "fail", "metric": "clip_score", "unit": "score", **common_fields}]
        print(f"Server at {server.pid} contacted successfully", flush=True)

        import base64
        import io
        import requests as req
        from PIL import Image

        print(f"Generating image with prompt: '{prompt}'")
        payload = {
            "model": model,
            "prompt": prompt,
            "size": f"{width}x{height}",
            "n": 1,
            "response_format": "b64_json",
        }
        if num_inference_steps is not None:
            payload["num_inference_steps"] = num_inference_steps
        if guidance_scale is not None:
            payload["guidance_scale"] = guidance_scale
        if seed is not None:
            payload["seed"] = seed

        resp = req.post(f"http://localhost:{port}/v1/images/generations", json=payload, timeout=600)
        resp.raise_for_status()
        resp_data = resp.json()
        image_b64 = resp_data["data"][0]["b64_json"]
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        clip_model, processor = _load_clip_model()
        score = _compute_clip_score(image, prompt, clip_model, processor)
        passed = score >= clip_threshold
        status = "PASS" if passed else "FAIL"

        print(f"CLIPScore: {score:.4f} (threshold={clip_threshold}, {status})")

        results = []
        results.append({
            "performance": "pass" if passed else "fail",
            "metric": "image_clip_accuracy",
            "unit": "pass/fail",
            **common_fields,
        })
        results.append({
            "performance": f"{score:.4f}",
            "metric": "clip_score",
            "unit": "score",
            **common_fields,
        })
        return results

    finally:
        kill_server_fn(server)


def _run_video_clip_accuracy(model: str, config: Dict[str, Any], eval_config: Dict[str, Any], wait_for_server_fn, kill_server_fn) -> List[Dict[str, Any]]:
    """Run CLIPScore accuracy test for text-to-video models.

    Generates a video, extracts 3 key frames (first, middle, last),
    and computes CLIPScore (text-to-image) for each frame against the prompt.
    """
    import tempfile

    port = 30000
    tp = int(config.get("tp", 1))

    # Extract video generation parameters from eval_config
    height = eval_config.get("height", 480)
    width = eval_config.get("width", 832)
    num_frames = eval_config.get("num_frames", 81)
    num_inference_steps = eval_config.get("num_inference_steps", 50)
    guidance_scale = eval_config.get("guidance_scale")
    seed = eval_config.get("seed", 42)
    prompt = eval_config.get("prompt", "A serene mountain lake at sunrise with mist rising from the water")
    clip_threshold = eval_config.get("clip_score_threshold", CLIP_SCORE_THRESHOLD)

    # Get env and extra args
    env_input = config.pop('env', {})
    extra_args_input = config.pop('extra_args', {})
    env_str = dict_to_env(env_input)
    extra_args_str = dict_to_args(extra_args_input)

    # Resolve parallelism (ulysses_degree, enable_cfg_parallel)
    ulysses_degree = config.pop('ulysses_degree', None)
    enable_cfg_parallel = config.pop('enable_cfg_parallel', None)
    ulysses_factor = int(ulysses_degree) if ulysses_degree is not None else 1
    cfg_factor = 2 if enable_cfg_parallel else 1
    num_gpus = tp * ulysses_factor * cfg_factor

    parallel_parts = [f"--num-gpus {num_gpus}", f"--tp-size {tp}"]
    if ulysses_degree is not None:
        parallel_parts.append(f"--ulysses-degree {ulysses_degree}")
    if enable_cfg_parallel:
        parallel_parts.append("--enable-cfg-parallel true")
    parallel_args = " ".join(parallel_parts)

    server_cmd = (
        f"sglang serve "
        f"--model-path {model} "
        + (f"--quantization {config.get('quantization')} " if config.get('quantization') else "")
        + f"{parallel_args} "
        f"--host 0.0.0.0 "
        f"--port {port} "
        f"{extra_args_str}"
    ).strip()
    server_cmd = f"{env_str} {server_cmd}".strip()
    config["cmd"] = server_cmd

    print(f"Starting video server for CLIPScore accuracy: {server_cmd}")
    server = subprocess.Popen(server_cmd, shell=True)

    common_fields = {
        "hf_pipeline_tag": config.get("hf_pipeline_tag"),
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "num_inference_steps": num_inference_steps,
        **config,
    }

    try:
        if not wait_for_server_fn(server, f"http://localhost:{port}/health"):
            print("Video server failed to start")
            return [{"performance": "fail", "metric": "clip_score", "unit": "score", **common_fields}]
        print(f"Server at {server.pid} contacted successfully", flush=True)

        import cv2
        import numpy as np
        import requests as req
        from PIL import Image

        print(f"Generating video with prompt: '{prompt}'")
        payload = {
            "prompt": prompt,
            "size": f"{width}x{height}",
            "num_inference_steps": num_inference_steps,
            "seed": seed,
        }
        if num_frames is not None:
            payload["n_frames"] = num_frames
        if guidance_scale is not None:
            payload["guidance_scale"] = guidance_scale

        resp = req.post(f"http://localhost:{port}/v1/videos", json=payload, timeout=1200)
        resp.raise_for_status()
        job = resp.json()
        job_id = job.get("id")
        print(f"Video job submitted: id={job_id}, status={job.get('status')}")

        poll_url = f"http://localhost:{port}/v1/videos/{job_id}"
        poll_deadline = time.time() + 1200  # 20 minute deadline for video generation
        while True:
            if time.time() > poll_deadline:
                raise RuntimeError(f"Video generation timed out after 1200s (job_id={job_id} stuck in non-terminal state)")
            poll_resp = req.get(poll_url, timeout=60)
            poll_resp.raise_for_status()
            status_data = poll_resp.json()
            status = status_data.get("status")
            if status == "completed":
                print(f"Video generation completed")
                break
            elif status == "failed":
                raise RuntimeError(f"Video generation failed: {status_data.get('error')}")
            time.sleep(2)

        content_url = f"http://localhost:{port}/v1/videos/{job_id}/content"
        content_resp = req.get(content_url, timeout=300)
        content_resp.raise_for_status()
        video_bytes = content_resp.content

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        try:
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                raise ValueError("Failed to open generated video")
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total < 1:
                raise ValueError("Video has no frames")
            key_indices = [0, total // 2, total - 1]
            frames = []
            for idx in key_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
        finally:
            os.unlink(tmp_path)

        print(f"Extracted {len(frames)} key frames (indices: {key_indices}, total: {total})")

        # Validate all key frames were decoded
        if len(frames) < len(key_indices):
            raise RuntimeError(
                f"Failed to decode all key frames: got {len(frames)} of {len(key_indices)} requested frames. "
                f"OpenCV failed to decode frames at indices {key_indices}."
            )

        clip_model, processor = _load_clip_model()

        frame_labels = ["first", "middle", "last"]
        clip_scores = []
        for i, frame in enumerate(frames):
            pil_image = Image.fromarray(frame)
            score = _compute_clip_score(pil_image, prompt, clip_model, processor)
            clip_scores.append(score)
            print(f"  Frame {frame_labels[i]} (idx {key_indices[i]}): CLIPScore = {score:.4f}")

        # Safe to compute min/mean now (validated len(clip_scores) == 3 above)
        min_score = min(clip_scores)
        mean_score = sum(clip_scores) / len(clip_scores)
        passed = min_score >= clip_threshold
        status = "PASS" if passed else "FAIL"

        print(f"\nCLIPScore summary: min={min_score:.4f}, mean={mean_score:.4f}, "
              f"threshold={clip_threshold}, {status}")

        results = []
        results.append({
            "performance": "pass" if passed else "fail",
            "metric": "video_clip_accuracy",
            "unit": "pass/fail",
            **common_fields,
        })
        results.append({
            "performance": f"{mean_score:.4f}",
            "metric": "clip_score_mean",
            "unit": "score",
            **common_fields,
        })
        results.append({
            "performance": f"{min_score:.4f}",
            "metric": "clip_score",
            "unit": "score",
            **common_fields,
        })
        for i, label in enumerate(frame_labels):
            results.append({
                "performance": f"{clip_scores[i]:.4f}",
                "metric": f"clip_score_{label}_frame",
                "unit": "score",
                **common_fields,
            })
        return results

    finally:
        kill_server_fn(server)


def run_vbench(model: str, config: Dict[str, Any], eval_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run vbench framework evaluation (not yet implemented)."""
    print(f"ERROR: vbench framework not yet implemented for SGLang")
    print("For video generation models, vbench evaluation is planned but not yet available")
    return [{**config, "performance": "fail", "metric": "error", "unit": "vbench_not_implemented"}]


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


def run_accuracy_sglang(model: str, config: Dict[str, Any], wait_for_server_fn, kill_server_fn) -> List[Dict[str, Any]]:
    """Run accuracy evaluation for sglang with support for multiple frameworks.

    Args:
        model: Model path or name
        config: Configuration dict
        wait_for_server_fn: Function to wait for server readiness
        kill_server_fn: Function to kill the server process

    Returns:
        List of result dicts containing performance metrics
    """
    # Get evaluations list from config
    evaluations = config.get("evaluations", [])

    # Evaluations must be specified in config (acc.yaml)
    if not evaluations:
        print(f"ERROR: No evaluations specified in config for model {config['model']}")
        print("acc.yaml must specify 'evaluations' list with 'framework' field")
        return [{**config, "performance": "fail", "metric": "error", "unit": "no_evaluations"}]

    # Check which framework(s) we're using
    frameworks = [eval_config.get("framework") for eval_config in evaluations]

    # compute_clip_score manages its own server lifecycle
    if "compute_clip_score" in frameworks:
        all_results = []
        for eval_config in evaluations:
            framework = eval_config.get("framework")
            if not framework:
                print(f"ERROR: No 'framework' specified in evaluation config")
                print("acc.yaml must explicitly specify framework (lm_eval, compute_clip_score, vbench)")
                return [{**config, "performance": "fail", "metric": "error", "unit": "no_framework"}]

            print(f"Running {framework} evaluation for model {config['model']}")

            if framework == "compute_clip_score":
                results = run_compute_clip_score(model, config.copy(), eval_config, wait_for_server_fn, kill_server_fn)
            elif framework == "lm_eval":
                print(f"ERROR: Cannot mix lm_eval and compute_clip_score in same config")
                return [{**config, "performance": "fail", "metric": "error", "unit": "mixed_frameworks"}]
            elif framework == "vbench":
                results = run_vbench(model, config, eval_config)
            else:
                print(f"ERROR: Unknown evaluation framework: {framework}")
                print("Supported frameworks: lm_eval, compute_clip_score, vbench")
                return [{**config, "performance": "fail", "metric": "error", "unit": "unknown_framework"}]

            all_results.extend(results)
        return all_results

    # Resolve tokenizer arg for models with only tokenizer.tok.json (e.g., Kimi)
    def _resolve_tokenizer_arg(model_path):
        """Return --tokenizer-path arg for tekken-only models, empty string otherwise."""
        if os.path.isdir(model_path):
            if os.path.isfile(os.path.join(model_path, "tokenizer.json")):
                return ""  # Standard HF tokenizer
            tok = os.path.join(model_path, "tokenizer.tok.json")
            if os.path.isfile(tok):
                return f"--tokenizer-path {tok} "
        # For hub IDs, check HF cache
        cache = os.environ.get("HF_HUB_CACHE") or os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface", "hub"
        )
        if os.path.isdir(cache):
            for entry in os.listdir(cache):
                if entry.startswith("models--") and model_path.replace("/", "--") in entry:
                    snapshot_dir = os.path.join(cache, entry, "snapshots")
                    if os.path.isdir(snapshot_dir):
                        for snap in os.listdir(snapshot_dir):
                            snap_path = os.path.join(snapshot_dir, snap)
                            if os.path.isdir(snap_path):
                                if os.path.isfile(os.path.join(snap_path, "tokenizer.json")):
                                    return ""
                                tok = os.path.join(snap_path, "tokenizer.tok.json")
                                if os.path.isfile(tok):
                                    return f"--tokenizer-path {tok} "
        return ""

    tokenizer_arg = _resolve_tokenizer_arg(model)

    # For lm_eval, start a shared server for all evaluations
    server_cmd = (
        "python -m sglang.launch_server "
        f"--model-path {model} "
        f"--dtype {config['dtype']} "
        + (f"--quantization {config.get('quantization')} " if config.get('quantization') else "")
        + (f"--kv-cache-dtype {config.get('kv_cache_dtype')} " if config.get('kv_cache_dtype') else "")
        + f"--tensor-parallel-size {config['tp']} "
        + tokenizer_arg
        + f"--host 0.0.0.0 "
        f"--port 8000 "
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
        # wait for server to start; timeout after 30 minutes
        server_ready = wait_for_server_fn(server, "http://localhost:8000/v1/models", timeout=5400)
        if not server_ready:
            print("ERROR: Server failed to start within timeout", flush=True)
            return [{**config, "performance": "fail", "metric": "error", "unit": "server_start_timeout"}]
        print(f"Server at {server.pid} contacted successfully", flush=True)

        all_results = []

        # Run each evaluation
        for eval_config in evaluations:
            framework = eval_config.get("framework")
            if not framework:
                print(f"ERROR: No 'framework' specified in evaluation config")
                print("acc.yaml must explicitly specify framework (lm_eval, compute_clip_score, vbench)")
                return [{**config, "performance": "fail", "metric": "error", "unit": "no_framework"}]

            print(f"Running {framework} evaluation for model {config['model']}")

            # Dispatch to appropriate framework
            if framework == "lm_eval":
                results = run_lm_eval(model, config, eval_config)
            elif framework == "vbench":
                results = run_vbench(model, config, eval_config)
            else:
                print(f"ERROR: Unknown evaluation framework: {framework}")
                print("Supported frameworks: lm_eval, compute_clip_score, vbench")
                return [{**config, "performance": "fail", "metric": "error", "unit": "unknown_framework"}]

            all_results.extend(results)

        return all_results

    finally:
        kill_server_fn(server)
