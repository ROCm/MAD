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
"""CLIPScore accuracy runner for vLLM-Omni diffusion models.

Generates an image via the OpenAI /v1/images/generations endpoint served by
vllm serve --omni, then computes CLIPScore against the prompt using
openai/clip-vit-base-patch32.
"""

import os
import subprocess
from typing import Dict, Any, List

CLIP_SCORE_THRESHOLD = 0.30


def _load_clip_model():
    """Load CLIP model and processor (openai/clip-vit-base-patch32)."""
    from transformers import CLIPModel, CLIPProcessor

    clip_model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model = CLIPModel.from_pretrained(clip_model_name)
    clip_model.eval()
    return clip_model, processor


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


def run_accuracy_vllm_omni(model: str, config: Dict[str, Any],
                           wait_for_server_fn, kill_server_fn,
                           build_server_cmd_fn, server_port: int) -> List[Dict[str, Any]]:
    """Run CLIPScore accuracy for a vLLM-Omni diffusion model.

    Args:
        model: Model path or name
        config: Configuration dict (must have 'evaluations' from acc.yaml merge)
        wait_for_server_fn: Function(server) -> bool to wait for server readiness
        kill_server_fn: Function(server) to kill the server process
        build_server_cmd_fn: Function(model, config, env, extra_args) -> str
        server_port: Port the server listens on
    """
    evaluations = config.get("evaluations", [])
    if not evaluations:
        print(f"ERROR: No evaluations specified in config for model {config['model']}")
        return [{**config, "performance": "fail", "metric": "error", "unit": "no_evaluations"}]

    all_results = []
    for eval_config in evaluations:
        framework = eval_config.get("framework")
        if framework != "compute_clip_score":
            print(f"ERROR: Unknown evaluation framework: {framework}")
            return [{**config, "performance": "fail", "metric": "error", "unit": "unknown_framework"}]

        print(f"Running {framework} evaluation for model {config['model']}")
        results = _run_image_clip_accuracy(
            model, config.copy(), eval_config,
            wait_for_server_fn, kill_server_fn,
            build_server_cmd_fn, server_port,
        )
        all_results.extend(results)

    return all_results


def _run_image_clip_accuracy(model: str, config: Dict[str, Any], eval_config: Dict[str, Any],
                             wait_for_server_fn, kill_server_fn,
                             build_server_cmd_fn, server_port: int) -> List[Dict[str, Any]]:
    """Run CLIPScore accuracy test for text-to-image models via vLLM-Omni."""
    height = eval_config.get("height", 1024)
    width = eval_config.get("width", 1024)
    num_inference_steps = eval_config.get("num_inference_steps", 30)
    guidance_scale = eval_config.get("guidance_scale")
    seed = eval_config.get("seed", 42)
    prompt = eval_config.get("prompt", "A professional photograph of an astronaut riding a horse on the moon")
    clip_threshold = eval_config.get("clip_score_threshold", CLIP_SCORE_THRESHOLD)

    env = config.pop("env", "")
    extra_args = config.pop("extra_args", "")
    server_cmd = build_server_cmd_fn(model, config, env, extra_args)
    config["cmd"] = server_cmd

    print(f"Starting vLLM-Omni server for CLIPScore accuracy: {server_cmd}")
    server = subprocess.Popen(server_cmd, shell=True)

    common_fields = {
        "hf_pipeline_tag": config.get("hf_pipeline_tag"),
        "height": height,
        "width": width,
        "num_inference_steps": num_inference_steps,
        **config,
    }

    try:
        if not wait_for_server_fn(server):
            print("vLLM-Omni server failed to start")
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

        resp = req.post(f"http://localhost:{server_port}/v1/images/generations", json=payload, timeout=600)
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
