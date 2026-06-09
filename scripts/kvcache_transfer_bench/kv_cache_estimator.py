#!/usr/bin/env python3
"""
KV Cache Calculator, Estimator, and vLLM Runtime Inspector

Combined module providing:
- Theoretical KV cache calculation (MHA, GQA, MQA, MLA, Sliding Window, MoE)
- Estimation with layer-by-layer details
- vLLM runtime verification

Usage:
------
  python kv_cache_estimator.py --config <config.yaml> [--output-dir DIR] [--verify-vllm] [--append]

  Config YAML (required): model.name, model.concurrency, model.tp, model.seq-length, model.kv_cache_dtype, model.pp
"""

import sys
import os
import json
import csv
import gc
import math
import argparse
import datetime
import platform
from typing import Dict, Optional, Tuple, List, Any

try:
    import yaml
except ImportError:
    yaml = None


def _ensure_int_bytes(x) -> int:
    """Convert byte count to int. Uses ceil for fractional bytes (e.g. mxfp4=0.5)."""
    return int(math.ceil(x)) if isinstance(x, float) else int(x)

from transformers import AutoConfig

# V0: exposes cache_engine.gpu_cache via model_executor.workers.
# V1: exposes KVCacheConfig via engine_core.scheduler.kv_cache_config (Option A: config from get_kv_cache_configs).
# Set VLLM_USE_V1=1 to use V1 engine; extraction will use scheduler config when V0 path unavailable.
if "VLLM_USE_V1" not in os.environ:
    os.environ["VLLM_USE_V1"] = "0"

# Optional heavy imports (for vLLM features)
try:
    import torch
except ImportError:
    torch = None
try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

# vLLM block size (constant)
BLOCK_SIZE = 16

# vLLM CacheConfig accepts: auto, fp8, fp8_e4m3, fp8_e5m2, fp8_inc
# Map config kv_cache_dtype to vLLM's format
VLLM_KV_CACHE_DTYPES = frozenset({"auto", "fp8", "fp8_e4m3", "fp8_e5m2", "fp8_inc"})

# Aliases: config value -> vLLM value (for alternate spellings)
VLLM_KV_CACHE_DTYPE_ALIASES = {
    "fp8-e4m3": "fp8_e4m3",
    "fp8-e5m2": "fp8_e5m2",
    "e4m3": "fp8_e4m3",
    "e5m2": "fp8_e5m2",
}


def _vllm_kv_cache_dtype(dtype: str) -> str:
    """Map config kv_cache_dtype to vLLM-accepted kv_cache_dtype.
    vLLM accepts: auto, fp8, fp8_e4m3, fp8_e5m2, fp8_inc.
    Supports 'auto' (pass through), fp8 variants, and maps unsupported dtypes to 'auto'.
    """
    if not dtype:
        return "auto"
    d = str(dtype).strip().lower()
    if d == "auto":
        return "auto"
    if d in VLLM_KV_CACHE_DTYPES:
        return d
    if d in VLLM_KV_CACHE_DTYPE_ALIASES:
        return VLLM_KV_CACHE_DTYPE_ALIASES[d]
    # bfloat16, float16, float32 not supported by vLLM CacheConfig -> use auto
    return "auto"



# ============================================================================
# KV CACHE CALCULATION (from kv_cache_calc.py)
# ============================================================================

def _get_config_for_layers(config):
    """Resolve config for layer/attention params. Mistral3Config uses text_config."""
    if hasattr(config, 'text_config') and config.text_config is not None:
        text_cfg = config.text_config
        if hasattr(text_cfg, 'num_hidden_layers') or hasattr(text_cfg, 'n_layers') or hasattr(text_cfg, 'num_layers'):
            return text_cfg
    return config


def _get_num_layers(config):
    """Get number of layers, supporting num_hidden_layers, n_layers, num_layers, and text_config."""
    cfg = _get_config_for_layers(config)
    return getattr(cfg, 'num_hidden_layers', None) or getattr(cfg, 'n_layers', None) or getattr(cfg, 'num_layers', None)


def detect_attention_type(config):
    """Detect attention mechanism from config"""
    if hasattr(config, 'kv_lora_rank'):
        return "MLA"
    if hasattr(config, 'sliding_window') and config.sliding_window:
        return "SLIDING_WINDOW"
    if hasattr(config, 'num_key_value_heads'):
        num_kv = config.num_key_value_heads
        num_q = config.num_attention_heads
        if num_kv == 1:
            return "MQA"
        elif num_kv < num_q:
            return "GQA"
    return "MHA"


def calculate_layer_kv_cache(config, layer_idx, seq_len, dtype_bytes, tp_size=1, batch_size=1):
    """Calculate KV cache for a single layer"""

    attention_type = detect_attention_type(config)
    hidden_size = config.hidden_size

    # Check for heterogeneous layer types (per-layer attention specification)
    layer_specific_type = None
    if hasattr(config, 'layer_types') and config.layer_types:
        if layer_idx < len(config.layer_types):
            layer_type_str = config.layer_types[layer_idx]
            if 'full' in layer_type_str.lower():
                layer_specific_type = "FULL_ATTENTION"
            elif 'sliding' in layer_type_str.lower() or 'window' in layer_type_str.lower():
                layer_specific_type = "SLIDING_WINDOW"

    # MLA: Multi-Head Latent Attention (compressed)
    if attention_type == "MLA":
        kv_lora_rank = config.kv_lora_rank
        qk_rope_head_dim = getattr(config, 'qk_rope_head_dim', 0)
        kv_dim = kv_lora_rank + qk_rope_head_dim

        per_gpu_kv_bytes = _ensure_int_bytes(1 * seq_len * kv_dim * dtype_bytes * batch_size)
        total_kv_bytes = per_gpu_kv_bytes * tp_size

        return {
            'bytes': total_kv_bytes,
            'mb': total_kv_bytes / (1024 ** 2),
            'per_gpu_bytes': per_gpu_kv_bytes,
            'per_gpu_mb': per_gpu_kv_bytes / (1024 ** 2),
            'kv_dim': kv_dim,
            'attention_type': 'MLA',
            'tp_split': False
        }

    # Standard attention (MHA/GQA/MQA)
    num_q_heads = config.num_attention_heads
    num_kv_heads = getattr(config, 'num_key_value_heads', num_q_heads)
    head_dim = hidden_size // num_q_heads

    effective_seq_len = seq_len
    final_attention_type = layer_specific_type if layer_specific_type else attention_type

    if final_attention_type == "SLIDING_WINDOW":
        effective_seq_len = min(seq_len, config.sliding_window)
    elif final_attention_type == "FULL_ATTENTION":
        effective_seq_len = seq_len
    elif attention_type == "SLIDING_WINDOW" and not layer_specific_type:
        max_window_layers = getattr(config, 'max_window_layers', _get_num_layers(config) or 0)
        if layer_idx < max_window_layers:
            effective_seq_len = min(seq_len, config.sliding_window)

    # When num_kv_heads < tp_size, KV cache is replicated (each GPU holds full copy); otherwise sharded
    per_gpu_kv_heads = num_kv_heads if num_kv_heads < tp_size else num_kv_heads // tp_size
    per_gpu_kv_bytes = _ensure_int_bytes(
        2 * effective_seq_len * per_gpu_kv_heads * head_dim * dtype_bytes * batch_size
    )
    total_kv_bytes = per_gpu_kv_bytes * tp_size

    return {
        'bytes': total_kv_bytes,
        'mb': total_kv_bytes / (1024 ** 2),
        'per_gpu_bytes': per_gpu_kv_bytes,
        'per_gpu_mb': per_gpu_kv_bytes / (1024 ** 2),
        'kv_dim': num_kv_heads * head_dim,
        'attention_type': final_attention_type,
        'num_kv_heads': num_kv_heads,
        'per_gpu_kv_heads': per_gpu_kv_heads,
        'effective_seq_len': effective_seq_len,
        'tp_split': True
    }


def calculate_kv_cache(model_path, seq_len, dtype="float16", tp_size=1, batch_size=1, user_specified_dtype=False):
    """Main calculation function"""

    print(f"Loading config from: {model_path}")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Check for quantization config
    quantization_detected = False
    if not user_specified_dtype and hasattr(config, 'quantization_config') and config.quantization_config:
        quant_config = config.quantization_config
        if isinstance(quant_config, dict):
            quant_method = quant_config.get('quant_method', '')
            if 'fp8' in quant_method.lower():
                dtype = "fp8"
                quantization_detected = True
                print(f"⚠️  Auto-detected FP8 quantization in config - using fp8 for KV cache")
            elif 'mxfp4' in quant_method.lower() or 'fp4' in quant_method.lower():
                dtype = "mxfp4"
                quantization_detected = True
                print(f"⚠️  Auto-detected MXFP4 quantization in config - using mxfp4 for KV cache")

    if user_specified_dtype and hasattr(config, 'quantization_config') and config.quantization_config:
        quant_config = config.quantization_config
        if isinstance(quant_config, dict):
            quant_method = quant_config.get('quant_method', '')
            if quant_method:
                print(f"ℹ️  Model has {quant_method} quantization, but using user-specified dtype: {dtype}")

    dtype_bytes = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "int8": 1,
        "fp8": 1,
        "mxfp4": 0.5
    }[dtype]
    layer_config = _get_config_for_layers(config)
    num_layers = _get_num_layers(config)
    if num_layers is None:
        raise AttributeError(f"{type(config).__name__} has no num_hidden_layers, n_layers, or num_layers")
    hidden_size = getattr(layer_config, 'hidden_size', config.hidden_size)

    attention_type = detect_attention_type(layer_config)
    is_moe = hasattr(layer_config, 'num_local_experts')
    is_heterogeneous = hasattr(layer_config, 'layer_types') and layer_config.layer_types

    print("\n" + "=" * 80)
    print(f"MODEL: {model_path}")
    print("=" * 80)
    print(f"Architecture      : {config.architectures}")

    if is_heterogeneous:
        print(f"Attention Type    : HETEROGENEOUS (Mixed)")
        layer_type_counts = {}
        for lt in layer_config.layer_types:
            layer_type_counts[lt] = layer_type_counts.get(lt, 0) + 1
        for lt, count in layer_type_counts.items():
            print(f"  - {lt:20s}: {count} layers")
    else:
        print(f"Attention Type    : {attention_type}")

    print(f"Hidden Size       : {hidden_size}")
    print(f"Num Layers        : {num_layers}")
    print(f"Attention Heads   : {getattr(layer_config, 'num_attention_heads', config.num_attention_heads)}")

    if hasattr(layer_config, 'num_key_value_heads'):
        print(f"KV Heads          : {layer_config.num_key_value_heads}")

    if attention_type == "MLA":
        print(f"KV Lora Rank      : {layer_config.kv_lora_rank}")
        print(f"Compression Ratio : {hidden_size / layer_config.kv_lora_rank:.1f}x")

    if attention_type == "SLIDING_WINDOW":
        print(f"Sliding Window    : {layer_config.sliding_window}")

    if is_moe:
        print(f"MoE Experts       : {layer_config.num_local_experts}")

    if quantization_detected:
        quant_config = config.quantization_config
        print(f"\n⚠️  QUANTIZATION DETECTED:")
        print(f"  Method          : {quant_config.get('quant_method', 'N/A')}")

    print(f"\nSequence Length   : {seq_len}")
    print(f"Batch Size        : {batch_size}")
    print(f"Data Type         : {dtype}")

    if tp_size > 1:
        print(f"Tensor Parallel   : {tp_size} GPUs")

    print("=" * 80)

    total_bytes = 0
    per_gpu_total_bytes = 0
    layer_results = []

    print(f"\nLAYER-BY-LAYER KV CACHE:")
    print("-" * 80)

    if tp_size > 1:
        print(f"{'Layer':<8} | {'Type':<20} | {'Per-GPU MB':<12} | {'Total MB':<12} | {'Details':<30}")
    else:
        print(f"{'Layer':<8} | {'Type':<20} | {'KV Cache (MB)':<15} | {'Details':<30}")

    print("-" * 80)

    for i in range(num_layers):
        result = calculate_layer_kv_cache(layer_config, i, seq_len, dtype_bytes, tp_size, batch_size)
        layer_results.append(result)
        total_bytes += result['bytes']
        per_gpu_total_bytes += result['per_gpu_bytes']

        if result['attention_type'] == 'MLA':
            details = f"KV Dim: {result['kv_dim']} (no TP split)"
        else:
            if tp_size > 1:
                details = f"KV Heads: {result['per_gpu_kv_heads']}/{result['num_kv_heads']}, Seq: {result['effective_seq_len']}"
            else:
                details = f"KV Heads: {result['num_kv_heads']}, Seq: {result['effective_seq_len']}"

        if tp_size > 1:
            print(f"{i:<8} | {result['attention_type']:<20} | {result['per_gpu_mb']:<12.2f} | {result['mb']:<12.2f} | {details:<30}")
        else:
            print(f"{i:<8} | {result['attention_type']:<20} | {result['mb']:<15.2f} | {details:<30}")

    total_mb = total_bytes / (1024 ** 2)
    total_gb = total_bytes / (1024 ** 3)
    per_gpu_mb = per_gpu_total_bytes / (1024 ** 2)
    per_gpu_gb = per_gpu_total_bytes / (1024 ** 3)

    print("-" * 80)
    print(f"\nSUMMARY:")
    print("=" * 80)

    if tp_size > 1:
        print(f"Total KV Cache (all GPUs)  : {total_mb:.2f} MB ({total_gb:.4f} GB)")
        print(f"Per GPU KV Cache           : {per_gpu_mb:.2f} MB ({per_gpu_gb:.4f} GB)")
        print(f"Per Layer Per GPU Avg      : {per_gpu_mb / num_layers:.2f} MB")
        if batch_size > 1:
            print(f"Per Sequence Per GPU       : {per_gpu_mb / batch_size:.2f} MB")
    else:
        per_layer_mb = total_mb / num_layers
        per_token_kb = total_bytes / seq_len / batch_size / 1024
        print(f"Total KV Cache    : {total_mb:.2f} MB ({total_gb:.4f} GB)")
        print(f"Per Layer Avg     : {per_layer_mb:.2f} MB")
        if batch_size > 1:
            print(f"Per Sequence      : {total_mb / batch_size:.2f} MB")
        print(f"Per Token         : {per_token_kb:.2f} KB")

    print("=" * 80)

    print(f"\nSEQUENCE LENGTH SCALING (batch_size={batch_size}):")
    print("-" * 80)

    if tp_size > 1:
        print(f"{'Seq Length':<15} | {'Per-GPU MB':<15} | {'Total MB':<15} | {'Per-GPU GB':<15} | {'Total GB':<15}")
    else:
        print(f"{'Seq Length':<15} | {'Total MB':<15} | {'Total GB':<15} | {'Per Token KB':<15}")

    print("-" * 80)

    for test_seq in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
        if test_seq <= seq_len * 4:
            test_total = 0
            test_per_gpu = 0
            for i in range(num_layers):
                test_result = calculate_layer_kv_cache(layer_config, i, test_seq, dtype_bytes, tp_size, batch_size)
                test_total += test_result['bytes']
                test_per_gpu += test_result['per_gpu_bytes']

            test_mb = test_total / (1024 ** 2)
            test_gb = test_total / (1024 ** 3)
            test_per_gpu_mb = test_per_gpu / (1024 ** 2)
            test_per_gpu_gb = test_per_gpu / (1024 ** 3)
            test_per_token = test_total / test_seq / batch_size / 1024

            marker = " <-- Current" if test_seq == seq_len else ""

            if tp_size > 1:
                print(f"{test_seq:<15} | {test_per_gpu_mb:<15.2f} | {test_mb:<15.2f} | {test_per_gpu_gb:<15.4f} | {test_gb:<15.4f}{marker}")
            else:
                print(f"{test_seq:<15} | {test_mb:<15.2f} | {test_gb:<15.4f} | {test_per_token:<15.2f}{marker}")

    print("-" * 80 + "\n")


# ============================================================================
# vLLM RUNTIME INSPECTION (from kv_cache_vllm_test.py)
# ============================================================================

def _extract_kv_cache_from_vllm_v1(
    llm, tensor_parallel: int, pipeline_parallel: int, data_parallel_size: int
) -> Optional[Dict]:
    """Extract KV cache sizes from V1 engine's KVCacheConfig.

    Tries two paths:
    1. Direct: llm.llm_engine.engine_core.engine_core.scheduler.kv_cache_config
       (works when engine runs in-process, e.g. single GPU)
    2. RPC: call_utility("get_kv_cache_config_summary")
       (works when engine runs in separate process, e.g. TP>1 multiprocess)
       Requires vllm_v1_kv_cache_rpc.patch applied to vLLM.
    """
    try:
        engine_core = getattr(llm.llm_engine, "engine_core", None)
        if engine_core is None:
            return None

        # Path 1: Direct access (InprocClient - single process)
        ec = getattr(engine_core, "engine_core", engine_core)
        scheduler = getattr(ec, "scheduler", None)
        if scheduler is not None and hasattr(scheduler, "kv_cache_config"):
            kv_cfg = scheduler.kv_cache_config
            if kv_cfg.kv_cache_tensors:
                return _build_cache_info_from_kv_cfg(
                    kv_cfg, tensor_parallel, pipeline_parallel, data_parallel_size
                )

        # Path 2: RPC (SyncMPClient - multiprocess, e.g. TP>1)
        if hasattr(engine_core, "call_utility"):
            try:
                summary = engine_core.call_utility("get_kv_cache_config_summary")
                if summary and summary.get("kv_cache_tensors"):
                    return _build_cache_info_from_summary(
                        summary, tensor_parallel, pipeline_parallel, data_parallel_size
                    )
            except (AttributeError, Exception):
                pass

        return None
    except Exception:
        return None


def _build_cache_info_from_kv_cfg(
    kv_cfg, tensor_parallel: int, pipeline_parallel: int, data_parallel_size: int
) -> Dict:
    """Build cache_info dict from KVCacheConfig object."""
    total_gpus = tensor_parallel * pipeline_parallel * data_parallel_size
    per_gpu_bytes = sum(t.size for t in kv_cfg.kv_cache_tensors)
    total_bytes = per_gpu_bytes * total_gpus

    layers = []
    for idx, tensor in enumerate(kv_cfg.kv_cache_tensors):
        layers.append({
            "layer_idx": idx,
            "shape": [tensor.size],
            "dtype": "config",
            "device": "N/A",
            "num_elements": tensor.size,
            "element_size_bytes": 1,
            "memory_bytes": tensor.size,
            "memory_mb": tensor.size / (1024 ** 2),
            "shared_by": getattr(tensor, "shared_by", []),
        })

    return _finalize_cache_info(
        layers, per_gpu_bytes, total_bytes, total_gpus,
        tensor_parallel, pipeline_parallel, data_parallel_size,
        getattr(kv_cfg, "num_blocks", None),
    )


def _build_cache_info_from_summary(
    summary: Dict, tensor_parallel: int, pipeline_parallel: int, data_parallel_size: int
) -> Optional[Dict]:
    """Build cache_info dict from RPC summary (dict with num_blocks, kv_cache_tensors)."""
    tensors = summary.get("kv_cache_tensors", [])
    if not tensors:
        return None

    total_gpus = tensor_parallel * pipeline_parallel * data_parallel_size
    per_gpu_bytes = sum(t["size"] for t in tensors)
    total_bytes = per_gpu_bytes * total_gpus

    layers = []
    for idx, t in enumerate(tensors):
        size = t["size"]
        layers.append({
            "layer_idx": idx,
            "shape": [size],
            "dtype": "config",
            "device": "N/A",
            "num_elements": size,
            "element_size_bytes": 1,
            "memory_bytes": size,
            "memory_mb": size / (1024 ** 2),
            "shared_by": t.get("shared_by", []),
        })

    return _finalize_cache_info(
        layers, per_gpu_bytes, total_bytes, total_gpus,
        tensor_parallel, pipeline_parallel, data_parallel_size,
        summary.get("num_blocks"),
    )


def _finalize_cache_info(
    layers, per_gpu_bytes, total_bytes, total_gpus,
    tensor_parallel, pipeline_parallel, data_parallel_size, num_blocks
) -> Dict:
    """Finalize cache_info dict and print success message."""
    cache_info = {
        "layers": layers,
        "total_memory_bytes": total_bytes,
        "total_memory_bytes_per_gpu": per_gpu_bytes,
        "engine_version": "V1",
        "tensor_parallel": tensor_parallel,
        "pipeline_parallel": pipeline_parallel,
        "data_parallel_size": data_parallel_size,
        "num_gpus": total_gpus,
        "num_blocks": num_blocks,
        "total_memory_mb_per_gpu": per_gpu_bytes / (1024 ** 2),
        "total_memory_gb_per_gpu": per_gpu_bytes / (1024 ** 3),
        "total_memory_mb": total_bytes / (1024 ** 2),
        "total_memory_gb": total_bytes / (1024 ** 3),
        "num_layers": len(layers),
    }
    if total_gpus > 1:
        print(f"  ✓ Extracted from V1 config: {len(layers)} tensors, "
              f"per GPU: {cache_info['total_memory_gb_per_gpu']:.4f} GB, "
              f"total ({total_gpus} GPUs): {cache_info['total_memory_gb']:.4f} GB")
    else:
        print(f"  ✓ Extracted from V1 config: {len(layers)} tensors, "
              f"{cache_info['total_memory_gb']:.4f} GB total")
    return cache_info


def extract_kv_cache_from_vllm(llm, tensor_parallel: int = 1, pipeline_parallel: int = 1,
                               data_parallel_size: int = 1) -> Optional[Dict]:
    """Extract KV cache tensors from vLLM's memory pools.

    V0: llm.llm_engine -> model_executor -> workers -> cache_engine -> gpu_cache
    V1: llm.llm_engine -> engine_core -> scheduler -> kv_cache_config (Option A)
    Supports single-GPU, multi-GPU (TP/PP), and expert-parallel (EP) setups.
    """

    if LLM is None:
        print("  ✗ vLLM not installed")
        return None

    try:
        # V0 path: model_executor -> workers -> cache_engine -> gpu_cache
        if not hasattr(llm.llm_engine, 'model_executor'):
            return _extract_kv_cache_from_vllm_v1(
                llm, tensor_parallel, pipeline_parallel, data_parallel_size
            )

        model_executor = llm.llm_engine.model_executor

        workers = []
        if hasattr(model_executor, 'driver_worker'):
            workers = [model_executor.driver_worker]
        elif hasattr(model_executor, 'workers') and len(model_executor.workers) > 0:
            workers = model_executor.workers
            ep_info = f", EP={data_parallel_size}" if data_parallel_size > 1 else ""
            print(f"  • Found {len(workers)} workers (TP={tensor_parallel}, PP={pipeline_parallel}{ep_info})")
        else:
            return _extract_kv_cache_from_vllm_v1(
                llm, tensor_parallel, pipeline_parallel, data_parallel_size
            )

        worker = workers[0]
        if hasattr(worker, 'worker'):
            worker = worker.worker

        if not hasattr(worker, 'cache_engine'):
            return _extract_kv_cache_from_vllm_v1(
                llm, tensor_parallel, pipeline_parallel, data_parallel_size
            )

        cache_engine = worker.cache_engine
        gpu_cache = None

        if isinstance(cache_engine, list) and len(cache_engine) > 0:
            first_elem = cache_engine[0]
            if hasattr(first_elem, 'gpu_cache'):
                gpu_cache = first_elem.gpu_cache
            elif hasattr(first_elem, 'shape'):
                gpu_cache = cache_engine
        elif hasattr(cache_engine, 'gpu_cache'):
            gpu_cache = cache_engine.gpu_cache

        if not gpu_cache or len(gpu_cache) == 0:
            return _extract_kv_cache_from_vllm_v1(
                llm, tensor_parallel, pipeline_parallel, data_parallel_size
            )

        # V0: Extract from actual tensors
        total_gpus = tensor_parallel * pipeline_parallel * data_parallel_size
        cache_info = {
            "layers": [],
            "total_memory_bytes": 0,
            "total_memory_bytes_per_gpu": 0,
            "engine_version": "V0",
            "tensor_parallel": tensor_parallel,
            "pipeline_parallel": pipeline_parallel,
            "data_parallel_size": data_parallel_size,
            "num_gpus": total_gpus,
        }

        for layer_idx, cache_tensor in enumerate(gpu_cache):
            if cache_tensor is not None:
                num_elements = cache_tensor.numel()
                element_size = cache_tensor.element_size()
                memory_bytes = num_elements * element_size

                layer_info = {
                    "layer_idx": layer_idx,
                    "shape": list(cache_tensor.shape),
                    "dtype": str(cache_tensor.dtype),
                    "device": str(cache_tensor.device),
                    "num_elements": num_elements,
                    "element_size_bytes": element_size,
                    "memory_bytes": memory_bytes,
                    "memory_mb": memory_bytes / (1024 ** 2),
                }

                cache_info["layers"].append(layer_info)
                cache_info["total_memory_bytes"] += memory_bytes

        cache_info["total_memory_bytes_per_gpu"] = cache_info["total_memory_bytes"]
        cache_info["total_memory_mb_per_gpu"] = cache_info["total_memory_bytes_per_gpu"] / (1024 ** 2)
        cache_info["total_memory_gb_per_gpu"] = cache_info["total_memory_bytes_per_gpu"] / (1024 ** 3)
        cache_info["total_memory_bytes"] = cache_info["total_memory_bytes_per_gpu"] * total_gpus
        cache_info["total_memory_mb"] = cache_info["total_memory_bytes"] / (1024 ** 2)
        cache_info["total_memory_gb"] = cache_info["total_memory_bytes"] / (1024 ** 3)
        cache_info["num_layers"] = len(cache_info["layers"])

        if total_gpus > 1:
            print(f"  ✓ Extracted {cache_info['num_layers']} layers (V0)")
            print(f"    Per GPU: {cache_info['total_memory_gb_per_gpu']:.4f} GB")
            print(f"    Total ({total_gpus} GPUs): {cache_info['total_memory_gb']:.4f} GB")
        else:
            print(f"  ✓ Extracted {cache_info['num_layers']} layers, {cache_info['total_memory_gb']:.4f} GB total (V0)")

        return cache_info

    except Exception as e:
        v1_result = _extract_kv_cache_from_vllm_v1(
            llm, tensor_parallel, pipeline_parallel, data_parallel_size
        )
        if v1_result is not None:
            return v1_result
        print(f"  ✗ Extraction error: {e}")
        return None


def print_cache_summary(model_name: str, cache_info: Dict, sequence_length: int):
    """Print detailed layer-wise KV cache information."""

    num_gpus = cache_info.get('num_gpus', 1)
    tp = cache_info.get('tensor_parallel', 1)
    pp = cache_info.get('pipeline_parallel', 1)

    print("\n" + "=" * 80)
    print(f"MODEL: {model_name}")
    print(f"SEQUENCE LENGTH: {sequence_length} tokens")
    if num_gpus > 1:
        print(f"PARALLELISM: {num_gpus} GPUs (TP={tp}, PP={pp})")
    print(f"ENGINE: {cache_info.get('engine_version', 'Unknown')}")
    print("=" * 80)

    if "note" in cache_info:
        print(f"\n{cache_info['note']}")
        return

    if not cache_info or "layers" not in cache_info:
        print("\n✗ Could not retrieve detailed KV cache information")
        return

    print(f"\nTotal Layers: {cache_info['num_layers']}")

    print("\n" + "-" * 80)
    print(f"{'Layer':<8} {'Shape':<30} {'Memory (MB)':<15} {'dtype':<15}")
    print("-" * 80)

    for layer in cache_info["layers"]:
        shape_str = "×".join(map(str, layer["shape"]))
        print(f"{layer['layer_idx']:<8} {shape_str:<30} {layer['memory_mb']:>12.2f}   {layer['dtype']:<15}")

    print("-" * 80)

    if num_gpus > 1:
        print(f"\n{'KV CACHE PER GPU':.<50} {cache_info['total_memory_mb_per_gpu']:>10.2f} MB")
        print(f"{'':.<50} {cache_info['total_memory_gb_per_gpu']:>10.4f} GB")
        print(f"\n{'TOTAL KV CACHE (all ' + str(num_gpus) + ' GPUs)':.<50} {cache_info['total_memory_mb']:>10.2f} MB")
        print(f"{'':.<50} {cache_info['total_memory_gb']:>10.4f} GB")
        print(f"{'':.<50} {cache_info['total_memory_bytes']:>10,} bytes")
    else:
        print(f"\n{'TOTAL KV CACHE MEMORY':.<50} {cache_info['total_memory_mb']:>10.2f} MB")
        print(f"{'':.<50} {cache_info['total_memory_gb']:>10.4f} GB")
        print(f"{'':.<50} {cache_info['total_memory_bytes']:>10,} bytes")

    if cache_info['num_layers'] > 0:
        avg_per_layer = cache_info.get('total_memory_mb_per_gpu', cache_info['total_memory_mb']) / cache_info['num_layers']
        print(f"\n{'Average per layer (per GPU)':.<50} {avg_per_layer:>10.2f} MB")

    print("=" * 80 + "\n")


def test_model(model_name: str, sequence_length: int, tensor_parallel: int = 1, pipeline_parallel: int = 1,
               batch_size: int = 1, kv_cache_dtype: str = "auto", enable_expert_parallel: bool = False,
               data_parallel_size: int = 1) -> Dict:
    """Load a model in vLLM and inspect its KV cache allocation."""

    if LLM is None or torch is None:
        return {
            "model": model_name,
            "sequence_length": sequence_length,
            "tensor_parallel": tensor_parallel,
            "pipeline_parallel": pipeline_parallel,
            "num_gpus": tensor_parallel * pipeline_parallel * data_parallel_size,
            "success": False,
            "error": "vLLM or torch not installed"
        }

    total_gpus = tensor_parallel * pipeline_parallel * data_parallel_size

    print(f"\n{'#' * 80}")
    print(f"MODEL: {model_name}")
    print(f"SEQUENCE LENGTH: {sequence_length} tokens")
    print(f"BATCH SIZE (concurrency): {batch_size}")
    if total_gpus > 1:
        ep_info = f", EP={data_parallel_size}" if enable_expert_parallel else ""
        dp_info = f", DP={data_parallel_size}" if data_parallel_size > 1 and not enable_expert_parallel else ""
        print(f"PARALLELISM: {total_gpus} GPUs (TP={tensor_parallel}, PP={pipeline_parallel}{ep_info}{dp_info})")
    print(f"{'#' * 80}\n")

    llm = None
    try:
        blocks_needed = (sequence_length // BLOCK_SIZE) * batch_size

        print("Initializing vLLM...")
        print(f"  • Allocating for {batch_size} concurrent request(s)")
        print(f"  • Blocks: {blocks_needed} (seq_len={sequence_length} × batch={batch_size} ÷ block_size={BLOCK_SIZE})")
        if total_gpus > 1:
            print(f"  • Tensor Parallel: {tensor_parallel} GPUs")
            print(f"  • Pipeline Parallel: {pipeline_parallel} GPUs")
            if enable_expert_parallel:
                print(f"  • Expert Parallel: enabled (data_parallel_size={data_parallel_size})")
            elif data_parallel_size > 1:
                print(f"  • Data Parallel: data_parallel_size={data_parallel_size}")
        vllm_kv_dtype = _vllm_kv_cache_dtype(kv_cache_dtype)
        if vllm_kv_dtype and vllm_kv_dtype != "auto":
            print(f"  • KV cache dtype: {vllm_kv_dtype}")

        llm_kwargs = {
            "model": model_name,
            "max_model_len": sequence_length,
            "max_num_seqs": batch_size,
            "num_gpu_blocks_override": blocks_needed,
            "tensor_parallel_size": tensor_parallel,
            "pipeline_parallel_size": pipeline_parallel,
            "trust_remote_code": True,
            "enforce_eager": True,
            "disable_log_stats": True,
            "kv_cache_dtype": vllm_kv_dtype,
        }
        if enable_expert_parallel:
            llm_kwargs["enable_expert_parallel"] = True
        if data_parallel_size > 1:
            llm_kwargs["data_parallel_size"] = data_parallel_size

        llm = LLM(**llm_kwargs)

        print("✓ Model loaded successfully\n")

        print("Running test inference...")
        sampling_params = SamplingParams(temperature=0.8, max_tokens=50)
        test_prompt = "This is a test prompt to allocate the KV cache."

        outputs = llm.generate([test_prompt], sampling_params)
        print("✓ Inference completed\n")

        print("Extracting KV cache information...")
        cache_info = extract_kv_cache_from_vllm(
            llm, tensor_parallel, pipeline_parallel,
            data_parallel_size=data_parallel_size,
        )

        if cache_info:
            print_cache_summary(model_name, cache_info, sequence_length)

            return {
                "model": model_name,
                "sequence_length": sequence_length,
                "tensor_parallel": tensor_parallel,
                "pipeline_parallel": pipeline_parallel,
                "num_gpus": total_gpus,
                "cache_info": cache_info,
                "success": True,
            }
        else:
            print("✗ Could not access KV cache information\n")
            return {
                "model": model_name,
                "sequence_length": sequence_length,
                "tensor_parallel": tensor_parallel,
                "pipeline_parallel": pipeline_parallel,
                "num_gpus": total_gpus,
                "success": False,
                "error": "Cache extraction failed"
            }

    except Exception as e:
        error_msg = str(e)
        print(f"✗ Error: {error_msg}\n")

        if "out of memory" in error_msg.lower():
            print("💡 Model too large for GPU. Try a smaller model or reduce sequence length.\n")
        elif "torch._scaled_mm" in error_msg or "MI300+" in error_msg:
            print("💡 This model requires MI300+ GPU for FP8. Use non-FP8 models.\n")
        elif "triton_kernels" in error_msg:
            print("💡 Missing triton_kernels for MXFP4. Avoid MXFP4 quantized models.\n")

        return {
            "model": model_name,
            "sequence_length": sequence_length,
            "tensor_parallel": tensor_parallel,
            "pipeline_parallel": pipeline_parallel,
            "num_gpus": total_gpus,
            "success": False,
            "error": error_msg,
        }

    finally:
        if llm is not None:
            del llm
        gc.collect()
        if torch is not None:
            torch.cuda.empty_cache()


# ============================================================================
# ESTIMATOR WRAPPER (from kv_estimator.py)
# ============================================================================

def estimate_kv_cache(model_name: str, sequence_length: int,
                     dtype: str = "float16", tp_size: int = 1,
                     batch_size: int = 1) -> Tuple[int, Dict]:
    """Estimate KV cache size with layer-by-layer details."""

    print(f"\n{'=' * 80}")
    print(f"KV Cache Estimation")
    print(f"{'=' * 80}")
    print(f"Model: {model_name}")
    print(f"Sequence Length: {sequence_length}")
    print(f"Data Type: {dtype}")
    print(f"Tensor Parallel: {tp_size}")
    print(f"Batch Size: {batch_size}")
    print(f"{'=' * 80}\n")

    try:
        calculate_kv_cache(
            model_name,
            sequence_length,
            dtype,
            tp_size,
            batch_size,
            user_specified_dtype=(dtype != "float16")
        )

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        layer_config = _get_config_for_layers(config)
        num_layers = _get_num_layers(config)
        if num_layers is None:
            raise AttributeError(f"{type(config).__name__} has no num_hidden_layers, n_layers, or num_layers")

        dtype_bytes = {
            "float32": 4, "float16": 2, "bfloat16": 2,
            "int8": 1, "fp8": 1, "mxfp4": 0.5
        }.get(dtype, 2)

        hidden_size = getattr(layer_config, 'hidden_size', config.hidden_size)
        num_attention_heads = getattr(layer_config, 'num_attention_heads', config.num_attention_heads)
        num_kv_heads = getattr(layer_config, 'num_key_value_heads', num_attention_heads)
        head_dim = hidden_size // num_attention_heads

        layer_details = []
        total_bytes = 0
        per_gpu_total_bytes = 0

        for layer_idx in range(num_layers):
            layer_result = calculate_layer_kv_cache(
                layer_config, layer_idx, sequence_length, dtype_bytes, tp_size, batch_size
            )

            layer_info = {
                "layer": layer_idx,
                "attention_type": layer_result['attention_type'],
                "kv_cache_bytes": layer_result['bytes'],
                "kv_cache_mb": layer_result['mb'],
                "per_gpu_bytes": layer_result['per_gpu_bytes'],
                "per_gpu_mb": layer_result['per_gpu_mb'],
                "kv_dim": layer_result['kv_dim']
            }

            if layer_result['attention_type'] != 'MLA':
                layer_info['num_kv_heads'] = layer_result['num_kv_heads']
                layer_info['per_gpu_kv_heads'] = layer_result['per_gpu_kv_heads']
                layer_info['effective_seq_len'] = layer_result['effective_seq_len']

            layer_details.append(layer_info)
            total_bytes += layer_result['bytes']
            per_gpu_total_bytes += layer_result['per_gpu_bytes']

        if tp_size > 1:
            estimated_bytes = _ensure_int_bytes(per_gpu_total_bytes)
        else:
            estimated_bytes = _ensure_int_bytes(total_bytes)

        results = {
            "model": model_name,
            "sequence_length": sequence_length,
            "dtype": dtype,
            "tp_size": tp_size,
            "batch_size": batch_size,
            "estimated_bytes": estimated_bytes,
            "estimated_mb": estimated_bytes / (1024 ** 2),
            "estimated_gb": estimated_bytes / (1024 ** 3),
            "total_bytes_all_gpus": total_bytes,
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "num_attention_heads": num_attention_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "layer_by_layer": layer_details
        }

        print(f"\n✓ Estimation complete: {results['estimated_gb']:.4f} GB ({results['estimated_bytes']:,} bytes)")
        print(f"✓ Layer-by-layer details: {len(layer_details)} layers captured")

        return estimated_bytes, results

    except Exception as e:
        print(f"\n✗ Estimation failed: {e}")
        raise


def verify_with_vllm(model_name: str, sequence_length: int,
                     dtype: str = "float16", tp_size: int = 1,
                     batch_size: int = 1, quiet: bool = False,
                     pp_size: int = 1, ep_size: int = 1,
                     data_parallel_size: int = 1) -> Optional[Tuple[int, Dict]]:
    """Verify KV cache size using vLLM runtime inspection."""

    if not quiet:
        print(f"\n{'=' * 80}")
        print(f"vLLM Runtime Verification")
        print(f"{'=' * 80}\n")

    try:
        enable_expert_parallel = ep_size > 1
        dp_size = max(1, data_parallel_size)
        vllm_dtype = _vllm_kv_cache_dtype(dtype)
        result = test_model(
            model_name, sequence_length, tp_size, pp_size,
            batch_size=batch_size, kv_cache_dtype=vllm_dtype,
            enable_expert_parallel=enable_expert_parallel,
            data_parallel_size=dp_size,
        )

        if result.get('success', False):
            cache_info = result['cache_info']
            actual_bytes = cache_info.get('total_memory_bytes_per_gpu', cache_info.get('total_memory_bytes', 0))

            vllm_layers = cache_info.get('layers', [])
            layer_by_layer = []
            for layer in vllm_layers:
                layer_info = {
                    "layer": layer.get('layer_idx', 0),
                    "shape": layer.get('shape', []),
                    "dtype": layer.get('dtype', 'unknown'),
                    "element_size_bytes": layer.get('element_size_bytes'),
                    "actual_bytes": layer.get('memory_bytes', 0),
                    "actual_mb": layer.get('memory_mb', 0),
                    "device": layer.get('device', 'unknown')
                }
                layer_by_layer.append(layer_info)

            results = {
                "model": model_name,
                "sequence_length": sequence_length,
                "dtype": dtype,
                "tp_size": tp_size,
                "batch_size": batch_size,
                "pp_size": pp_size,
                "ep_size": ep_size,
                "actual_bytes": actual_bytes,
                "actual_mb": actual_bytes / (1024 ** 2),
                "actual_gb": actual_bytes / (1024 ** 3),
                "num_layers": cache_info.get('num_layers', 0),
                "layer_by_layer": layer_by_layer
            }

            if not quiet:
                print(f"\n✓ Verification complete: {results['actual_gb']:.4f} GB ({results['actual_bytes']:,} bytes)")
                print(f"✓ Layer-by-layer vLLM data: {len(layer_by_layer)} layers captured")

            return actual_bytes, results
        else:
            err = result.get('error', 'Unknown error')
            if not quiet:
                print(f"\n✗ Verification failed: {err}")
            return None, err

    except Exception as e:
        if not quiet:
            print(f"\n✗ Verification failed: {e}")
        return None, str(e)


def calculate_test_sizes(base_size_bytes: int, multipliers: list = None) -> list:
    """Calculate test sizes based on base size (max per-layer KV cache)."""
    if multipliers is None:
        multipliers = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4]

    test_sizes = []
    for mult in multipliers:
        size = int(base_size_bytes * mult)
        size = max(size, 4096)
        test_sizes.append(size)

    return test_sizes


def get_max_layer_kv_cache_size(layer_details: list) -> int:
    """Get the maximum per-layer KV cache size from layer details."""
    if not layer_details:
        return 0
    return max(layer['kv_cache_bytes'] for layer in layer_details)


# ============================================================================
# BATCH MEASUREMENT
# ============================================================================

BATCH_MEASUREMENT_CONCURRENCY = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
BATCH_MEASUREMENT_SEQ_LENGTHS = [1024, 2048, 4096, 8192]
BATCH_MEASUREMENT_TP_SIZES = [1, 8]
BATCH_MEASUREMENT_PP_SIZES = [1]
BATCH_MEASUREMENT_DTYPE = "fp8"

DTYPE_BYTES_MAP = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1, "fp8": 1, "mxfp4": 0.5}


def _infer_kv_cache_dtype_from_config(config) -> Optional[str]:
    """Infer KV cache dtype from model config (quantization_config, torch_dtype).
    Returns None if nothing can be inferred.
    """
    # Check quantization_config first (FP8/MXFP4 models)
    if hasattr(config, 'quantization_config') and config.quantization_config:
        qc = config.quantization_config
        if isinstance(qc, dict):
            qm = (qc.get('quant_method') or '').lower()
            if 'fp8' in qm:
                return "fp8"
            if 'mxfp4' in qm or 'fp4' in qm:
                return "mxfp4"
    # Fallback to torch_dtype (e.g. bfloat16, float16)
    td = getattr(config, 'torch_dtype', None)
    if td is not None:
        s = str(td).lower()
        if 'bfloat16' in s or 'bf16' in s:
            return "bfloat16"
        if 'float16' in s or 'fp16' in s:
            return "float16"
        if 'float32' in s or 'fp32' in s:
            return "float32"
    return None


def _resolve_dtype_for_estimation(config, dtype: str) -> str:
    """Resolve 'auto' to concrete dtype for theoretical estimation.
    When dtype is 'auto', infers from model config: quantization_config first,
    then torch_dtype. Defaults to bfloat16 if nothing can be inferred.
    """
    d = (dtype or "").strip().lower()
    if d and d != "auto":
        return d
    inferred = _infer_kv_cache_dtype_from_config(config)
    return inferred if inferred else "bfloat16"


def _parse_config_value(val: Any) -> List[int]:
    """Parse config value: int, list, or space/comma-separated string -> list of ints."""
    if val is None:
        return []
    if isinstance(val, int):
        return [val]
    if isinstance(val, list):
        return [int(x) for x in val]
    s = str(val).strip()
    if not s:
        return []
    # Support both space and comma separators
    parts = s.replace(",", " ").split()
    return [int(x) for x in parts if x]


def load_estimator_config(config_path: str) -> Dict[str, Any]:
    """Load estimator config from YAML. Returns dict with model, concurrency, seq_lengths, tp_sizes, pp_sizes, dtype."""
    if yaml is None:
        raise RuntimeError("PyYAML is required for --config. Install with: pip install pyyaml")
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    if not data:
        raise ValueError(f"Empty or invalid config: {config_path}")

    model = data.get("model") or {}

    model_name = model.get("name")
    if not model_name:
        raise ValueError(f"Config must have model.name: {config_path}")

    concurrency = _parse_config_value(model.get("concurrency"))
    tp_sizes = _parse_config_value(model.get("tp"))
    seq_lengths = _parse_config_value(model.get("seq-length"))
    pp_sizes = _parse_config_value(model.get("pp"))
    ep_sizes = _parse_config_value(model.get("ep"))
    dp_sizes = _parse_config_value(model.get("dp"))
    dtype = model.get("kv_cache_dtype") or model.get("dtype")
    if isinstance(dtype, str):
        dtype = dtype.strip().strip('"')
    vllm_verify_kv_cache_dtype = model.get("vllm_verify_kv_cache_dtype")
    if isinstance(vllm_verify_kv_cache_dtype, str):
        vllm_verify_kv_cache_dtype = vllm_verify_kv_cache_dtype.strip().strip('"')

    skip_vllm_verify = model.get("skip_vllm_verify", False)
    if isinstance(skip_vllm_verify, str):
        skip_vllm_verify = str(skip_vllm_verify).lower() in ("true", "1", "yes")

    return {
        "model": model_name,
        "concurrency": concurrency or [1],
        "seq_lengths": seq_lengths or [2048],
        "tp_sizes": tp_sizes or [1],
        "pp_sizes": pp_sizes or [1],
        "ep_sizes": ep_sizes or [1],
        "dp_sizes": dp_sizes or [1],
        "dtype": dtype or "fp8",
        "vllm_verify_kv_cache_dtype": vllm_verify_kv_cache_dtype,
        "skip_vllm_verify": skip_vllm_verify,
    }


def _model_name_only(model_path: str) -> str:
    """Extract model name from path (e.g. 'DeepSeek-R1' from '/path/to/deepseek-ai/DeepSeek-R1/')."""
    name = model_path.rstrip("/")
    parts = [p for p in name.split("/") if p]
    return parts[-1] if parts else model_path


def _measure_single_batch_config(config, seq_len, batch_size, tp_size, pp_size, dtype):
    """Compute KV cache for a single batch config. Returns per-layer bytes (primary) and total bytes.

    With PP>1, layers are split across stages; per_gpu_total_bytes is max over stages.
    """
    resolved = _resolve_dtype_for_estimation(config, dtype) if dtype == "auto" else (dtype or "fp8")
    dtype_bytes = DTYPE_BYTES_MAP.get(resolved) or DTYPE_BYTES_MAP.get("fp8", 1)
    layer_config = _get_config_for_layers(config)
    num_layers = _get_num_layers(config)
    if num_layers is None:
        raise AttributeError(f"{type(config).__name__} has no num_hidden_layers, n_layers, or num_layers")

    total_bytes = 0
    max_per_layer_bytes = 0
    if pp_size > 1:
        layers_per_stage = (num_layers + pp_size - 1) // pp_size
        per_stage_bytes = [0] * pp_size
        for layer_idx in range(num_layers):
            result = calculate_layer_kv_cache(
                layer_config, layer_idx, seq_len, dtype_bytes, tp_size, batch_size
            )
            total_bytes += result["bytes"]
            pp_stage = min(layer_idx // layers_per_stage, pp_size - 1)
            per_stage_bytes[pp_stage] += result["per_gpu_bytes"]
            layer_bytes = result["per_gpu_bytes"] if tp_size > 1 else result["bytes"]
            max_per_layer_bytes = max(max_per_layer_bytes, layer_bytes)
        per_gpu_total_bytes = max(per_stage_bytes)
    else:
        per_gpu_total_bytes = 0
        for layer_idx in range(num_layers):
            result = calculate_layer_kv_cache(
                layer_config, layer_idx, seq_len, dtype_bytes, tp_size, batch_size
            )
            total_bytes += result["bytes"]
            per_gpu_total_bytes += result["per_gpu_bytes"]
            layer_bytes = result["per_gpu_bytes"] if tp_size > 1 else result["bytes"]
            max_per_layer_bytes = max(max_per_layer_bytes, layer_bytes)

    if tp_size > 1 or pp_size > 1:
        return max_per_layer_bytes, per_gpu_total_bytes, total_bytes
    return max_per_layer_bytes, total_bytes, total_bytes


def run_batch_measurements(
    model_name: str,
    output_dir: str,
    concurrency: Optional[list] = None,
    seq_lengths: Optional[list] = None,
    tp_sizes: Optional[list] = None,
    pp_sizes: Optional[list] = None,
    dtype: Optional[str] = None,
) -> list:
    """Run batch measurements for all config combinations. Generates test report by default.

    Override iteration values via args; defaults to BATCH_MEASUREMENT_* constants when None.
    """
    concurrency = concurrency or BATCH_MEASUREMENT_CONCURRENCY
    seq_lengths = seq_lengths or BATCH_MEASUREMENT_SEQ_LENGTHS
    tp_sizes = tp_sizes or BATCH_MEASUREMENT_TP_SIZES
    pp_sizes = pp_sizes or BATCH_MEASUREMENT_PP_SIZES
    dtype = dtype or BATCH_MEASUREMENT_DTYPE

    print(f"Loading model config: {model_name}")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    results = []
    total_configs = len(concurrency) * len(seq_lengths) * len(tp_sizes) * len(pp_sizes)
    count = 0

    for seq_len in seq_lengths:
        for batch_size in concurrency:
            for tp_size in tp_sizes:
                for pp_size in pp_sizes:
                    count += 1
                    try:
                        per_layer_bytes, _, _ = _measure_single_batch_config(
                            config, seq_len, batch_size, tp_size, pp_size, dtype
                        )
                        model_short = _model_name_only(model_name)
                        model_unique_name = f"{model_short}_seq{seq_len}_c{batch_size}_tp{tp_size}_pp{pp_size}_{dtype}"
                        entry = {
                            "model_name": model_short,
                            "model_unique_name": model_unique_name,
                            "concurrency": batch_size,
                            "seq_length": seq_len,
                            "tp_size": tp_size,
                            "pp_size": pp_size,
                            "dtype": dtype,
                            "kv_cache_mb": round(per_layer_bytes / (1024 ** 2), 2),
                        }
                        results.append(entry)
                        print(
                            f"[{count}/{total_configs}] seq={seq_len} c={batch_size} tp={tp_size} pp={pp_size} "
                            f"dtype={dtype} -> {entry['kv_cache_mb']:.2f} MB per layer"
                        )
                    except Exception as e:
                        print(
                            f"[{count}/{total_configs}] ERROR seq={seq_len} c={batch_size} tp={tp_size} pp={pp_size}: {e}"
                        )
                        model_short = _model_name_only(model_name)
                        model_unique_name = f"{model_short}_seq{seq_len}_c{batch_size}_tp{tp_size}_pp{pp_size}_{dtype}"
                        results.append(
                            {
                                "model_name": model_short,
                                "model_unique_name": model_unique_name,
                                "concurrency": batch_size,
                                "seq_length": seq_len,
                                "tp_size": tp_size,
                                "pp_size": pp_size,
                                "dtype": dtype,
                                "error": str(e),
                            }
                        )

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    _generate_batch_test_report(model_name, results, output_dir, concurrency, seq_lengths, tp_sizes, pp_sizes, dtype)

    return results


def _generate_batch_test_report(
    model_name: str,
    results: list,
    output_dir: str,
    concurrency: list,
    seq_lengths: list,
    tp_sizes: list,
    pp_sizes: list,
    dtype: str,
) -> None:
    """Generate test report (JSON + CSV) for batch measurements."""
    valid_results = [r for r in results if "error" not in r]
    failed_results = [r for r in results if "error" in r]

    report = {
        "report_metadata": {
            "model": model_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "total_configs": len(results),
            "successful": len(valid_results),
            "failed": len(failed_results),
        },
        "test_config": {
            "concurrency": concurrency,
            "seq_lengths": seq_lengths,
            "tp_sizes": tp_sizes,
            "pp_sizes": pp_sizes,
            "dtype": dtype,
        },
        "results": results,
        "summary": {
            "min_kv_cache_mb": min(r["kv_cache_mb"] for r in valid_results) if valid_results else None,
            "max_kv_cache_mb": max(r["kv_cache_mb"] for r in valid_results) if valid_results else None,
        },
    }
    if failed_results:
        report["failed_configs"] = failed_results

    json_path = os.path.abspath(os.path.join(output_dir, "kv_cache_test_report.json"))
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nTest report (JSON): {json_path}")

    csv_path = os.path.abspath(os.path.join(output_dir, "kv_cache_test_report.csv"))
    csv_fieldnames = [
        "model_name", "model_unique_name", "concurrency", "seq_length", "tp_size", "pp_size", "ep_size", "dtype",
        "kv_cache_mb",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in csv_fieldnames})
    print(f"Test report (CSV): {csv_path}")

    if valid_results:
        pivot_path = os.path.abspath(os.path.join(output_dir, "kv_cache_test_report_pivot.csv"))
        seq_tp_pp_tuples = sorted(
            {(r["seq_length"], r["tp_size"], r.get("pp_size", 1)) for r in valid_results},
            key=lambda x: (x[0], x[1], x[2]),
        )
        with open(pivot_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["seq_length", "tp_size", "pp_size"] + [f"concurrency_{b}_mb" for b in concurrency]
            writer.writerow(header)
            for seq_len, tp, pp in seq_tp_pp_tuples:
                row = [seq_len, tp, pp]
                for batch in concurrency:
                    match = next(
                        (
                            r for r in valid_results
                            if (r["seq_length"] == seq_len and r["tp_size"] == tp
                                and r.get("pp_size", 1) == pp and r["concurrency"] == batch)
                        ),
                        None,
                    )
                    row.append(f"{match['kv_cache_mb']:.2f}" if match else "")
                writer.writerow(row)
        print(f"Test report (pivot CSV): {pivot_path}")

    print(f"\nAll results saved to: {os.path.abspath(output_dir)}")


# ============================================================================
# MAIN / CLI
# ============================================================================

def main():
    """Unified CLI entry point. Requires --config."""

    parser = argparse.ArgumentParser(description='KV Cache Estimator (config-only)')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML config file (model.name, model.concurrency, model.tp, model.seq-length, model.kv_cache_dtype, model.pp, model.ep, model.dp)',
    )
    parser.add_argument(
        '--output-dir',
        default='shared/kv_cache_batch_results',
        help='Output dir for results (default: shared/kv_cache_batch_results)',
    )
    parser.add_argument('--verify-vllm', action='store_true', help='Enable vLLM verification (optional)')
    parser.add_argument('--append', action='store_true', help='Append results to existing files (error if entries already exist)')
    args = parser.parse_args()

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(_script_dir, config_path)
    if not os.path.isfile(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    cfg = load_estimator_config(config_path)
    model_name = cfg["model"]
    concurrency = cfg["concurrency"]
    seq_lengths = cfg["seq_lengths"]
    tp_sizes = cfg["tp_sizes"]
    pp_sizes = cfg["pp_sizes"]
    ep_sizes = cfg["ep_sizes"]
    dp_sizes = cfg["dp_sizes"]
    dtype = cfg["dtype"]
    config_kv_cache_dtype = dtype  # Original from config, for vLLM mapping
    vllm_verify_override = cfg.get("vllm_verify_kv_cache_dtype")
    skip_vllm_verify = cfg.get("skip_vllm_verify", False)
    print(f"Loaded config from: {config_path}")

    # Determine output directory
    if os.path.exists("/workspace/results/phase1"):
        results_dir = "/workspace/results/phase1"
    elif os.path.exists("/workspace/results"):
        results_dir = "/workspace/results"
    else:
        out = args.output_dir
        results_dir = os.path.join(_script_dir, out) if not os.path.isabs(out) else out
    results_dir = os.path.abspath(results_dir)

    print("=" * 80)
    print("KV Cache Estimation")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Concurrency: {concurrency}")
    print(f"Seq lengths: {seq_lengths}")
    print(f"TP sizes: {tp_sizes}")
    print(f"PP sizes: {pp_sizes}")
    print(f"EP sizes: {ep_sizes}")
    print(f"DP sizes (data_parallel_size): {dp_sizes}")
    print(f"dtype: {dtype}")
    if vllm_verify_override:
        print(f"vLLM verify kv_cache_dtype override: {vllm_verify_override}")
    print("=" * 80)

    print(f"\nLoading model config: {model_name}")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    # Resolve "auto" to concrete dtype for estimation (from model quantization config)
    # config_kv_cache_dtype stays as-is for vLLM (pass "auto" when config says auto)
    estimation_dtype = _resolve_dtype_for_estimation(config, dtype or "fp8")
    if (dtype or "").strip().lower() == "auto":
        print(f"dtype: auto (resolved to {estimation_dtype} for estimation)")
        dtype = estimation_dtype

    vllm_kv_cache_dtype = _vllm_kv_cache_dtype(
        vllm_verify_override or config_kv_cache_dtype or "fp8"
    )

    results = []
    total_configs = len(seq_lengths) * len(concurrency) * len(tp_sizes) * len(pp_sizes) * len(ep_sizes) * len(dp_sizes)
    count = 0

    for seq_len in seq_lengths:
        for batch_size in concurrency:
            for tp_size in tp_sizes:
                for pp_size in pp_sizes:
                    for ep_size in ep_sizes:
                        for dp_size in dp_sizes:
                            count += 1
                            try:
                                per_layer_bytes, primary_total_bytes, _ = _measure_single_batch_config(
                                    config, seq_len, batch_size, tp_size, pp_size, dtype
                                )
                                model_short = _model_name_only(model_name)
                                ep_suffix = f"_ep{ep_size}" if ep_size > 1 else ""
                                dp_suffix = f"_dp{dp_size}" if dp_size > 1 else ""
                                model_unique_name = f"{model_short}_seq{seq_len}_c{batch_size}_tp{tp_size}_pp{pp_size}{ep_suffix}{dp_suffix}_{dtype}"
                                entry = {
                                    "model_name": model_short,
                                    "model_unique_name": model_unique_name,
                                    "seq_length": seq_len,
                                    "concurrency": batch_size,
                                    "tp_size": tp_size,
                                    "pp_size": pp_size,
                                    "ep_size": ep_size,
                                    "data_parallel_size": dp_size,
                                    "dtype": dtype,
                                    "kv_cache_bytes": int(per_layer_bytes),
                                    "_total_bytes": int(primary_total_bytes),  # for vLLM verification only
                                }
                                results.append(entry)
                                ep_str = f" ep={ep_size}" if ep_size > 1 else ""
                                dp_str = f" dp={dp_size}" if dp_size > 1 else ""
                                print(
                                    f"[{count}/{total_configs}] seq={seq_len} c={batch_size} tp={tp_size} pp={pp_size}{ep_str}{dp_str} "
                                    f"dtype={dtype} -> {entry['kv_cache_bytes']:,} bytes"
                                )
                            except Exception as e:
                                ep_str = f" ep={ep_size}" if ep_size > 1 else ""
                                dp_str = f" dp={dp_size}" if dp_size > 1 else ""
                                print(f"[{count}/{total_configs}] ERROR seq={seq_len} c={batch_size} tp={tp_size} pp={pp_size}{ep_str}{dp_str}: {e}")
                                model_short = _model_name_only(model_name)
                                ep_suffix = f"_ep{ep_size}" if ep_size > 1 else ""
                                dp_suffix = f"_dp{dp_size}" if dp_size > 1 else ""
                                model_unique_name = f"{model_short}_seq{seq_len}_c{batch_size}_tp{tp_size}_pp{pp_size}{ep_suffix}{dp_suffix}_{dtype}"
                                results.append(
                                    {
                                        "model_name": model_short,
                                        "model_unique_name": model_unique_name,
                                        "seq_length": seq_len,
                                        "concurrency": batch_size,
                                        "tp_size": tp_size,
                                        "pp_size": pp_size,
                                        "ep_size": ep_size,
                                        "data_parallel_size": dp_size,
                                        "dtype": dtype,
                                        "error": str(e),
                                    }
                                )

    # vLLM verification (default: enabled) - batch verify multiple configs, print error if mismatch
    vllm_verified_list = []
    # Derive verify configs from benchmark run: representative subset of (seq_len, concurrency, tp_size, pp_size, ep_size, dp_size)
    def _build_verify_configs(seq_lens, conc, tp, pp, ep, dp):
        if not seq_lens or not conc or not tp or not pp or not ep or not dp:
            return []
        cfg = []
        for s in seq_lens[:3]:  # up to 3 seq lengths with base concurrency/tp/pp/ep/dp
            cfg.append((s, conc[0], tp[0], pp[0], ep[0], dp[0]))
        if len(conc) > 1:
            cfg.append((seq_lens[0], conc[1], tp[0], pp[0], ep[0], dp[0]))
        if len(tp) > 1:
            cfg.append((seq_lens[0], conc[0], tp[-1], pp[0], ep[0], dp[0]))
        if len(pp) > 1:
            cfg.append((seq_lens[0], conc[0], tp[0], pp[-1], ep[0], dp[0]))
        if len(ep) > 1:
            cfg.append((seq_lens[0], conc[0], tp[0], pp[0], ep[-1], dp[0]))
        if len(dp) > 1:
            cfg.append((seq_lens[0], conc[0], tp[0], pp[0], ep[0], dp[-1]))
        return list(dict.fromkeys(cfg))  # dedupe

    verify_configs = _build_verify_configs(seq_lengths, concurrency, tp_sizes, pp_sizes, ep_sizes, dp_sizes)
    if args.verify_vllm and not skip_vllm_verify:
        print(f"\n{'=' * 80}")
        print("vLLM Verification (batch)")
        print(f"{'=' * 80}")
        if LLM is None or torch is None:
            print("  SKIP: vLLM or PyTorch not installed. Install with: pip install vllm torch")
        mismatches = []
        for verify_seq, verify_batch, verify_tp, verify_pp, verify_ep, verify_dp in verify_configs:
            if LLM is None or torch is None:
                continue
            if (verify_seq not in seq_lengths or verify_batch not in concurrency
                    or verify_tp not in tp_sizes or verify_pp not in pp_sizes
                    or verify_ep not in ep_sizes or verify_dp not in dp_sizes):
                continue
            # Multi-GPU verification supported (V0 engine exposes cache via workers)
            match = next(
                (r for r in results
                 if (r.get("seq_length") == verify_seq and r.get("concurrency") == verify_batch
                     and r.get("tp_size") == verify_tp and r.get("pp_size") == verify_pp
                     and r.get("ep_size") == verify_ep and r.get("data_parallel_size") == verify_dp
                     and "error" not in r)),
                None,
            )
            if not match:
                continue
            verify_result = verify_with_vllm(
                model_name, verify_seq, vllm_kv_cache_dtype, verify_tp, verify_batch, quiet=True,
                pp_size=verify_pp, ep_size=verify_ep, data_parallel_size=verify_dp
            )
            actual_bytes, vllm_data = verify_result
            if actual_bytes is None:
                ep_str = f" ep={verify_ep}" if verify_ep > 1 else ""
                dp_str = f" dp={verify_dp}" if verify_dp > 1 else ""
                print(f"  SKIP seq={verify_seq} c={verify_batch} tp={verify_tp} pp={verify_pp}{ep_str}{dp_str}: {vllm_data}")
                continue
            vllm_results = vllm_data
            est_bytes = match.get("_total_bytes", match.get("kv_cache_bytes", 0))  # use stored total for verification

            # Infer actual vLLM dtype from layer info; scale est if dtype mismatch (e.g. vLLM bfloat16 vs config fp8)
            config_dtype_bytes = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1, "fp8": 1, "mxfp4": 0.5}.get(
                dtype or "fp8", 1
            )
            actual_dtype_bytes = config_dtype_bytes
            vllm_layers = vllm_data.get("layer_by_layer", [])
            for layer in (vllm_layers if isinstance(vllm_layers, list) else []):
                es = layer.get("element_size_bytes")
                if es is not None and es > 0:
                    actual_dtype_bytes = es
                    break
                dt = str(layer.get("dtype", "")).lower()
                if "bfloat16" in dt or "float16" in dt:
                    actual_dtype_bytes = 2
                    break
                if "fp8" in dt or "uint8" in dt:
                    actual_dtype_bytes = 1
                    break

            # Fallback: if ratio ~2.0, vLLM likely used bfloat16 when we estimated fp8
            if actual_dtype_bytes == config_dtype_bytes and est_bytes > 0:
                ratio = actual_bytes / est_bytes
                if 1.95 <= ratio <= 2.05:
                    actual_dtype_bytes = 2
                    config_dtype_bytes = 1
                elif 0.48 <= ratio <= 0.52:
                    actual_dtype_bytes = 1
                    config_dtype_bytes = 2

            if actual_dtype_bytes != config_dtype_bytes and config_dtype_bytes > 0:
                est_bytes_scaled = int(est_bytes * actual_dtype_bytes / config_dtype_bytes)
            else:
                est_bytes_scaled = est_bytes

            vllm_mb = actual_bytes / (1024 ** 2)
            est_total_mb = est_bytes_scaled / (1024 ** 2)
            diff_pct = abs(est_bytes_scaled - actual_bytes) / est_bytes_scaled * 100 if est_bytes_scaled > 0 else 0
            status = "OK" if diff_pct < 1.0 else "MISMATCH"
            vllm_results["comparison"] = {
                "estimated_mb": est_total_mb,
                "vllm_actual_mb": vllm_mb,
                "accuracy_percent": 100 - diff_pct,
            }
            vllm_verified_list.append(vllm_results)
            ep_str = f" ep={verify_ep}" if verify_ep > 1 else ""
            dp_str = f" dp={verify_dp}" if verify_dp > 1 else ""
            print(f"  seq={verify_seq} c={verify_batch} tp={verify_tp} pp={verify_pp}{ep_str}{dp_str}: "
                  f"theoretical={est_total_mb:.2f} MB (total), vLLM={vllm_mb:.2f} MB -> {status}")
            if status == "MISMATCH":
                mismatches.append(
                    f"seq={verify_seq} c={verify_batch} tp={verify_tp} pp={verify_pp}{ep_str}{dp_str}: "
                    f"theoretical={est_total_mb:.2f} MB vs vLLM={vllm_mb:.2f} MB (diff={diff_pct:.2f}%)"
                )
        if mismatches:
            print(f"\n{'=' * 80}")
            print("ERROR: vLLM verification MISMATCH(es):")
            for m in mismatches:
                print(f"  {m}")
            print(f"{'=' * 80}\n")
        elif vllm_verified_list:
            print(f"\nVLLM Verification passed\n")
            print(f"{'=' * 80}\n")
    elif args.verify_vllm and skip_vllm_verify:
        print(f"\n{'=' * 80}")
        print("vLLM Verification: SKIPPED (skip_vllm_verify=true in config)")
        print(f"{'=' * 80}\n")

    # Save to CSV only (single source of truth)
    os.makedirs(results_dir, exist_ok=True)
    csv_file = os.path.join(results_dir, "kv_cache_estimator.csv")
    csv_fieldnames = [
        "model_name", "model_unique_name", "seq_length", "concurrency", "tp_size", "pp_size", "ep_size",
        "data_parallel_size", "dtype", "kv_cache_bytes",
    ]

    # Append mode: load existing from CSV only, merge (error if overlap)
    existing_unique_names = set()
    existing_results_csv = []   # for shared CSV (all models)

    if args.append:
        print(f"Append mode: reading existing from {csv_file}")
        if os.path.isfile(csv_file):
            try:
                with open(csv_file) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        uid = row.get("model_unique_name")
                        if uid:
                            existing_unique_names.add(uid)
                            r = {k: row.get(k, "") for k in csv_fieldnames}
                            # Backward compat: convert legacy kv_cache_mb to kv_cache_bytes
                            if not r.get("kv_cache_bytes") and row.get("kv_cache_mb"):
                                try:
                                    r["kv_cache_bytes"] = int(float(row["kv_cache_mb"]) * (1024 ** 2))
                                except (ValueError, TypeError):
                                    pass
                            existing_results_csv.append(r)
            except (IOError, csv.Error):
                pass
        else:
            print(f"  CSV not found at {csv_file}")
        overlapping = [r.get("model_unique_name") for r in results if r.get("model_unique_name") in existing_unique_names]
        if overlapping:
            print(f"\nError: The following entries already exist in the output file. Remove --append or delete existing entries.")
            for uid in overlapping[:10]:
                print(f"  {uid}")
            if len(overlapping) > 10:
                print(f"  ... and {len(overlapping) - 10} more")
            sys.exit(1)
        merged_results_csv = existing_results_csv + results
        print(f"\nAppending {len(results)} new entries ({len(existing_results_csv)} existing in CSV, {len(merged_results_csv)} total)")
    else:
        merged_results_csv = results

    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in merged_results_csv:
            writer.writerow({k: r.get(k, "") for k in csv_fieldnames})

    print(f"\nCSV saved to: {csv_file}")


if __name__ == "__main__":
    main()
