###############################################################################
#
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
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
# Llama-3.1-405B on gfx950 (MI355X). Companion to gfx950_llama3.1_405b.yml.
#
# Other configs here span nodes too - gfx950_llama3_8b.yml sets dcn_data_parallelism: -1,
# which replicates across nodes and drives an inter-node gradient all-reduce once per step.
# What is different here is dcn_fsdp_parallelism: -1: parameters are SHARDED across nodes,
# so every layer additionally all-gathers weights and reduce-scatters gradients over the
# fabric. Same fabric, far more of it per step.
#
# Kept in step with gfx950_llama3_70b_env.sh.

export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=.97
export LD_LIBRARY_PATH=/usr/local/lib/:/opt/rocm/lib:$LD_LIBRARY_PATH
export NVTE_USE_HIPBLASLT=1
# XLA_AUTOTUNE_LEVEL is overridable so a long exhaustive-autotune compile can be
# dialled down for a multi-run campaign. It applies identically to both A/B arms,
# so it cannot bias the comparison - but it does change absolute numbers, so any
# non-default value belongs in the reported config.
export XLA_FLAGS="--xla_gpu_memory_limit_slop_factor=95 --xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592 --xla_gpu_enable_command_buffer='' --xla_gpu_enable_latency_hiding_scheduler=True --xla_gpu_all_gather_combine_threshold_bytes=8589934592 --xla_gpu_enable_triton_gemm=False --xla_gpu_enable_cublaslt=True --xla_gpu_autotune_level=${XLA_AUTOTUNE_LEVEL:-4} --xla_gpu_enable_all_gather_combine_by_dim=FALSE"
export GPU_MAX_HW_QUEUES=2
export HIP_FORCE_DEV_KERNARG=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NVTE_FUSED_ATTN=1
# Honour a pre-set NCCL_DEBUG instead of clobbering it. This file is sourced AFTER
# the RCCL variant selector, so a hard assignment here silently undid the selector
# raising the level to INFO for attestation - and the attestation then "failed" for
# a reason that had nothing to do with the shim.
export NCCL_DEBUG="${NCCL_DEBUG:-VERSION}"
export NVTE_CK_USES_BWD_V3=1
export NVTE_CK_USES_FWD_V3=1
export NVTE_CK_IS_V3_ATOMIC_FP32=0
export NVTE_CK_HOW_V3_BF16_CVT=2
export NVTE_FUSED_ATTN_CK=1
export NVTE_FUSED_ATTN_AOTRITON=0
# gfx950-only RCCL WarpSpeed optimisation, on by default in gfx950 builds, can produce
# NaN losses during training. See benchmark/jax_maxtext/README.md.
export RCCL_WARP_SPEED_AUTO=0

# Append-only hook: lets a manifest add XLA flags (e.g. disabling NCCL comm splitting
# for a multi-host clique problem) without forking this file per experiment. Applied
# identically to every arm of an A/B, so it cannot bias a comparison.
# XLA_EXTRA_FLAGS is applied centrally in jax-maxtext_benchmark_report.sh, after this
# file is sourced. Appending it here too would apply it twice.

