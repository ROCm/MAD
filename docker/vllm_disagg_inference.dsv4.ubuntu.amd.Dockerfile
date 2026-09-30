# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
###############################################################################
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
# =============================================================================
# vllm_disagg_inference.dsv4.ubuntu.amd.Dockerfile
#   DeepSeek-V4 Flash-FP8 / Pro-FP8 MoRI-EP WideEP disagg image.
#   PER-MODEL image, isolated from the base and glmv5.1 Dockerfiles.
#
#   This file only pins a prebuilt image by digest; it does not rebuild the stack.
#   The image is internal (rocm/pytorch-private) — there is no public equivalent yet.
#
#   Contents (/app/versions.txt in the image):
#     base    rocm/dev-ubuntu-22.04:7.2.3-complete (via vllm/vllm-openai-rocm:v0.29.0)
#     vLLM    98dff2a81d74 (v0.29.0) + DSV4 MoRI/MoRIIO fixes applied in-source:
#             combine,trim,attn_backend,storage_span,mixed_bs,gate,attn_xfer,rdma_wait
#     AITER   10f8874dc2cd (source build), flydsl 0.3.2
#     MoRI    07bdace2ff73
#     router  vllm-project/router f962dfcf (/usr/local/bin/vllm-router)
#
# THIS IMAGE IS THE CONTRACT, as for glmv5.1: MAD ships no runtime patchers, so every
# DSV4 fix must already be in the image's vLLM. Do not substitute a stock v0.29.0 image:
# it boots but fails the DSV4 MoRIIO KV transfer. Pin by digest; the tag has been
# re-pushed and different digests carry different fixes.
#
# Runtime switches read by the baked code (set per model in scripts/vllm_dissag/models.yaml):
#   DSV4_TRANSFER_ATTN   default 0 — the recipe sets 1; 0 skips the full-attention KV
#                        transfer and long-context retrieval fails.
#   MORI_TRIM_DISPATCH   default 1 — sizes the expert GEMM to live tokens, not the full
#                        MoRI recv buffer (decode ITL ~24 ms vs ~290 ms at con=1, EP8).
# =============================================================================

ARG BASE_IMAGE=rocm/pytorch-private:vllm-recent-source-basem-v0290-aiter-10f8874-mori07bdace-tk@sha256:49e87adaece784e040dde49b79852b0fdbc515696619d559296a9397ecfd9137
FROM ${BASE_IMAGE}

ENTRYPOINT []
WORKDIR /app
