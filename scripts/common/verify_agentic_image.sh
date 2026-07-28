#!/bin/bash
# Phase 0 pre-flight: verify a disagg Docker image can support agentic replay
# BEFORE spending a Slurm allocation. Fails fast with actionable messages.
#
# Checks:
#   1. SGLang exposes the OpenAI chat route aiperf needs (/v1/chat/completions).
#      A router that only serves /generate will make aiperf 404 on every turn.
#   2. The Mooncake disaggregation transfer backend is importable (Variant B
#      uses RUN_MORI=0 -> KV_TRANSFER_BACKEND=mooncake). If absent, use a
#      MoRI-built image or add Mooncake.
#
# Usage:
#   DOCKER_IMAGE_NAME=<tag> bash scripts/common/verify_agentic_image.sh
#
# --------------------------------------------------------------------------
# TIMEOUT FORMULA (size Slurm --time from this; agentic replay adds phases the
# random sweep does not have):
#
#   --time (s) >= T_pull      # docker pull per node (skip if cached)
#              + T_load        # server weight load: DeepSeek-V3 671B + 642G
#                              #   local-NVMe read dominates; allow >=1200s
#              + T_venv        # aiperf uv venv build          (~300s)
#              + N_retry*T_dl  # HF trace download, 3 x up to 900s worst case
#              + T_warmup      # cache warmup + grace          (<=1800s)
#              + DURATION      # measurement window
#              + T_agg         # aggregation + plots           (~120s)
#
#   Example DeepSeek-V3 smoke (DURATION=120, cached image, one clean download):
#     ~1200 + 300 + 900 + 900(grace) + 120 + 120 ~= 3540s -> request --time>=3600.
#   Raise SGLang server-ready/watchdog timeouts for 671B (ROUTER_READY_TIMEOUT_SECONDS).
# --------------------------------------------------------------------------
set -uo pipefail

IMG="${DOCKER_IMAGE_NAME:-}"
[ -n "$IMG" ] || { echo "[verify][ERROR] set DOCKER_IMAGE_NAME" >&2; exit 2; }

fail() { echo "[verify][FAIL] $*" >&2; exit 1; }
ok()   { echo "[verify][OK] $*"; }

echo "[verify] image: $IMG"

# 1. Mooncake transfer backend importable inside the image. The canonical import
#    for SGLang's --disaggregation-transfer-backend mooncake is
#    `from mooncake.engine import TransferEngine` (see
#    sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py and
#    MAD-private scripts/kvcache_transfer_bench/backends/mooncake/*.py).
if docker run --rm --entrypoint bash "$IMG" -lc \
     'python3 -c "from mooncake.engine import TransferEngine" 2>/dev/null'; then
    ok "Mooncake transfer backend importable (mooncake.engine.TransferEngine)"
else
    fail "mooncake.engine.TransferEngine not importable in image. Use RUN_MORI=1 with a MoRI-built image, or add Mooncake (mooncake-transfer-engine)."
fi

# 2. SGLang serves the OpenAI chat endpoint (served by sglang.launch_server /
#    sglang_router). Use importlib.util.find_spec to LOCATE the http_server /
#    openai serving_chat modules without executing them -- importing them pulls
#    in the GPU engine, which fails in a GPU-less pre-flight container and would
#    give a false negative. Module presence is sufficient to confirm the route.
if docker run --rm --entrypoint bash "$IMG" -lc \
     'python3 - <<PY
import importlib.util as u, sys
mods = ["sglang.srt.entrypoints.http_server",
        "sglang.srt.entrypoints.openai.serving_chat",
        "sglang.srt.openai_api.adapter"]
sys.exit(0 if any(u.find_spec(m) for m in mods) else 1)
PY'; then
    ok "SGLang OpenAI HTTP server present (serves /v1/chat/completions)"
else
    fail "Could not confirm SGLang OpenAI server module. Verify /v1/chat/completions is served (not only /generate)."
fi

ok "image pre-flight passed"
