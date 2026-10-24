#!/bin/bash
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
# Shared entry point for configs generated from InferenceX e2e scenarios by
# tools/infx2mad.py.
#
# One body for every engine: scripts/<engine>/run_infx.sh sources this and
# passes the engine's own runner (run_sglang.py / run_vllm.py / run_atom.py).
# The generated config carries everything else, so the per-engine file stays a
# two-liner and the engines cannot drift apart in how they are invoked.
#
# Usage from scripts/<engine>/run_infx.sh:
#     source ../scripts/common/infx_run_lib.sh
#     infx_main run_sglang.py "$@"

infx_main() {
    local runner="$1"; shift
    local MODEL="" CONFIG_ARG="" VARIANT_ARG="all"

    while [[ "$#" -gt 0 ]]; do
        case $1 in
            --model_repo) MODEL="$2"; shift ;;
            --config)     CONFIG_ARG="$2"; shift ;;
            --variant)    VARIANT_ARG="$2"; shift ;;
            *) echo "Unknown parameter passed: $1" >&2; return 1 ;;
        esac
        shift
    done

    : "${CONFIG:=$CONFIG_ARG}"
    : "${VARIANT:=$VARIANT_ARG}"
    [[ -n "$CONFIG" ]] || { echo "--config is required" >&2; return 1; }
    [[ -n "$MODEL" ]]  || { echo "--model_repo is required" >&2; return 1; }

    if [[ -z "${HF_HUB_CACHE:-}" ]]; then
        export HF_HUB_CACHE="/myworkspace"
    fi
    # The per-engine wrappers run under `set -ex`, so assigning the token
    # plainly would print it to stderr and into every captured run log.
    local _xtrace_was_on=0
    case "$-" in *x*) _xtrace_was_on=1; set +x ;; esac
    export HF_TOKEN="${MAD_SECRETS_HFTOKEN:-${HF_TOKEN:-}}"
    [ "$_xtrace_was_on" -eq 1 ] && set -x

    # Download accelerator only. When MAD_DATAHOME points at weights already on
    # the cluster there is nothing to download, and an image without outbound
    # network must not fail the run over it.
    pip install -qqq hf-transfer || echo "hf-transfer unavailable; continuing"

    # run_vllm.py and run_atom.py have no --variant; only pass it where the
    # runner declares support, which infx2mad.py mirrors when it writes args.
    local variant_arg=()
    if grep -q -- "--variant" "$runner" 2>/dev/null; then
        variant_arg=(--variant "$VARIANT")
    fi

    # ATOM auto-merges its sibling configs/perf.yaml, which would fold MAD's
    # own tuning into a translated recipe without any fidelity note.
    local extra=()
    if [[ "$runner" == run_atom.py ]] && grep -q -- "--no-perf-merge" "$runner" 2>/dev/null; then
        extra=(--no-perf-merge)
    fi
    python3 -u "$runner" --config "$CONFIG" --model "$MODEL" "${variant_arg[@]}" "${extra[@]}"

    # madengine collects the CSV from the parent of run_directory.
    local model_name
    model_name="$(basename "$MODEL")"
    mv "perf_${model_name}.csv" ../
}
