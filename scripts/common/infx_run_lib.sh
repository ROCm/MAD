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

    # All three runners auto-merge their sibling configs/perf.yaml by default
    # (#290 extended it from ATOM to SGLang and vLLM), which would fold MAD's own
    # tuning into a translated recipe with no fidelity note -- e.g. a vLLM
    # Kimi-K3 block picking up an AITER MoE env var the recipe omits. The run
    # has to be exactly the configuration translated and validated, so opt out
    # wherever the runner supports it.
    # All three runners have both flags today. A runner that renames one must
    # fail here, not quietly go back to merging MAD's own tuning into a recipe
    # that is reported as faithful.
    local extra=()
    if grep -q -- "--no-perf-merge" "$runner" 2>/dev/null; then
        extra=(--no-perf-merge)
    else
        echo "[infx] ERROR: $runner does not accept --no-perf-merge; refusing to run with MAD's perf.yaml merged into the recipe" >&2
        return 1
    fi
    # The same holds for the sibling configs/acc.yaml: ATOM folds its matching
    # entry's extra_args and env into every config, not only accuracy ones (a
    # Kimi-K3 entry would cap the context and turn prefix caching off). An
    # empty source leaves nothing to merge; all three runners take --acc.
    if grep -q -- "'--acc'" "$runner" 2>/dev/null; then
        extra+=(--acc /dev/null)
    else
        echo "[infx] ERROR: $runner does not accept --acc; refusing to run with its acc.yaml merged into the recipe" >&2
        return 1
    fi
    # madengine collects the CSV from the parent of run_directory.
    local model_name csv
    model_name="$(basename "$MODEL")"
    csv="perf_${model_name}.csv"
    # The runners append to an existing CSV. A previous attempt that died
    # before the move below leaves its rows here, and a retry would count
    # them as this run's. Set it aside so only this invocation is checked.
    if [ -f "$csv" ]; then
        mv "$csv" "${csv}.previous.$(date +%s)"
    fi

    # Serving results of an earlier attempt are moved aside, not just
    # skipped: a client that exits 0 without rewriting its file would let the
    # runner turn the old file into this run's CSV rows. The marker then
    # bounds what this invocation wrote.
    if compgen -G "*_serving_*.json" >/dev/null; then
        local prev_dir=".infx_previous/$(date +%s)"
        mkdir -p "$prev_dir" && mv -- *_serving_*.json "$prev_dir"/
    fi
    touch .infx_run_start
    python3 -u "$runner" --config "$CONFIG" --model "$MODEL" "${variant_arg[@]}" "${extra[@]}"

    # A runner whose server never became ready still writes a config-only row
    # with blank metric/performance and exits 0. Keep the file for the record
    # but fail the run, so a partially failed ladder is not reported as a
    # measurement.
    local rc=0
    python3 - "$csv" "$CONFIG" "$VARIANT" <<'PY' || rc=$?
import csv, sys, yaml
path, config, variant = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    rows = list(csv.DictReader(open(path)))
except OSError as exc:
    sys.exit(f"[infx] no result CSV: {exc}")
import math
def measured(r):
    # Non-empty is not enough: 'nan', 'inf' or a zero throughput are cells a
    # failed point can produce too.
    try:
        v = float(r.get("performance") or "")
    except ValueError:
        return False
    if not math.isfinite(v) or not (r.get("metric") or "").strip():
        return False
    return v > 0 if "throughput" in r["metric"] else True
bad = [i for i, r in enumerate(rows, 1) if not measured(r)]
if not rows or bad:
    sys.exit(f"[infx] {path}: {len(rows)} row(s), unmeasured rows {bad}; "
             "the run is not a measurement")
# A runner that gets no metrics for a point can return an empty list for it
# and write nothing, so the rows present can all be good while a point is
# missing. Compare selected and reported points on the identity columns every
# runner writes verbatim from the block (variant only where the runner has
# it). Points that coincide on all of them collapse on both sides, so this
# can miss such a point but never rejects a complete run.
blocks = yaml.safe_load(open(config)) or []
if variant and variant != "all":
    blocks = [b for b in blocks if b.get("variant") == variant]
cols = [c for c in ("benchmark", "variant", "tp", "max_concurrency", "inp", "out",
                    "dtype", "kv_cache_dtype")
        if c in rows[0]]
def norm(v):
    try:
        return str(float(v))
    except (TypeError, ValueError):
        return str(v or "")
import json
from collections import defaultdict
# Within one identity, points that differ in their server (extra_args, env)
# launch different commands, and `cmd` is in every runner's CSV; points that
# differ only in replay settings (agentic_env) launch the same command and
# are counted once on both sides, so a complete run is never rejected.
# The CSV keeps throughput and latency but not how many requests completed,
# so a point where most requests failed still yields finite rows. Each serving
# client leaves a result JSON here with `completed`; compare it with the prompts
# requested (the JSON's own num_prompts, else the field the runners put in the
# file name, <..>_<num_prompts>_<concurrency>.json).
import glob, os
start = os.path.getmtime(".infx_run_start") if os.path.exists(".infx_run_start") else 0
max_fail = float(os.environ.get("INFX_MAX_REQUEST_FAILURE_RATE", "0.10"))
for res in sorted(glob.glob("*_serving_*.json")):
    if os.path.getmtime(res) < start:
        continue
    # vLLM and ATOM write one JSON document; SGLang appends one JSON line per
    # run, so a retried point holds several -- the last is this run's. A
    # result of this invocation that cannot be read fails the run rather than
    # skipping its accounting.
    # Earlier attempts' files were moved aside before the runner started, so
    # every record here is this invocation's, and every one is checked: a
    # later successful execution must not hide an earlier failed one.
    try:
        text = open(res).read()
        try:
            docs = [json.loads(text)]
        except ValueError:
            docs = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    except (OSError, ValueError):
        docs = []
    if not docs or not all(isinstance(d, dict) for d in docs):
        sys.exit(f"[infx] {res}: serving result of this run cannot be read; "
                 "request accounting cannot be checked")
    for doc in docs:
        done = doc.get("completed")
        asked = doc.get("num_prompts")
        if asked is None:
            try:
                asked = int(os.path.splitext(res)[0].split("_")[-2])
            except (ValueError, IndexError):
                asked = None
        if not isinstance(done, (int, float)) or not asked:
            # Every client the runners drive (sglang.bench_serving, vllm bench
            # serve, atom.benchmarks.benchmark_serving) writes `completed`; a
            # result without the counts cannot show it is within the failure
            # limit, so it is not accepted.
            sys.exit(f"[infx] {res}: no completed/requested request count, so the "
                     "failure rate cannot be checked; the run is not accepted")
        if done < asked * (1 - max_fail):
            sys.exit(f"[infx] {res}: {int(done)} of {int(asked)} requests completed "
                     f"(more than {max_fail:.0%} failed); the run is not a measurement")
want = defaultdict(set)
for b in blocks:
    # random_range_ratio is part of the client command each runner records
    # in `cmd`, so points differing only in it are separate points too.
    want[tuple(norm(b.get(c)) for c in cols)].add(
        json.dumps([b.get("extra_args"), b.get("env"), b.get("random_range_ratio")],
                   sort_keys=True, default=str))
have = defaultdict(set)
for r in rows:
    have[tuple(norm(r.get(c)) for c in cols)].add(r.get("cmd", ""))
short = {k: (len(v), len(have.get(k, ()))) for k, v in want.items()
         if len(have.get(k, ())) < len(v)}
if short:
    sys.exit(f"[infx] {path}: selected point(s) with no metrics: "
             + "; ".join(f"{dict(zip(cols, k))} expected {e} server config(s), got {g}"
                         for k, (e, g) in sorted(short.items()))
             + "; the run is incomplete")
PY
    # Under the wrappers' `set -e` a bare failing check would exit before the
    # file is moved; keep it, then fail.
    if [ -f "$csv" ]; then mv "$csv" ../; fi
    return $rc
}
