#!/bin/bash
# Backend-agnostic OFFLINE gate for the AgentX agentic suite. No server, no GPU,
# no cluster, no network. Runs the REAL code paths where possible so a future
# edit to the config loader / generator / verifier / suite driver is caught here.
#
# Checks:
#   1. bash -n every backend hook + connector + the suite driver + agentic_lib.
#   2. Every workload in agentic.example.yaml resolves (--emit-workload-shell).
#   3. Deterministic gen+verify smoke on the tiny `small` profile (13/13 axes).
#   4. DRY_RUN suite driver prints a plan line for each workload and exits 0.
#
# Usage: bash scripts/common/agentx/tests/run_offline.sh   (exit 0 = all pass)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENTX_DIR="$(cd "$HERE/.." && pwd)"
COMMON_DIR="$(cd "$AGENTX_DIR/.." && pwd)"
REPO_ROOT="$(cd "$COMMON_DIR/../.." && pwd)"
CONFIG="$AGENTX_DIR/agentic.example.yaml"
SUITE_DRIVER="$COMMON_DIR/benchmark_agentic_suite.sh"
PY="python3"

pass=0; fail=0
_pass() { printf "  PASS  %s\n" "$1"; pass=$((pass+1)); }
_fail() { printf "  FAIL  %s\n" "$1"; fail=$((fail+1)); }

# Isolated tmp workspace (corpus + profile JSON), cleaned up on exit.
TMP="$(mktemp -d "${TMPDIR:-/tmp}/agentx_offline.XXXXXX")"
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

# ---------------------------------------------------------------------------
echo "=== 1. bash -n syntax check (hooks + connectors + driver + lib) ==="
SYNTAX_TARGETS=(
    "$COMMON_DIR/benchmark_agentic.sh"
    "$REPO_ROOT/scripts/sglang_disagg/benchmark_agentic.sh"
    "$REPO_ROOT/scripts/vllm_dissag/benchmark_agentic.sh"
    "$REPO_ROOT/scripts/vllm_dissag/connectors/rixl.sh"
    "$REPO_ROOT/scripts/vllm_dissag/connectors/moriio.sh"
    "$SUITE_DRIVER"
    "$COMMON_DIR/agentic_lib.sh"
    "$HERE/resolve_ctx_offline.sh"
)
for f in "${SYNTAX_TARGETS[@]}"; do
    rel="${f#"$REPO_ROOT/"}"
    if [ ! -f "$f" ]; then
        _fail "missing: $rel"
    elif bash -n "$f" 2>/tmp/agentx_offline_syn.$$; then
        _pass "bash -n $rel"
    else
        _fail "bash -n $rel"
        sed 's/^/        /' /tmp/agentx_offline_syn.$$ || true
    fi
    rm -f /tmp/agentx_offline_syn.$$
done
# The shared ctx-resolve stub is Python, so py_compile it (bash -n won't do).
STUB_PY="$HERE/_stub_server.py"
stub_rel="${STUB_PY#"$REPO_ROOT/"}"
if [ ! -f "$STUB_PY" ]; then
    _fail "missing: $stub_rel"
elif "$PY" -m py_compile "$STUB_PY" 2>/tmp/agentx_offline_pyc.$$; then
    _pass "py_compile $stub_rel"
else
    _fail "py_compile $stub_rel"
    sed 's/^/        /' /tmp/agentx_offline_pyc.$$ || true
fi
rm -f /tmp/agentx_offline_pyc.$$

# ---------------------------------------------------------------------------
echo ""
echo "=== 2. per-workload config resolution (agentic.example.yaml) ==="
# Enumerate workloads from the resolved config rather than hardcoding.
mapfile -t WORKLOADS < <("$PY" "$AGENTX_DIR/agentx_config.py" --config "$CONFIG" --dump-json \
    | "$PY" -c 'import sys,json; [print(w["name"]) for w in json.load(sys.stdin)["workloads"]]')
if [ "${#WORKLOADS[@]}" -eq 0 ]; then
    _fail "enumerate workloads from --dump-json"
else
    _pass "enumerated ${#WORKLOADS[@]} workloads: ${WORKLOADS[*]}"
fi
for name in "${WORKLOADS[@]}"; do
    if "$PY" "$AGENTX_DIR/agentx_config.py" --config "$CONFIG" --workload "$name" \
            --profile-out "$TMP/${name}.profile.json" --emit-workload-shell >/dev/null 2>"$TMP/wl.err"; then
        _pass "resolve workload '$name'"
    else
        _fail "resolve workload '$name'"
        sed 's/^/        /' "$TMP/wl.err" || true
    fi
done

# ---------------------------------------------------------------------------
echo ""
echo "=== 3. deterministic gen+verify smoke (profiles/small.yaml, seed=42) ==="
SMALL_JSON="$TMP/small.profile.json"
SMALL_CORPUS="$TMP/small_corpus"
if "$PY" "$AGENTX_DIR/agentx_config.py" --profile "$AGENTX_DIR/profiles/small.yaml" \
        --emit-json > "$SMALL_JSON" 2>"$TMP/small.err"; then
    _pass "emit small profile JSON"
else
    _fail "emit small profile JSON"; sed 's/^/        /' "$TMP/small.err" || true
fi
if "$PY" "$AGENTX_DIR/gen_agentx_profile.py" --profile "$SMALL_JSON" --seed 42 \
        --out-dir "$SMALL_CORPUS" >/dev/null 2>"$TMP/gen.err"; then
    _pass "generate small corpus"
else
    _fail "generate small corpus"; sed 's/^/        /' "$TMP/gen.err" || true
fi
# verify exits 0 iff all axes pass; its final line is "N/N axes within band".
if verify_out="$("$PY" "$AGENTX_DIR/verify_agentx_profile.py" --profile "$SMALL_JSON" \
        --corpus "$SMALL_CORPUS" 2>&1)"; then
    band_line="$(echo "$verify_out" | tail -n1)"
    _pass "verify small corpus (${band_line})"
else
    _fail "verify small corpus"
    echo "$verify_out" | sed 's/^/        /'
fi

# ---------------------------------------------------------------------------
echo ""
echo "=== 4. DRY_RUN suite driver (no server) ==="
# DRY_RUN must not need the aiperf venv or network. If it does, degrade to WARN.
dry_out=""; dry_rc=0
dry_out="$(DRY_RUN=1 AGENTIC_CONFIG="$CONFIG" bash "$SUITE_DRIVER" 2>&1)" || dry_rc=$?
if [ "$dry_rc" -ne 0 ]; then
    if echo "$dry_out" | grep -qiE 'network|download|uv |venv|pip|install'; then
        printf "  WARN  DRY_RUN suite driver exited %s (looks env/network related)\n" "$dry_rc"
        echo "$dry_out" | tail -n 20 | sed 's/^/        /'
    else
        _fail "DRY_RUN suite driver exited $dry_rc"
        echo "$dry_out" | tail -n 20 | sed 's/^/        /'
    fi
else
    _pass "DRY_RUN suite driver exit 0"
    for name in "${WORKLOADS[@]}"; do
        if echo "$dry_out" | grep -q "workload='${name}'"; then
            _pass "plan line for '$name'"
        else
            _fail "no plan line for '$name'"
        fi
    done
fi

# ---------------------------------------------------------------------------
echo ""
echo "=== 5. source=corpus resolution + configurable scenario ==="
# corpus workload resolves to an --input-file replay (no download/generate) and
# skips verification when it carries no profile/preset.
corpus_wl="$("$PY" "$AGENTX_DIR/agentx_config.py" --config "$CONFIG" --workload my_corpus \
    --emit-workload-shell 2>/dev/null)" || true
if echo "$corpus_wl" | grep -q "WL_SOURCE='corpus'" \
        && echo "$corpus_wl" | grep -Eq "WL_INPUT_DIR='.+'"; then
    _pass "source=corpus resolves input_dir"
else
    _fail "source=corpus resolves input_dir"
fi
if echo "$corpus_wl" | grep -q "WL_PROFILE_FILE=''"; then
    _pass "source=corpus verification optional (no profile -> no pre-gate)"
else
    _fail "source=corpus verification optional (no profile -> no pre-gate)"
fi
# scenario: defaults to inferencex-agentx-mvp, overridable via AGENTIC_SCENARIO.
ovr_sc="$(AGENTIC_SCENARIO=my-scenario "$PY" "$AGENTX_DIR/agentx_config.py" \
    --config "$CONFIG" --emit-config-shell | grep '^SUITE_SCENARIO=')"
if [ "$ovr_sc" = "SUITE_SCENARIO='my-scenario'" ]; then
    _pass "AGENTIC_SCENARIO overrides run.scenario"
else
    _fail "AGENTIC_SCENARIO overrides run.scenario (got $ovr_sc)"
fi
if echo "$dry_out" | grep -Eq "^  run.scenario +: inferencex-agentx-mvp"; then
    _pass "DRY_RUN plan shows default scenario"
else
    _fail "DRY_RUN plan shows default scenario"
fi
sc_out="$(DRY_RUN=1 AGENTIC_SCENARIO=my-scenario AGENTIC_CONFIG="$CONFIG" bash "$SUITE_DRIVER" 2>&1)" || true
if echo "$sc_out" | grep -q -- "--scenario my-scenario"; then
    _pass "DRY_RUN replay command honors AGENTIC_SCENARIO"
else
    _fail "DRY_RUN replay command honors AGENTIC_SCENARIO"
fi

# ---------------------------------------------------------------------------
echo ""
echo "=== 6. offline ctx-window resolver (stub server) ==="
# Runs the shared backend-looped resolver test once (sglang + vllm x 4 paths).
# Guard the invocation so its non-zero exit doesn't abort us under `set -e`.
CTX_TEST="$HERE/resolve_ctx_offline.sh"
ctx_rc=0
ctx_out="$(bash "$CTX_TEST" 2>&1)" || ctx_rc=$?
echo "$ctx_out" | sed 's/^/        /'
for bk in sglang vllm; do
    if echo "$ctx_out" | grep -q "^---- \[$bk\] summary: ALL PASS"; then
        _pass "ctx resolver [$bk] all paths"
    else
        _fail "ctx resolver [$bk] all paths"
    fi
done
if [ "$ctx_rc" -ne 0 ]; then
    echo "        (resolve_ctx_offline.sh exited $ctx_rc)"
fi

# ---------------------------------------------------------------------------
echo ""
# ---------------------------------------------------------------------------
echo ""
echo "=== 7. N1 parser + robustness guards ==="

# 7.1 N1: workloads: at the SAME indent as its `-` items must resolve to a
#     NON-EMPTY workload list via the fallback YAML loader (regression for the
#     silently-dropped same-indent block sequence).
cat > "$TMP/n1_same_indent.yaml" <<'YAML'
serving:
  model: auto
  max_model_len: 524288
run:
  concurrency: [2]
  duration: 900
  scenario: inferencex-agentx-mvp
workloads:
- name: samelevel
  source: profile
  preset: conformance_256k
YAML
n1_count="$(AGENTX_YAML_FALLBACK=1 "$PY" "$AGENTX_DIR/agentx_config.py" \
    --config "$TMP/n1_same_indent.yaml" --dump-json 2>/dev/null \
    | "$PY" -c 'import sys,json; print(len(json.load(sys.stdin).get("workloads") or []))')" || n1_count=0
if [ "${n1_count:-0}" -ge 1 ]; then
    _pass "N1 fallback parser resolves same-indent workloads ($n1_count)"
else
    _fail "N1 fallback parser resolves same-indent workloads (got '${n1_count}')"
fi

# 7.2 gen guard: block_size 0 -> non-zero exit + explicit message.
"$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); d["block_size"]=0; json.dump(d,open(sys.argv[2],"w"))' \
    "$SMALL_JSON" "$TMP/bs0.json"
bs0_out="$("$PY" "$AGENTX_DIR/gen_agentx_profile.py" --profile "$TMP/bs0.json" --seed 42 \
    --out-dir "$TMP/bs0_corpus" 2>&1)" && bs0_rc=0 || bs0_rc=$?
if [ "$bs0_rc" -ne 0 ] && echo "$bs0_out" | grep -q '\[gen\] block_size must be >= 1'; then
    _pass "gen guard: block_size 0 rejected"
else
    _fail "gen guard: block_size 0 rejected (rc=$bs0_rc)"
    echo "$bs0_out" | sed 's/^/        /'
fi

# 7.3 gen guard: empty turns.values -> non-zero exit + explicit message.
"$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); d["turns"]["values"]=[]; json.dump(d,open(sys.argv[2],"w"))' \
    "$SMALL_JSON" "$TMP/tv0.json"
tv0_out="$("$PY" "$AGENTX_DIR/gen_agentx_profile.py" --profile "$TMP/tv0.json" --seed 42 \
    --out-dir "$TMP/tv0_corpus" 2>&1)" && tv0_rc=0 || tv0_rc=$?
if [ "$tv0_rc" -ne 0 ] && echo "$tv0_out" | grep -q '\[gen\] turns must have non-empty'; then
    _pass "gen guard: empty turns.values rejected"
else
    _fail "gen guard: empty turns.values rejected (rc=$tv0_rc)"
    echo "$tv0_out" | sed 's/^/        /'
fi

# 7.4 verify guard: session JSON lacking 'requests' -> non-zero exit + message.
mkdir -p "$TMP/badcorpus"
echo '{}' > "$TMP/badcorpus/s1.json"
vf_out="$("$PY" "$AGENTX_DIR/verify_agentx_profile.py" --profile "$SMALL_JSON" \
    --corpus "$TMP/badcorpus" 2>&1)" && vf_rc=0 || vf_rc=$?
if [ "$vf_rc" -ne 0 ] && echo "$vf_out" | grep -q "missing 'requests'"; then
    _pass "verify guard: missing 'requests' rejected"
else
    _fail "verify guard: missing 'requests' rejected (rc=$vf_rc)"
    echo "$vf_out" | sed 's/^/        /'
fi

# 7.5 preset guard: unknown preset name -> reported, not silently loaded.
cat > "$TMP/preset_missing.yaml" <<'YAML'
serving:
  model: auto
  max_model_len: 524288
run:
  concurrency: [2]
  duration: 900
workloads:
  - name: bad
    preset: __missing__
YAML
pm_out="$("$PY" "$AGENTX_DIR/agentx_config.py" --config "$TMP/preset_missing.yaml" --dump-json 2>&1)" || true
if echo "$pm_out" | grep -q "preset not found"; then
    _pass "preset guard: missing preset reported"
else
    _fail "preset guard: missing preset reported"
    echo "$pm_out" | sed 's/^/        /'
fi

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
echo ""
echo "=== 8. shell driver static guards (fixes 2/3/4) ==="
SLURM_LAUNCHER="$REPO_ROOT/scripts/sglang_disagg/run_xPyD_models.slurm"

# 8.1 suite driver must no longer swallow replay failures with `|| true`
#     (exclude comment lines to avoid false positives).
if grep -vE '^[[:space:]]*#' "$SUITE_DRIVER" \
        | grep -q 'run_agentic_replay_and_write_outputs.*|| true'; then
    _fail "suite driver: active '|| true' on replay call removed"
else
    _pass "suite driver: active '|| true' on replay call removed"
fi

# 8.2 sglang launcher must set pipefail before the tee pipe.
if grep -vE '^[[:space:]]*#' "$SLURM_LAUNCHER" | grep -q 'set -o pipefail'; then
    _pass "sglang launcher: 'set -o pipefail' present"
else
    _fail "sglang launcher: 'set -o pipefail' present"
fi

# 8.3 sglang launcher must expand $HOME in AGENTIC_CONFIG before docker forward.
if grep -vE '^[[:space:]]*#' "$SLURM_LAUNCHER" | grep -q 'AGENTIC_CONFIG="\$(eval echo'; then
    _pass "sglang launcher: AGENTIC_CONFIG \$HOME expansion present"
else
    _fail "sglang launcher: AGENTIC_CONFIG \$HOME expansion present"
fi

# ---------------------------------------------------------------------------
echo "======================================================"
echo "  run_offline: ${pass} passed, ${fail} failed"
echo "======================================================"
[ "$fail" -eq 0 ]
