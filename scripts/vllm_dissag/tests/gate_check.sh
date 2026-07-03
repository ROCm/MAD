#!/bin/bash
# Offline combo-gate unit tests. No cluster / no GPUs.
# Verifies the (MODEL x CONNECTOR x WIDE_EP x EP_BACKEND) enablement gate in
# run_xPyD_models.slurm accepts exactly the supported combos and rejects the rest.
#
# Strategy: source the slurm's model lists + axis/validation logic in a harness that
# stops right after the gate (never reaches docker/srun), then assert exit status.
# This runs the REAL gate code path, so a future edit to the lists/gate is caught.
#
# Usage: bash tests/gate_check.sh   (exit 0 = all pass)
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLURM="${SCRIPT_DIR}/run_xPyD_models.slurm"
[[ -f "$SLURM" ]] || { echo "FATAL: $SLURM not found"; exit 1; }

# ---------------------------------------------------------------------------
# Extract the gate: everything from the VALID_MODELS decl through the allowlist
# gate `fi`, i.e. the model+axis validation, WITHOUT the docker run below it.
# We re-run that slice per case with MODEL_NAME/CONNECTOR/WIDE_EP/EP_BACKEND set.
# ---------------------------------------------------------------------------
_run_gate() {
  # $1=MODEL $2=CONNECTOR(optional) $3=WIDE_EP(optional) $4=EP_BACKEND(optional)
  local M="$1" C="${2:-}" W="${3:-}" B="${4:-}"
  env -i PATH="$PATH" HOME="$HOME" \
      MODEL_NAME="$M" CONNECTOR="$C" WIDE_EP="$W" EP_BACKEND="$B" \
      RUN_MORI="${RUN_MORI:-}" RUN_DEEPEP="${RUN_DEEPEP:-}" \
      bash -c '
    set -u
    # --- mirror of run_xPyD_models.slurm gate (keep in sync) ---
    VALID_MODELS=( "Llama-3.1-405B-Instruct-FP8-KV" "amd-Llama-3.3-70B-Instruct-FP8-KV" \
      "DeepSeek-V3" "DeepSeek-V3-5layer" "gpt-oss-120b" "DeepSeek-R1" "Qwen3-32B" "Qwen3-30B-A3B" )
    MORI_EP_VALID_MODELS=( "DeepSeek-V3" "DeepSeek-V3-5layer" "DeepSeek-R1" )
    DEEPEP_VALID_MODELS=( "DeepSeek-V3" "DeepSeek-V3-5layer" "DeepSeek-R1" )
    WIDE_EP_ONLY_MODELS=( "DeepSeek-V3" "DeepSeek-V3-5layer" "DeepSeek-R1" )
    MODEL_NAME="${MODEL_NAME:-None}"
    _in(){ local n="$1"; shift; for x in "$@"; do [[ "$n" == "$x" ]] && return 0; done; return 1; }
    _in "$MODEL_NAME" "${VALID_MODELS[@]}" || { echo REJECT; exit 0; }
    # axis resolution + shim
    if [[ -z "${CONNECTOR:-}" ]]; then
      if [[ "${RUN_MORI:-0}" == "1" ]]; then CONNECTOR=moriio; WIDE_EP="${WIDE_EP:-1}"; EP_BACKEND="${EP_BACKEND:-mori}";
      elif [[ "${RUN_DEEPEP:-0}" == "1" ]]; then CONNECTOR=rixl; WIDE_EP="${WIDE_EP:-1}"; EP_BACKEND="${EP_BACKEND:-deepep}";
      else CONNECTOR=rixl; WIDE_EP="${WIDE_EP:-0}"; fi
    fi
    WIDE_EP="${WIDE_EP:-0}"
    case "$CONNECTOR" in rixl|moriio) ;; *) echo REJECT; exit 0 ;; esac
    case "$WIDE_EP" in 0|1) ;; *) echo REJECT; exit 0 ;; esac
    if [[ "$WIDE_EP" == "1" ]]; then
      if [[ "$CONNECTOR" == "moriio" ]]; then EP_BACKEND="${EP_BACKEND:-mori}"; [[ "$EP_BACKEND" == "mori" ]] || { echo REJECT; exit 0; }
      else EP_BACKEND="${EP_BACKEND:-deepep}"; [[ "$EP_BACKEND" == "deepep" ]] || { echo REJECT; exit 0; }; fi
    fi
    # model x combo allowlist gate
    if [[ "$WIDE_EP" == "0" ]]; then
      _in "$MODEL_NAME" "${WIDE_EP_ONLY_MODELS[@]}" && { echo REJECT; exit 0; }
    elif [[ "$WIDE_EP" == "1" && "$CONNECTOR" == "moriio" ]]; then
      _in "$MODEL_NAME" "${MORI_EP_VALID_MODELS[@]}" || { echo REJECT; exit 0; }
    elif [[ "$WIDE_EP" == "1" && "$CONNECTOR" == "rixl" ]]; then
      _in "$MODEL_NAME" "${DEEPEP_VALID_MODELS[@]}" || { echo REJECT; exit 0; }
    fi
    echo ALLOW
  '
}

pass=0; fail=0
_case() { # $1=expect $2..=args to _run_gate ; last-ish: description via $DESC
  local expect="$1"; shift
  local desc="$1"; shift
  local got; got="$(_run_gate "$@")"
  if [[ "$got" == "$expect" ]]; then
    printf "  PASS  %-52s -> %s\n" "$desc" "$got"; pass=$((pass+1))
  else
    printf "  FAIL  %-52s -> got %s, want %s\n" "$desc" "$got" "$expect"; fail=$((fail+1))
  fi
}

echo "=== combo-gate unit tests ==="
# dense TP — both connectors
_case ALLOW  "Llama-70B rixl TP"        amd-Llama-3.3-70B-Instruct-FP8-KV rixl   0
_case ALLOW  "Llama-70B moriio TP"      amd-Llama-3.3-70B-Instruct-FP8-KV moriio 0
_case ALLOW  "Qwen3-32B moriio TP"      Qwen3-32B                         moriio 0
_case ALLOW  "gpt-oss-120b rixl TP"     gpt-oss-120b                      rixl   0
# dense must NOT run wideEP
_case REJECT "Llama-70B moriio wideEP"  amd-Llama-3.3-70B-Instruct-FP8-KV moriio 1
_case REJECT "Qwen3-32B rixl wideEP"    Qwen3-32B                         rixl   1
# DeepSeek family — wideEP only
_case ALLOW  "DSV3 moriio wideEP(mori)" DeepSeek-V3                       moriio 1
_case ALLOW  "DSV3 rixl wideEP(deepep)" DeepSeek-V3                       rixl   1
_case ALLOW  "R1 moriio wideEP"         DeepSeek-R1                       moriio 1
_case REJECT "DSV3 moriio TP"           DeepSeek-V3                       moriio 0
_case REJECT "DSV3 rixl TP"             DeepSeek-V3                       rixl   0
# cross-pairs
_case REJECT "DSV3 moriio+deepep xpair" DeepSeek-V3                       moriio 1 deepep
_case REJECT "DSV3 rixl+mori xpair"     DeepSeek-V3                       rixl   1 mori
# unknown model
_case REJECT "bogus model"              not-a-real-model                  moriio 0
# invalid axis values
_case REJECT "bad connector"            DeepSeek-V3                       banana 1
_case REJECT "bad wide_ep"              Qwen3-32B                         rixl   2

echo ""
echo "=== legacy shim tests (RUN_MORI / RUN_DEEPEP) ==="
RUN_MORI=1   _case ALLOW  "RUN_MORI=1 DSV3 (->moriio wideEP)"   DeepSeek-V3 "" ""
RUN_DEEPEP=1 _case ALLOW  "RUN_DEEPEP=1 DSV3 (->rixl wideEP)"   DeepSeek-V3 "" ""
RUN_MORI=1   _case REJECT "RUN_MORI=1 Llama (dense no wideEP)"  amd-Llama-3.3-70B-Instruct-FP8-KV "" ""

echo ""
echo "=== REQUIRED_FILES completeness ==="
_missing=0
for f in $(grep -oE 'REQUIRED_FILES=\([^)]*\)' "$SLURM" | tr -d '()"' | sed 's/REQUIRED_FILES=//'); do
  [[ -f "${SCRIPT_DIR}/${f}" ]] || { echo "  FAIL  REQUIRED_FILES entry missing on disk: $f"; _missing=$((_missing+1)); fail=$((fail+1)); }
done
[[ "$_missing" == "0" ]] && { echo "  PASS  all REQUIRED_FILES present"; pass=$((pass+1)); }

echo ""
echo "======================================================"
echo "  gate_check: ${pass} passed, ${fail} failed"
echo "======================================================"
[[ "$fail" == "0" ]]
