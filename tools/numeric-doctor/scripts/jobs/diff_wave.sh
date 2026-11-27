#!/bin/bash
#SBATCH --job-name=nd-diff-wave
#SBATCH --nodes=1
#SBATCH --time=01:00:00
# debug-numeric differential mode (`step2-diff.md`): the same model command under the user's configuration (A),
# A again (A2: the noise floor), the conservative reference (REF) and A with one switch flipped per switch
# (flip_<s>: the bisection), side by side on one node -- then scripts/output_diff.py compares what each run
# returned (scripts/diff_tap records it, no model edit).
#
#   CMD='<the model, once, as nd run takes it: e.g. python3 $ND_REPO/examples/.../serve.py --seed 0>' \
#   [FRAMEWORK=auto|vllm|sglang|torch] [ONLY=a,b] [DROP=a,b] [EXTRA=fp32] [SETS='VAR=VALUE ...'] \
#   [GPUS_PER_RUN=1] [ARMS='A A2 REF ...'] [DTYPE=bfloat16] [TAP=gen,step|fwd] [FUNC=[module:]name] \
#     ND_ENV=<ARTIFACT_DIR>/env.sh scripts/nd submit --name diff --gpus 8 --exclusive -- diff_wave.sh
#
# The inputs must be the same in every run (fixed seed, fixed data order): output_diff says INPUTS_DIFFER
# otherwise. Arms run GPUS_PER_RUN GPUs each, as many at once as the node holds; the rest queue.
# Writes <ART>/diff/<job>/: <arm>/outputs.jsonl, <arm>.log, diff.json, diff.txt. Last line:
#   DIFF_WAVE_RESULT=<AGREE|DIVERGE|NONDETERMINISTIC|INPUTS_DIFFER|NO_OUTPUTS> switch=<s|-> dir=<dir>
#
# Localisation (MODULES=<request / step indices>): only A and REF run (ARMS to change), each with per-module
# recording for those requests (ND_DIFF_MODULES) and LOCATE_ENV added to both (e.g. ND_VLLM_COMPILE=0
# ND_VLLM_CUDAGRAPH_MODE=NONE: module forwards must run in Python); last line
#   DIFF_LOCATE_RESULT=<FOUND module=<m>|NO_JUMP|NONE_ABOVE|NO_OVERLAP> dir=<dir>
set -u
# shellcheck disable=SC1091
source "${ND_REPO_HOST:?run through nd submit, or export ND_REPO_HOST}/scripts/env/lib.sh"
nd_job_init
: "${CMD:?CMD=<the model command, once>}"
GPR=${GPUS_PER_RUN:-1}
OUT=${OUT:-${ND_ART_HOST:-$PWD}/diff/${SLURM_JOB_ID:-$$}}
mkdir -p "$OUT" && chmod 777 "$OUT"
PY=$(command -v python3 || command -v python) || nd_die "no python3 on this host (output_diff.py needs one, no torch)"
FW=${FRAMEWORK:-auto}
if [ "$FW" = auto ]; then
  FW=$("$PY" "$ND_REPO_HOST/scripts/configs.py" detect --root "$ND_REPO_HOST" $CMD | sed -n 's/^FRAMEWORK=//p')
fi
case $FW in vllm|sglang|torch) ;; *) nd_die "framework '$FW': set FRAMEWORK=vllm|sglang|torch" ;; esac
setargs=(); for s in ${SETS:-}; do setargs+=(--set "$s"); done
mapfile -t ARMLINES < <("$PY" "$ND_REPO_HOST/scripts/configs.py" arms --framework "$FW" --only "${ONLY:-}" \
                          --drop "${DROP:-}" --extra "${EXTRA:-}" "${setargs[@]+"${setargs[@]}"}")
if [ -n "${MODULES:-}" ]; then ARMS=${ARMS:-"A REF"}; fi
if [ -n "${ARMS:-}" ]; then
  keep=(); for l in "${ARMLINES[@]}"; do for a in $ARMS; do [ "${l%%|*}" = "$a" ] && keep+=("$l"); done; done
  ARMLINES=("${keep[@]}")
fi
NG=$(nd_ngpus); SLOTS=$((NG / GPR)); [ "$SLOTS" -ge 1 ] || nd_die "GPUS_PER_RUN=$GPR > $NG GPUs on this node"
echo "diff wave: framework=$FW arms=${#ARMLINES[@]} gpus_per_run=$GPR slots=$SLOTS out=$OUT${MODULES:+ modules=$MODULES}"
for l in "${ARMLINES[@]}"; do echo "  arm ${l%%|*}: ${l#*|}"; done
TAP_PP=$(rt_path "$ND_REPO_HOST/scripts/diff_tap"):$(rt_path "$ND_REPO_HOST/scripts")

run_arm() { # run_arm <line> <gpu list>
  local line=$1 gl=$2 arm=${1%%|*} envw=${1#*|} ao w; ao=$OUT/$arm; mkdir -p "$ao" && chmod 777 "$ao"
  local envs=(--env "ND_DIFF_OUT=$(rt_path "$ao")" --env "PYTHONPATH=$TAP_PP" --env PYTHONUNBUFFERED=1)
  [ -n "${TAP:-}" ] && envs+=(--env "ND_DIFF_TAP=$TAP")
  [ -n "${FUNC:-}" ] && envs+=(--env "ND_DIFF_FUNC=$FUNC")
  if [ -n "${MODULES:-}" ]; then
    envs+=(--env "ND_DIFF_MODULES=$MODULES")
    for w in ${LOCATE_ENV:-}; do envs+=(--env "$w"); done
  fi
  # shellcheck disable=SC2086
  for w in $envw; do envs+=(--env "$(eval "printf '%s' $w")"); done
  "$ND" run --gpus "$gl" --name "nd-diff-${SLURM_JOB_ID:-$$}-$arm" "${envs[@]}" -- \
    "echo DIFF_ARM $arm env=[$envw]; $CMD" > "$OUT/$arm.log" 2>&1
  local rc=$? n; n=$(grep -cv '"kind": "meta"' "$ao/outputs.jsonl" 2>/dev/null; true)
  echo "DIFF_ARM_DONE $arm rc=$rc records=${n:-0} tap_errors=$(grep -c 'tap failed\|patching .* failed' "$OUT/$arm.log"; true)"
}

# batches of SLOTS arms, slot s = GPUs s*GPR .. s*GPR+GPR-1 (a fixed loop: the site's sbatch filter rejects
# open-ended waits)
n=${#ARMLINES[@]}
for b in $(seq 0 "$SLOTS" $((n - 1))); do
  for s in $(seq 0 $((SLOTS - 1))); do
    k=$((b + s)); [ "$k" -lt "$n" ] || break
    run_arm "${ARMLINES[$k]}" "$(seq -s, $((s * GPR)) $((s * GPR + GPR - 1)))" &
  done
  wait
done

if [ -n "${MODULES:-}" ]; then
  a=${ARMS%% *}; r=${ARMS##* }
  "$PY" "$ND_REPO_HOST/scripts/output_diff.py" --modules "$OUT/$a/modules.jsonl" --modules-ref "$OUT/$r/modules.jsonl" \
    ${DTYPE:+--dtype "$DTYPE"} ${MODULE_THR:+--module-thr "$MODULE_THR"} | tee "$OUT/locate.txt"
  v=$(sed -n 's/^ND_DIFF_LOCATE verdict=\([A-Z_]*\).*/\1/p' "$OUT/locate.txt")
  m=$(sed -n 's/^ND_DIFF_LOCATE .* module=\([^ ]*\).*/\1/p' "$OUT/locate.txt")
  echo "DIFF_LOCATE_RESULT=${v:-ERROR}${m:+ module=$m} dir=$OUT"
  exit 0
fi

case " ${ARMS:-A REF} " in *" A "*) ;; *) echo "DIFF_WAVE_RESULT=RAN switch=- dir=$OUT  (no A arm)"; exit 0 ;; esac
case " ${ARMS:-A REF} " in *" REF "*) ;; *) echo "DIFF_WAVE_RESULT=RAN switch=- dir=$OUT  (no REF arm: nothing to compare -- check mode reads each arm's log)"; exit 0 ;; esac
args=(--a "$OUT/A/outputs.jsonl" --ref "$OUT/REF/outputs.jsonl" --json "$OUT/diff.json")
[ -s "$OUT/A2/outputs.jsonl" ] && args+=(--a2 "$OUT/A2/outputs.jsonl")
for l in "${ARMLINES[@]}"; do
  a=${l%%|*}; case $a in flip_*) [ -s "$OUT/$a/outputs.jsonl" ] && args+=(--arm "${a#flip_}=$OUT/$a/outputs.jsonl") ;; esac
done
[ -n "${DTYPE:-}" ] && args+=(--dtype "$DTYPE")
for f in "$OUT/A/outputs.jsonl" "$OUT/REF/outputs.jsonl"; do
  [ -s "$f" ] || { echo "DIFF_WAVE_RESULT=NO_OUTPUTS switch=- dir=$OUT  ($f empty: the tap saw no generate / backward"\
    "-- read ${f%/outputs.jsonl}.log; a hand-written loop: TAP=fwd, or FUNC=<the function returning the outputs>)"; exit 0; }
done
"$PY" "$ND_REPO_HOST/scripts/output_diff.py" "${args[@]}" | tee "$OUT/diff.txt"
v=$(sed -n 's/^ND_DIFF verdict=\([A-Z_]*\).*/\1/p' "$OUT/diff.txt")
sw=$(sed -n 's/^ND_DIFF_SWITCH=\([^ ]*\).*/\1/p' "$OUT/diff.txt")
echo "DIFF_WAVE_RESULT=${v:-ERROR} switch=${sw:--} dir=$OUT"
