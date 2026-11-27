#!/bin/bash
# debug-numeric step 5N (N4): run one native candidate under load on every GPU of the node.
#
#   bash native_run.sh <ARTIFACT_DIR>/native/repro_<k>.py [args for the candidate...]
#
# Starts COPIES copies of the candidate per GPU (HIP_VISIBLE_DEVICES=<g>, each given
# --copy <c> --gpu <g>), plus COTENANT once per GPU when set, waits for the copies,
# kills the co-tenants, and sums the candidates' last lines.  Runs in the candidate's
# directory; per-copy logs in out.<jobid>.<host>/ beside it (.2, .3, ... for later calls in the same job).
#
# RANKS=k > 1 (a multi-rank candidate, step5n-native.md *World size > 1*): the node's NGPUS GPUs
# form NGPUS/k groups of k (0..k-1, k..2k-1, ...); each copy is ONE `torchrun --nproc_per_node k`
# on its group (HIP_VISIBLE_DEVICES=<group>, its own --master_port), given --copy <c> --gpu
# <first GPU of the group>; COTENANT starts once per group.
#
# Candidate contract (step5n-native.md N3): accepts --copy/--gpu, prints `HIT ...`
# lines and ends with `NATIVE_RESULT=<hits>/<runs> ...` -- multi-rank: rank 0 prints it,
# counting an iteration as a hit when ANY rank's suspect output is bad (reduce the flag
# across ranks); only the first NATIVE_RESULT line of each copy's log is counted.
# Env: COPIES (per GPU / group, default 3 = capture's processes per GPU + 1), NGPUS (8), RANKS (1),
#      COTENANT (command, started once per GPU / group; killed at the end),
#      COTENANT_WARMUP (seconds before the copies start, default 30).
# Last line: NATIVE_TOTAL=<hits>/<runs> copies=<c>x<gpus or groups> [ranks=<k>] out=<dir>
#
# Any runtime: ND_ENV=<ARTIFACT_DIR>/env.sh scripts/nd submit --name cand<k> --gpus 8 --exclusive -- exec.sh \
#        -- 'bash $ND_REPO/scripts/native_run.sh $ND_ART/native/repro_<k>.py'   (paths as $ND_ART/...)
set -uo pipefail
cand=$(readlink -f -- "$1"); shift
cd -- "$(dirname -- "$cand")" || exit 2
COPIES=${COPIES:-3}; NGPUS=${NGPUS:-8}; RANKS=${RANKS:-1}
[ "$RANKS" -ge 1 ] && [ $((NGPUS % RANKS)) = 0 ] || { echo "NGPUS=$NGPUS is not a multiple of RANKS=$RANKS"; exit 2; }
out=out.${ND_JOB_ID:-${SLURM_JOB_ID:-$(date +%s).$$}}.$(hostname -s)
# several calls in one job (arms run one after another) each get their own dir -- one dir per job
# let every arm overwrite the last one's copy logs (2026-11-23: the stock arm's SIGNATURE lines lost)
if [ -e "$out" ]; then n=2; while [ -e "$out.$n" ]; do n=$((n + 1)); done; out=$out.$n; fi
mkdir -p "$out"
units=(); for ((g = 0; g < NGPUS; g += RANKS)); do
  u=$g; for ((r = 1; r < RANKS; r++)); do u+=",$((g + r))"; done; units+=("$u")
done
cot=()
if [[ -n "${COTENANT:-}" ]]; then
  for u in "${units[@]}"; do
    HIP_VISIBLE_DEVICES=$u setsid bash -c "$COTENANT" >"$out/cotenant_g${u//,/-}.log" 2>&1 &
    cot+=($!)
  done
  sleep "${COTENANT_WARMUP:-30}"
fi
pids=(); port=29600
for u in "${units[@]}"; do
  for ((c = 0; c < COPIES; c++)); do
    log=$out/g${u//,/-}_c${c}.log
    if [ "$RANKS" = 1 ]; then
      HIP_VISIBLE_DEVICES=$u python3 "$cand" --copy "$c" --gpu "$u" "$@" >"$log" 2>&1 &
    else
      HIP_VISIBLE_DEVICES=$u torchrun --nproc_per_node "$RANKS" --master_port "$port" \
        "$cand" --copy "$c" --gpu "${u%%,*}" "$@" >"$log" 2>&1 &
      port=$((port + 1))
    fi
    pids+=($!)
  done
done
wait "${pids[@]}"
for p in "${cot[@]}"; do kill -- -"$p" 2>/dev/null; done
grep -h '^HIT ' "$out"/g*_c*.log | head -20
grep -L '^NATIVE_RESULT=' "$out"/g*_c*.log | sed 's/^/NO_RESULT /'
grep -l Traceback "$out"/g*_c*.log | sed 's/^/TRACEBACK /'
awk -F'[=/ ]' 'FNR == 1 {seen = 0} /^NATIVE_RESULT=/ && !seen {h += $2; n += $3; seen = 1}
  END {printf "NATIVE_TOTAL=%d/%d copies=%sx%s%s out=%s\n", h, n, c, u, (k > 1 ? " ranks=" k : ""), o}' \
  c="$COPIES" u="${#units[@]}" k="$RANKS" o="$out" "$out"/g*_c*.log
