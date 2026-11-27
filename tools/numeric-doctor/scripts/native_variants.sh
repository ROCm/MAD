#!/bin/bash
# debug-numeric step 5N (N5): one variant per GPU (RANKS=k: per group of k GPUs), all concurrent,
# one node -- a candidate and its controls / reductions in a single job.
#
#   bash native_variants.sh <ARTIFACT_DIR>/native/repro_<k>.py <variants file>
#
# Variants file: line i (0-based; `#` and blank lines skipped) runs on GPU i as
#   <copies>|<env VAR=VALUE ...>|<candidate args>|<optional co-tenant command>
# e.g.  3|||                                        the candidate, 3 copies
#       1|||                                        solo
#       3|TORCH_BLAS_PREFER_HIPBLASLT=0||           a library toggle -- it may not reroute the op
#                                                   (fp32 TF32 mm kept hitting: 889 vs 991/1M, 2026-11-11):
#                                                   check the dispatch in a hipBLASLt call log before reading it
#       1||--mode op|python3 .../transformer_hog.py --mode fwdbwd --procs 2
# $VARS in the env and args columns expand inside the runtime ($ND_ART/..., $ND_REPO/...).
# The co-tenant starts once on its GPU before the copies and is killed after them.
#
# RANKS=k > 1 (a multi-rank candidate), as native_run.sh: the node's NGPUS GPUs form NGPUS/k
# groups of k (0..k-1, k..2k-1, ...) and line i runs on group i -- each copy ONE
# `torchrun --nproc_per_node k` (HIP_VISIBLE_DEVICES=<group>, its own --master_port, given
# --copy <c> --gpu <first GPU of the group>), the co-tenant once per group; at most NGPUS/k lines.
# (2026-11-23, vLLM TP=2: no RANKS here, so a 2-rank candidate's arms ran one after another
# through native_run.sh by hand.)
#
# Runs in the candidate's directory; logs in out.<jobid>.<host>/ (.2, .3, ... for later calls in
# the same job, as native_run.sh).  Candidate contract as native_run.sh: only the first
# NATIVE_RESULT line of each copy's log is counted.  Env: NGPUS (8), RANKS (1).
# Prints one `VARIANT_TOTAL gpu=<i, or the group's GPUs i0,i1,..> <hits>/<runs> results=<n>
# variant=<line>` per line, `NO_RESULT <log>` / `TRACEBACK <log>` per bad copy, then `VARIANTS_DONE`.
#
# Any runtime: ND_ENV=<ARTIFACT_DIR>/env.sh scripts/nd submit --name cand<k> --gpus 8 --exclusive -- exec.sh \
#        -- '[RANKS=<k>] bash $ND_REPO/scripts/native_variants.sh $ND_ART/native/repro_<k>.py $ND_ART/native/variants_<k>.txt'   (paths as $ND_ART/...)
set -uo pipefail
cand=$(readlink -f -- "$1"); vf=$(readlink -f -- "$2")
cd -- "$(dirname -- "$cand")" || exit 2
NGPUS=${NGPUS:-8}; RANKS=${RANKS:-1}
[ "$RANKS" -ge 1 ] && [ $((NGPUS % RANKS)) = 0 ] || { echo "NGPUS=$NGPUS is not a multiple of RANKS=$RANKS"; exit 2; }
out=out.${ND_JOB_ID:-${SLURM_JOB_ID:-$(date +%s).$$}}.$(hostname -s)
if [ -e "$out" ]; then n=2; while [ -e "$out.$n" ]; do n=$((n + 1)); done; out=$out.$n; fi
mkdir -p "$out"
units=(); for ((g = 0; g < NGPUS; g += RANKS)); do
  u=$g; for ((r = 1; r < RANKS; r++)); do u+=",$((g + r))"; done; units+=("$u")
done
pids=(); cpids=(); i=0; port=29600
while IFS='|' read -r copies envs args cot; do
  [[ -z "$copies" || "$copies" == \#* ]] && continue
  [ "$i" -lt "${#units[@]}" ] || { echo "more variant lines than GPU groups (${#units[@]}: NGPUS=$NGPUS RANKS=$RANKS)"; exit 2; }
  u=${units[$i]}
  eval "envs=\"$envs\" args=\"$args\""     # $ND_ART, $ND_REPO, ... expand, as in a wave.sh spec
  if [[ -n "${cot:-}" ]]; then
    HIP_VISIBLE_DEVICES=$u setsid bash -c "$cot" >"$out/cotenant_g$i.log" 2>&1 &
    cpids+=($!)
  fi
  for ((c = 0; c < copies; c++)); do
    if [ "$RANKS" = 1 ]; then
      env HIP_VISIBLE_DEVICES=$u $envs python3 "$cand" --copy "$c" --gpu "$u" $args >"$out/g${i}_c${c}.log" 2>&1 &
    else
      env HIP_VISIBLE_DEVICES=$u $envs torchrun --nproc_per_node "$RANKS" --master_port "$port" \
        "$cand" --copy "$c" --gpu "${u%%,*}" $args >"$out/g${i}_c${c}.log" 2>&1 &
      port=$((port + 1))
    fi
    pids+=($!)
  done
  echo "$copies|$envs|$args|${cot:-}" >"$out/variant_g$i.txt"
  i=$((i + 1))
done <"$vf"
wait "${pids[@]}"
for p in "${cpids[@]}"; do kill -- -"$p" 2>/dev/null; done
for ((j = 0; j < i; j++)); do
  v=$(cat "$out/variant_g$j.txt")
  n=${v%%|*}; [ "$n" -gt 0 ] || { echo "VARIANT_TOTAL gpu=${units[$j]} 0/0 results=0 variant=$v"; continue; }
  logs=(); for ((c = 0; c < n; c++)); do logs+=("$out/g${j}_c${c}.log"); done
  r=$(awk -F'[=/ ]' 'FNR == 1 {seen = 0} /^NATIVE_RESULT=/ && !seen {h += $2; n += $3; k++; seen = 1}
    END {printf "%d/%d results=%d", h, n, k}' "${logs[@]}")
  echo "VARIANT_TOTAL gpu=${units[$j]} $r variant=$v"
done
grep -L '^NATIVE_RESULT=' "$out"/g*_c*.log | sed 's/^/NO_RESULT /'
grep -l Traceback "$out"/g*_c*.log | sed 's/^/TRACEBACK /'
echo "VARIANTS_DONE out=$out"
