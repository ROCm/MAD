#!/bin/bash
#SBATCH --job-name=nd-rate
#SBATCH --nodes=1
#SBATCH --time=01:00:00
# debug-numeric step 2 for a model that runs on more than one node: its NaN rate as it runs -- TRIALS
# uncaptured runs of CMD, each ONE job over NODES nodes (step2-rate.md).  A single-node model's
# rate is a wave.sh wave instead (its trials share one node).
#
#   CMD='<the model once, written with $ND_NNODES / $ND_NODE_RANK / $ND_MASTER_ADDR>' \
#   NAN_SIGNAL='<regex>' NODES=<nodes per run> [TRIALS=8] [COPIES=1] \
#     ND_ENV=<ARTIFACT_DIR>/env.sh scripts/nd submit --name rate --gpus 8 --nodes <NODES x groups> \
#       --exclusive --step step2 -- rate.sh
#
# The job's nodes split into groups of NODES (job nodes / NODES groups; a remainder idles); the
# groups run trials side by side, each starting the next trial when its last one ends -- TRIALS
# runs in ceil(TRIALS / groups) rounds.  Each run is one `nd run --gpus all` per node of its group
# (nd_multinode), with ND_NNODES / ND_NODE_RANK / ND_MASTER_ADDR, MASTER_PORT, RUN_OUT (a fresh
# dir, runtime path), RUN_TRIAL and RUN_COPY -- write CMD's torchrun with --master_port=$MASTER_PORT.
#
# COPIES=k (step 3 for a multi-node run, step3-raise.md): k copies of the same job at once on the
# SAME group of nodes -- k processes per GPU, the multi-node form of oversubscribing the GPUs.
# Each copy is one scored run (its own MASTER_PORT 29500 + 10 x copy, RUN_OUT, log), so TRIALS
# runs take ceil(TRIALS / (groups x k)) rounds and the rate is the rate at k per GPU: carry k
# into the capture as NAN_CAPTURE_CONTAINERS_PER_GPU (step1-ask.md, Declare-contention).
#
# Knobs: NAN_ERROR (the model's own error line, wave_summary.py --error), OUT (<ART>/rate/<tag>),
# STEP (ledger step, step2).
#
# Logs: <OUT>/all_<i>.rate.<tag>.<host>.log ending TRIAL_RC=<rc> (wave_summary.py's shape).  Last
# lines: the per-trial wave_summary.py table, then RATE_RESULT=<nan>/<scored> nodes_per_run=<n>
# trials=<m> (excluded: ERROR / DIED runs), also ledgered.
set -uo pipefail
# shellcheck disable=SC1091
source "${ND_REPO_HOST:?run through nd submit, or export ND_REPO_HOST}/scripts/env/lib.sh"
nd_job_init
: "${CMD:?CMD=<the model once>}"
: "${NAN_SIGNAL:?NAN_SIGNAL=<regex of the NaN line of the model>}"
NODES=${NODES:-1} TRIALS=${TRIALS:-8} COPIES=${COPIES:-1} STEP=${STEP:-step2}
[[ $NODES$TRIALS$COPIES =~ ^[0-9]+$ ]] && [ "$NODES" -ge 1 ] && [ "$TRIALS" -ge 1 ] && [ "$COPIES" -ge 1 ] ||
  nd_die "NODES=$NODES TRIALS=$TRIALS COPIES=$COPIES: positive integers"
tag=$ND_JOB_TAG; host=$(hostname -s)
G=$(( ${SLURM_JOB_NUM_NODES:-1} / NODES ))
[ "$G" -ge 1 ] || nd_die "NODES=$NODES per run, but the job has ${SLURM_JOB_NUM_NODES:-1} node(s)"
OUT=${OUT:-${ND_ART_HOST:-$PWD}/rate/$tag}
mkdir -p "$OUT" && chmod 777 "$OUT"
LEDGER=${LEDGER:-${ND_ART_HOST:-/nonexistent}/ledger.md}
LG() { [ -f "$LEDGER" ] && python3 "$(nd_tool ledger.py)" "$LEDGER" "$@" >/dev/null; return 0; }
say() { echo "[$(date '+%F %T')] $*"; }
[ "$COPIES" -gt 1 ] && nd_drop_pid_ns && say "copies on the same GPUs: apptainer without --pid (APPTAINER_FLAGS='$APPTAINER_FLAGS', lib.sh nd_drop_pid_ns)"
say "host=$host tag=$tag runtime=$RUNTIME image=$(nd_image_desc) runs=$TRIALS nodes/run=$NODES groups=$G copies/group=$COPIES out=$OUT"

declare -A pids=(); next=1      # pids[<group>] = its running copies
launch() {      # launch <group>: the next runs, up to COPIES of them, side by side on its nodes
  local s=$1 c i d p=""
  nd_multinode "$NODES" $((s * NODES))
  for c in $(seq 0 $((COPIES - 1))); do
    [ "$next" -le "$TRIALS" ] || break
    i=$next; next=$((next + 1)); d=$OUT/t$i
    mkdir -p "$d" && chmod 777 "$d"
    say "run $i on group $s ($ND_HOSTS) copy $c"
    nd_bg_trial "$OUT/all_$i.rate.$tag.$host.log" "${ND_LAUNCH[@]}" "$ND" run --gpus all \
      --name "nd-rate-$tag-$i" --env "MASTER_PORT=$((29500 + 10 * c))" --env "RUN_OUT=$(rt_path "$d")" \
      --env "RUN_TRIAL=$i" --env "RUN_COPY=$c" -- "$CMD"
    p+=" $!"
  done
  pids[$s]=$p
}
alive() { local q; for q in $1; do kill -0 "$q" 2>/dev/null && return 0; done; return 1; }
# shellcheck disable=SC2086
trap 'say "TERM -- stopping the runs"; kill -TERM ${pids[*]} 2>/dev/null' TERM INT
while :; do
  busy=0
  for s in $(seq 0 $((G - 1))); do
    if alive "${pids[$s]:-}"; then busy=1
    elif [ "$next" -le "$TRIALS" ]; then launch "$s"; busy=1
    fi
  done
  [ "$busy" = 1 ] || break
  sleep 5 & wait $!
done

python3 "$(nd_tool wave_summary.py)" "$OUT"/all_*.rate.$tag.$host.log --signal "$NAN_SIGNAL" \
    ${NAN_ERROR:+--error "$NAN_ERROR"} >"$OUT/summary.txt" 2>&1
cat "$OUT/summary.txt"
r=$(sed -nE 's/^ +rate +rate ([0-9]+\/[0-9]+) \(excl\. ([0-9]+) ERROR, ([0-9]+) DIED.*/\1 err=\2 died=\3/p' "$OUT/summary.txt" | head -1)
steps=$(sed -nE 's/^== NaN step: *//p' "$OUT/summary.txt" | head -1)
res="${r%% *} nodes_per_run=$NODES copies=$COPIES runs=$TRIALS ${r#* }${steps:+ nan_steps: $steps}"
LG note "$STEP" "RATE_RESULT=$res (rate.sh job $tag, $G group(s) of $NODES node(s), $COPIES per group at once) -> ${OUT#"${ND_ART_HOST:-}"/}/summary.txt"
echo "RATE_RESULT=${r:+$res}"
