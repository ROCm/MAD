#!/bin/bash
#SBATCH --job-name=nd-model-ab
#SBATCH --nodes=1
#SBATCH --time=01:00:00
# debug-numeric step 8 (3b): the model, stock vs each env-setting workaround, interleaved (round 1:
# stock, arm1, arm2, ...; round 2: ...), one run at a time on the whole node -- the shape that
# fits a multi-rank job (step4-distributed.md), where two arms cannot share a node.  Single-
# process trainers: put all arms in one wave.sh spec instead.
#
#   CMD='<the model, once: e.g. torchrun ... $ND_REPO/scripts/torchtitan_capture_train.py ...>' \
#   NAN_SIGNAL='<regex>' [SPIKE='<regex>'] [ARMS="VAR=VALUE ..."] [ROUNDS=auto|N] \
#     ND_ENV=<ARTIFACT_DIR>/env.sh scripts/nd submit --name model_ab --gpus 8 [--nodes <CAPTURE_NODES>] --exclusive -- model_ab.sh
#
# Multi-node (NODES > 1, default the job's node count): each run is ONE job over the first NODES
# nodes -- CMD runs once per node (srun), with ND_NNODES / ND_NODE_RANK / ND_MASTER_ADDR for its
# torchrun --nnodes / --node_rank / --master_addr, as a capture.spec `all` line (wave.sh).  Write
# CMD with them and it runs on any node count; the log holds every node's output.
#
# CMD runs through `nd run --gpus all --env <arm>`, so the arm's variable reaches the model in
# any runtime; it gets AB_OUT (a fresh dir per run, runtime path).  Or LAUNCHER=<script> (the old
# form: it must pass AB_ENV into its own container and print `ARM_ENV ...`).  Per run one line:
#   AB <tag> env=<seen> last_step=N signal=<count> spikes=<count> hits=<lines that are either, once> rc=<rc>
# SPIKE (default: torchtitan's `grad_norm: >= 100 or nan/inf`) is counted on step lines.
# ROUNDS=auto (default): round 2 only when round 1 is not decisive -- decisive = stock shows
# >= 3 hits (signal lines + spikes: the power floor) and every arm ran at least as many steps
# with 0.  The second round of a decisive A/B re-measured what round 1 had settled and was half
# of step 8's wall clock in every qwen3 MoE run (2026-11-13/14).  A number = that many rounds.
set -u
# shellcheck disable=SC1091
source "${ND_REPO_HOST:?run through nd submit, or export ND_REPO_HOST}/scripts/env/lib.sh"
nd_job_init
[ -n "${CMD:-}${LAUNCHER:-}" ] || nd_die "CMD (or LAUNCHER) is required"
: "${NAN_SIGNAL:?NAN_SIGNAL=<regex of the NaN line of the model>}"
[ -n "${SPIKE:-}" ] || SPIKE='grad_norm: +([1-9][0-9]{2,}|nan|inf)'   # no ${:-}: its } would end at {2,}
ARMS=${ARMS:-"TENSILE_FIXED_WGMXCC=1"} ROUNDS=${ROUNDS:-auto}
MAXR=$ROUNDS; [ "$ROUNDS" = auto ] && MAXR=2
NODES=${NODES:-${SLURM_JOB_NUM_NODES:-1}}
[ "$NODES" = 1 ] || [ -n "${CMD:-}" ] || nd_die "NODES=$NODES needs CMD (a LAUNCHER runs on this node only)"
nd_multinode "$NODES"
echo "model A/B: nodes=$NODES${ND_LAUNCH[*]:+ ($ND_HOSTS)} arms=[STOCK $ARMS] rounds=$ROUNDS"
OUT=${OUT:-${ND_ART_HOST:-$PWD}/model_ab/${SLURM_JOB_ID:-$$}}
mkdir -p "$OUT" && chmod 777 "$OUT"
for r in $(seq 1 "$MAXR"); do
  decisive=1 stock_hits=0 stock_last=0
  for arm in STOCK $ARMS; do
    # the value too: arms that set one variable to two values shared a log and dir (2026-11-27)
    tag=r${r}_$(printf %s "$arm" | tr -c 'A-Za-z0-9_.-' '_'); abo=$OUT/$tag; mkdir -p "$abo" && chmod 777 "$abo"
    if [ -n "${CMD:-}" ]; then
      envs=(--env "AB_OUT=$(rt_path "$abo")"); [ "$arm" = STOCK ] || envs+=(--env "$arm")
      "${ND_LAUNCH[@]}" "$ND" run --gpus all --name "nd-ab-${SLURM_JOB_ID:-$$}-$tag" "${envs[@]}" -- \
        "echo ARM_ENV ${arm/STOCK/stock}; $CMD" > "$OUT/$tag.log" 2>&1; rc=$?
    else
      if [ "$arm" = STOCK ]; then export AB_ENV=; else export AB_ENV=$arm; fi
      AB_OUT=$abo bash "$LAUNCHER" > "$OUT/$tag.log" 2>&1; rc=$?
    fi
    clean=$OUT/$tag.clean; sed 's/\x1b\[[0-9;]*m//g' "$OUT/$tag.log" > "$clean"
    last=$(grep -oE 'step: *[0-9]+' "$clean" | tail -1 | grep -oE '[0-9]+')
    seen=$(grep -m1 -o 'ARM_ENV.*' "$clean" | cut -c9-)
    nsig=$(grep -cE "$NAN_SIGNAL" "$clean")
    nspk=$(grep -E 'step: *[0-9]+' "$clean" | grep -cE "$SPIKE")
    # hits = lines that are a signal OR a spike, each once (a NAN_SIGNAL that includes the grad_norm
    # regex matched every spike line twice: "stock 60 hit(s)" for 30, 2026-11-14)
    nhit=$({ grep -nE "$NAN_SIGNAL" "$clean"; grep -nE 'step: *[0-9]+' "$clean" | grep -E "$SPIKE"; } | cut -d: -f1 | sort -u | wc -l)
    echo "AB $tag env=${seen:-?} last_step=${last:-none} signal=$nsig spikes=$nspk hits=$nhit rc=$rc"
    rm -f "$clean"
    if [ "$arm" = STOCK ]; then stock_hits=$nhit; stock_last=${last:-0}
    elif [ "$nhit" -ne 0 ] || [ "${last:-0}" -lt "$stock_last" ]; then decisive=0; fi
  done
  if [ "$ROUNDS" = auto ] && [ "$r" = 1 ]; then
    if [ "$decisive" = 1 ] && [ "$stock_hits" -ge 3 ]; then
      echo "AB_DECISIVE after round 1: stock $stock_hits hit(s) by step $stock_last, every arm 0 over >= as many steps -- round 2 skipped"
      break
    fi
    echo "AB round 1 not decisive (stock hits $stock_hits, need >= 3 and every arm 0 over >= $stock_last steps): round 2"
  fi
done
echo AB_DONE
