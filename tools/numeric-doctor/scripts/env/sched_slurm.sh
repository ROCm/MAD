# SCHED=slurm -- sourced by lib.sh; contract in sched_local.sh.
#
# Profile knobs: SLURM_ARGS (account, partition, ... e.g. "--account=<account> --partition=<partition>"),
# SLURM_GPUS_FLAG (how this site requests N GPUs on a node, with {n} for the count, e.g.
# "--gpus-per-node={n}"; empty on a site whose nodes come whole), SINGLE_SESSION (1 = this
# login allows one SSH session: `nd wait` refuses -- schedule one-shot `nd status`).
#
# Hygiene that each cost a job: the -o directory exists before sbatch (Slurm opens it at job
# start and a missing one loses the output silently); --signal=B:TERM@60 so the job can
# shut its workload down before the walltime KILL; the job learns the repo and profile from
# the environment (ND_REPO_HOST, ND_ENV), never from $0 -- Slurm runs a spool copy.

sch_submit() {
  local name=$1 log=$2 time=$3 excl=$4 gpus=$5 nodes=$6 args=() g; shift 6
  # shellcheck disable=SC2206
  # --no-requeue: a requeued job (node failure, preemption -- even after it finished,
  # 2026-11-21) reruns its script into the same artifact dir hours later, over the result it wrote.
  # A launch failure now ends FAILED rather than held: resubmit with --exclude <that node>.
  args=(--parsable --job-name="$name" -o "$log" -e "$log" --signal=B:TERM@60 --no-requeue --export=ALL ${SLURM_ARGS:-})
  [ -n "$time" ] && args+=(--time="$time")
  [ "$excl" = 1 ] && args+=(--exclusive)
  [ -n "${ND_NODELIST:-}" ] && args+=(--nodelist="$ND_NODELIST")   # nd submit --node
  [ -n "${ND_EXCLUDE:-}" ] && args+=(--exclude="$ND_EXCLUDE")      # nd submit --exclude
  args+=(--nodes="$nodes")               # overrides the job script's #SBATCH --nodes=1
  # A site with a GPU flag rejects a job without a GPU request: default to 1 [phase 4].
  if [ -n "${SLURM_GPUS_FLAG:-}" ]; then g=${SLURM_GPUS_FLAG//\{n\}/${gpus:-1}}; args+=($g); fi
  local id q a
  if head -c2 "$1" | grep -q '^#!'; then
    id=$(sbatch "${args[@]}" "$@") || nd_die "sbatch failed"
  else          # sbatch insists on a #! line; run a script without one through bash
    q=""; for a in "$@"; do q="$q $(printf '%q' "$a")"; done
    id=$(sbatch "${args[@]}" --wrap "bash$q") || nd_die "sbatch failed"
  fi
  echo "slurm:${id%%;*}"
}

# sch_node_state <host>: "idle" only when a job would start there now -- scontrol's full state,
# since sinfo's %T hides IDLE+COMPLETING+PLANNED (already promised to a pending job); else that
# state, lower-case; empty if unknown
sch_node_state() {
  local s; s=$(scontrol show node "$1" 2>/dev/null | grep -o 'State=[^ ]*' | head -1 | cut -d= -f2)
  case $s in
    "") ;;
    IDLE|IDLE+DYNAMIC_NORM|IDLE+CLOUD) echo idle ;;
    *) tr 'A-Z' 'a-z' <<<"$s" ;;
  esac
}

# sch_node_only_mine <host>: true when every job on the node is this user's (or none is)
sch_node_only_mine() {
  ! squeue -h -w "$1" -o '%u' 2>/dev/null | grep -qvxF "$(id -un)"
}

# sch_job_log <id>: the job's stdout path -- scontrol while Slurm still holds the job, else sacct
# (its StdOut may hold %j / %x / %u patterns, relative to WorkDir)
sch_job_log() {
  local id=${1#slurm:} o w n
  o=$(scontrol show job "$id" 2>/dev/null | grep -o 'StdOut=[^ ]*' | head -1 | cut -d= -f2-)
  if [ -z "$o" ]; then
    IFS='|' read -r o w n < <(sacct -n -P -X -j "$id" -o StdOut,WorkDir,JobName 2>/dev/null | head -1)
    [ -n "$o" ] || return 0
    o=${o//%j/$id}; o=${o//%A/$id}; o=${o//%x/$n}; o=${o//%u/$(id -un)}
    case $o in /*) ;; *) o=$w/$o ;; esac
  fi
  printf '%s\n' "$o"
}

sch_state() {
  local id=${1#slurm:} s r
  IFS='|' read -r s r < <(squeue -h -j "$id" -o '%T %S|%r' 2>/dev/null | head -1)
  # held (`launch failed requeued held`, an admin/user hold): it never starts by itself -- say so
  # rather than PENDING, which once cost 6 min of waiting in 3 of 5 runs (2026-11-13/14)
  if [[ $s == REQUEUE_HOLD* || $s == SPECIAL_EXIT* || ( $s == PENDING* && ${r,,} == *held* ) ]]; then
    # the node the launch failed on: only the requeued record has it (plain sacct says
    # `None assigned`; an agent spent a failed resubmit finding it, 2026-11-14)
    local n; n=$(sacct -n -P -X --duplicates -j "$id" -o NodeList 2>/dev/null | grep -v '^None' | tail -n 1)
    echo "HELD (${r:-$s}${n:+ on $n})"; return
  fi
  if [ -z "$s" ]; then
    local ec
    read -r s ec < <(sacct -n -X -j "$id" -o State%20,ExitCode,Start 2>/dev/null | head -1 | awk '{print $1 " " $3 "|" $2}')
    # One site's accounting records jobs that ran to the end as PREEMPTED with exit 0:0 (2026-11-23: three
    # finished jobs, each with its complete result line, read as FAILED) -- exit 0:0 is a finished job;
    # a really preempted one ends on a signal (0:15 / 0:9)
    case "$s|${ec#*|}" in
      COMPLETED*) s="DONE ${ec%%|*}" ;;
      PREEMPTED*"|0:0") s="DONE ${ec%%|*} (slurm: PREEMPTED, exit 0:0 -- judge the result from the log)" ;;
      "|"*|"") s="DONE -" ;;
      *) s="FAILED $s ${ec%%|*}" ;;
    esac
  fi
  echo "$s"
}

# TERM to the batch shell only, like the walltime warning: the job's trap stops its workload
# gracefully.  A plain scancel TERMs every process at once and, once the batch shell exits,
# Slurm kills the rest before a trainer can save or a copy can report [L2].  A job that
# ignores it still ends at its walltime; `scancel <id>` forces.  A job not running (pending,
# or held after `launch failed requeued held`) has no batch shell to signal: `--batch --signal`
# fails on it (`Job can not be altered now`, 2026-11-14), so it gets a plain scancel.
sch_stop() {
  local id=${1#slurm:} st
  st=$(squeue -h -j "$id" -o '%T' 2>/dev/null | head -1)
  case $st in
    RUNNING|COMPLETING|"") scancel --batch --signal=TERM "$id" && echo "scancel --batch --signal=TERM $id" ;;
    *) scancel "$id" && echo "scancel $id (was $st: nothing running to stop gracefully)" ;;
  esac
}
