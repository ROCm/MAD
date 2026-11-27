# SCHED=local -- sourced by lib.sh.  Contract (every sched_*.sh):
#   sch_submit <name> <log> <time> <exclusive 0|1> <gpus> <nodes> <job> [args]  -> prints the job id
#   sch_state <id>   one of PENDING RUNNING DONE FAILED (slurm also HELD: never starts by itself), then the start time
#   sch_stop <id>
#   sch_node_state <host>       "idle" when a job would start there now
#   sch_node_only_mine <host>   true when every job on the node is this user's
#   sch_job_log <id>            the job's log, for one submitted outside nd (nd adopt / nd wait <id>)
#
# A detached process group per job (setsid, nohup, stdin closed) that survives the agent's
# shell; the id is the REAL pid of the job (the child writes its own pid before exec -- never a
# wrapper's or a pgrep self-match).  Ported from wave_ctl.sh.

sch_submit() {
  local name=$1 log=$2 pidf; shift 5
  [ "$1" = 1 ] || nd_die "SCHED=local runs on this node only: --nodes $1 needs SCHED=slurm"; shift
  [ -z "${ND_NODELIST:-}" ] || [ "${ND_NODELIST%%.*}" = "$(hostname -s)" ] ||
    nd_die "SCHED=local runs on this node ($(hostname -s)) only: --node $ND_NODELIST needs SCHED=slurm"
  pidf=${log%.out}.pid
  rm -f "$pidf"
  setsid nohup bash -c 'echo $$ > "$0"; exec "$@"' "$pidf" bash "$@" > "$log" 2>&1 < /dev/null &
  local i; for i in $(seq 1 50); do [ -s "$pidf" ] && break; sleep 0.1; done
  [ -s "$pidf" ] || nd_die "local job $name did not start; see $log"
  echo "local:$(cat "$pidf")"
}

# sch_node_state <host>: this node is the only one; "idle" when it is this host
sch_node_state() { [ "${1%%.*}" = "$(hostname -s)" ] && echo idle; }

sch_node_only_mine() { return 0; }
sch_job_log() { :; }          # local jobs are always nd's own: their record has the log

sch_state() {
  local pid=${1#local:} rc_line
  if kill -0 "$pid" 2>/dev/null; then echo "RUNNING $(ps -o lstart= -p "$pid" 2>/dev/null | xargs)"; return; fi
  echo "DONE -"
}

sch_stop() {
  local pid=${1#local:} g pgids alive d
  _desc() { local c; for c in $(ps -o pid= --ppid "$1" 2>/dev/null); do echo "$c"; _desc "$c"; done; }
  # the job's own group and every descendant's (torchrun and friends setsid into their own);
  # never by name pattern -- other sessions' jobs share the node
  pgids=$( { ps -o pgid= -p "$pid" 2>/dev/null; for d in $(_desc "$pid"); do ps -o pgid= -p "$d" 2>/dev/null; done; } |
           tr -d ' ' | sort -u | grep -v '^$' )
  [ -n "$pgids" ] || { echo "not running"; return 0; }
  _alive() { alive=""; for g in $pgids; do pgrep -g "$g" >/dev/null 2>&1 && alive="$alive $g"; done; }
  _wait_alive() { for _ in $(seq 1 "$1"); do _alive; [ -z "$alive" ] && return; sleep 1; done; _alive; }
  # 1. TERM the job script alone, as Slurm's `scancel --batch --signal=TERM`: its trap passes the
  #    stop down the chain (nd run -> the command), so the command stops once, gracefully, and the
  #    job records the command's real rc.  TERMing every group at once raced that chain.
  kill -TERM "$pid" 2>/dev/null
  _wait_alive "${ND_STOP_GRACE:-60}"
  # 2. a job without a trap (or a hung one): TERM every group, then KILL
  local termed=$alive
  for g in $alive; do kill -TERM -- "-$g" 2>/dev/null; done
  [ -n "$alive" ] && _wait_alive 30
  for g in $alive; do kill -KILL -- "-$g" 2>/dev/null; done
  echo "stopped process group(s) $(echo $pgids)${termed:+ (group-TERMed:$termed)}${alive:+ (KILLed:$alive)}"
}
