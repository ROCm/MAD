#!/bin/bash
# How a launcher reaches its nodes. Sourced by cluster.sh, so every launcher has it.
#
# A launcher needs four things from the scheduler, and only these go through here:
#   dispatch_node_addresses NODE        that node's `hostname -I`
#   dispatch_gather STEP CMD...         run CMD once on every node of the run; print
#                                       every node's output, in rank order
#   dispatch_run STEP NODELIST CMD...   run CMD once per node of NODELIST, as ranks
#                                       0..n-1 (SLURM_PROCID); the worst exit status
#   dispatch_is_leader                  where the job-level steps (the verdict) run
# STEP names the step; the srun dispatcher ignores it.
#
# MAD_DISPATCH selects how; when it is unset, a job-array task means spur-array:
#
#   slurm-srun  One allocation spanning every node; each call is the srun it always
#               was. The default, and byte-for-byte what the launchers did before.
#   spur-array  One single-node task per node (sbatch --array=0..N-1, as madengine's
#               Spur backend submits): srun cannot fan out across nodes on Spur. The
#               launcher runs once per task, all tasks the same code; the tasks meet
#               in a rendezvous directory on shared storage (MAD_RENDEZVOUS_DIR,
#               default $LOG_PATH/rendezvous/<array job id>). Each task publishes its
#               hostname and addresses, then SLURM_JOB_ID / SLURM_NNODES /
#               SLURM_JOB_NODELIST / SLURM_PROCID describe the whole run, as they
#               would under one allocation. A gather is also a barrier: every task
#               sees every task's output, so all make the same decisions. A run step
#               runs this task's own rank; rank 0 then waits for every rank's status
#               and makes the verdict. Node count: MAD_NNODES, else the array size,
#               else SLURM_NNODES when a wrapper (madengine) set it above 1.
#               MAD_RENDEZVOUS_TIMEOUT (default 900 s) bounds each wait for the
#               other tasks, except the wait for a run step, which lasts as long as
#               the run.

dispatch_mode() {
    if [ -n "${MAD_DISPATCH:-}" ]; then
        echo "$MAD_DISPATCH"
    elif [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
        echo spur-array
    else
        echo slurm-srun
    fi
}

dispatch_is_leader() {
    [ "$(dispatch_mode)" != spur-array ] || [ "${_DISPATCH_RANK:-${SLURM_ARRAY_TASK_ID:-0}}" = 0 ]
}

dispatch_node_addresses() {  # NODE
    case "$(dispatch_mode)" in
        spur-array)
            _dispatch_array_init || return 1
            local r
            r="$(_dispatch_rank_of "$1")" || { echo "dispatch: $1 is not a node of this run" >&2; return 1; }
            cat "$_DISPATCH_DIR/addr_rank$r"
            ;;
        *) srun --nodes=1 --ntasks=1 --time=00:20:00 --nodelist="$1" bash -c 'hostname -I' ;;
    esac
}

dispatch_gather() {  # STEP CMD...
    local step="$1"; shift
    case "$(dispatch_mode)" in
        spur-array)
            _dispatch_array_init || return 1
            local out rc=0 r
            out="$("$@")" || rc=$?
            _dispatch_publish "gather_${step}_rank${_DISPATCH_RANK}" "$out"
            _dispatch_wait_all "gather_${step}_rank" "$_DISPATCH_NNODES" "${MAD_RENDEZVOUS_TIMEOUT:-900}" || return 1
            for r in $(seq 0 $((_DISPATCH_NNODES - 1))); do
                cat "$_DISPATCH_DIR/gather_${step}_rank$r"
                echo
            done | sed '/^$/d'
            return "$rc"
            ;;
        *)
            local n="${SLURM_NNODES:-1}"
            srun --nodes="$n" --ntasks="$n" "$@"
            ;;
    esac
}

dispatch_run() {  # STEP NODELIST CMD...
    local step="$1" nodelist="$2"; shift 2
    case "$(dispatch_mode)" in
        spur-array)
            _dispatch_array_init || return 1
            local me i=0 rc=0 n=0 worst=0 r node
            me="$(hostname)"
            for node in ${nodelist//,/ }; do
                if [ "$node" = "$me" ]; then
                    SLURMD_NODENAME="$node" SLURM_PROCID="$i" "$@" || rc=$?
                    _dispatch_publish "run_${step}_rank${i}" "$rc"
                fi
                i=$((i + 1))
            done
            n=$i
            dispatch_is_leader || return "$rc"
            _dispatch_wait_all "run_${step}_rank" "$n" 0 || return 1
            for r in $(seq 0 $((n - 1))); do
                rc="$(cat "$_DISPATCH_DIR/run_${step}_rank$r")"
                [ "$rc" -gt "$worst" ] 2>/dev/null && worst="$rc"
            done
            return "$worst"
            ;;
        *) srun --nodelist="$nodelist" "$@" ;;
    esac
}

# --- spur-array internals -----------------------------------------------------

_dispatch_array_init() {
    [ -n "${_DISPATCH_DIR:-}" ] && return 0
    local job="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-}}" rank="${SLURM_ARRAY_TASK_ID:-}"
    local n="${MAD_NNODES:-${SLURM_ARRAY_TASK_COUNT:-}}"
    [ -z "$n" ] && [ -n "${SLURM_ARRAY_TASK_MAX:-}" ] && n=$(( SLURM_ARRAY_TASK_MAX + 1 ))
    # A task's own SLURM_NNODES is 1; above 1 it was set by a wrapper that knows the
    # run's size (madengine's Spur backend exports it for every task).
    [ -z "$n" ] && [ "${SLURM_NNODES:-1}" -gt 1 ] 2>/dev/null && n="$SLURM_NNODES"
    if [ -z "$job" ] || [ -z "$rank" ] || [ -z "$n" ]; then
        echo "dispatch: spur-array needs SLURM_ARRAY_JOB_ID, SLURM_ARRAY_TASK_ID and MAD_NNODES (or the array size)" >&2
        return 1
    fi
    _DISPATCH_DIR="${MAD_RENDEZVOUS_DIR:-${LOG_PATH:?dispatch: LOG_PATH must be set (shared storage)}/rendezvous/${job}}"
    _DISPATCH_RANK="$rank"
    _DISPATCH_NNODES="$n"
    mkdir -p "$_DISPATCH_DIR" || return 1
    _dispatch_publish "host_rank$rank" "$(hostname)"
    _dispatch_publish "addr_rank$rank" "$(hostname -I)"
    _dispatch_wait_all host_rank "$n" "${MAD_RENDEZVOUS_TIMEOUT:-900}" || return 1
    local r hosts=""
    for r in $(seq 0 $((n - 1))); do hosts="${hosts:+$hosts,}$(cat "$_DISPATCH_DIR/host_rank$r")"; done
    # The whole run, as one allocation would describe it.
    export SLURM_JOB_ID="$job" SLURM_PROCID="$rank" SLURM_NNODES="$n" SLURM_JOB_NODELIST="$hosts"
    export _DISPATCH_DIR _DISPATCH_RANK _DISPATCH_NNODES
    echo "[dispatch] spur-array: rank ${rank} of ${n} on $(hostname); nodes ${hosts}; rendezvous ${_DISPATCH_DIR}"
}

_dispatch_publish() {  # NAME VALUE: written whole, so a reader never sees half of it
    printf '%s\n' "$2" > "$_DISPATCH_DIR/.$1.$$" && mv -f "$_DISPATCH_DIR/.$1.$$" "$_DISPATCH_DIR/$1"
}

_dispatch_wait_all() {  # PREFIX N TIMEOUT_S (0: no limit): every PREFIX0..PREFIX(N-1)
    local prefix="$1" n="$2" timeout="$3" start=$SECONDS r missing
    while :; do
        missing=""
        for r in $(seq 0 $((n - 1))); do
            [ -e "$_DISPATCH_DIR/${prefix}$r" ] || missing="$missing $r"
        done
        [ -z "$missing" ] && return 0
        if [ "$timeout" -gt 0 ] && [ $((SECONDS - start)) -ge "$timeout" ]; then
            echo "dispatch: rank(s)${missing} did not report ${prefix}* within ${timeout}s (${_DISPATCH_DIR})" >&2
            return 1
        fi
        sleep 1
    done
}

_dispatch_rank_of() {  # NODE: its rank, from the published hostnames
    local r
    for r in $(seq 0 $((_DISPATCH_NNODES - 1))); do
        [ "$(cat "$_DISPATCH_DIR/host_rank$r")" = "$1" ] && { echo "$r"; return 0; }
    done
    return 1
}
