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
#               The tasks of a job array are not scheduled together: they start as
#               nodes free up, possibly one after another. So the first meeting
#               (joining) waits MAD_RENDEZVOUS_JOIN_TIMEOUT (default
#               MAD_RENDEZVOUS_TIMEOUT when set, else 3600 s); MAD_RENDEZVOUS_TIMEOUT
#               (default 900 s) bounds each later wait, except the wait for a run
#               step, which lasts as long as the run. A task that gives up leaves an
#               `abort` file with its reason, and every other task's waits stop on it
#               at once. Two ranks on one node (a later task started on the node an
#               earlier one left) end the run at once: every rank must be its own
#               node.
#   k8s-indexed The same, one pod per node of a Kubernetes Indexed Job (set by
#               k8s/mad_k8s_node.sh): rank JOB_COMPLETION_INDEX, run id
#               MAD_K8S_JOB_ID, size MAD_K8S_NNODES, this node's name MAD_K8S_SELF
#               (the pod's; with host networking `hostname` is the Kubernetes
#               node's). The rendezvous is on the shared results volume, under
#               LOG_PATH. The pod is the container: `docker run` in it is the
#               stand-in from k8s/mad_k8s.py.

dispatch_mode() {
    if [ -n "${MAD_DISPATCH:-}" ]; then
        echo "$MAD_DISPATCH"
    elif [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
        echo spur-array
    else
        echo slurm-srun
    fi
}

# A mode where every node runs the launcher as its own task.
_dispatch_per_task() {
    case "$(dispatch_mode)" in spur-array|k8s-indexed) return 0 ;; esac
    return 1
}

dispatch_is_leader() {
    _dispatch_per_task || return 0
    [ "${_DISPATCH_RANK:-${SLURM_ARRAY_TASK_ID:-${JOB_COMPLETION_INDEX:-0}}}" = 0 ]
}

dispatch_node_addresses() {  # NODE
    case "$(dispatch_mode)" in
        spur-array|k8s-indexed)
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
        spur-array|k8s-indexed)
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
        spur-array|k8s-indexed)
            _dispatch_array_init || return 1
            local me i=0 rc=0 n=0 worst=0 r node
            me="$_DISPATCH_SELF"
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

# --- per-task internals (spur-array, k8s-indexed) ----------------------------

_dispatch_array_init() {
    [ -n "${_DISPATCH_DIR:-}" ] && return 0
    local job rank n self mode
    mode="$(dispatch_mode)"
    if [ "$mode" = k8s-indexed ]; then
        job="${MAD_K8S_JOB_ID:-${SLURM_JOB_ID:-}}"
        rank="${JOB_COMPLETION_INDEX:-}"
        n="${MAD_K8S_NNODES:-}"
        self="${MAD_K8S_SELF:-$(hostname)}"
    else
        job="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-}}"
        rank="${SLURM_ARRAY_TASK_ID:-}"
        n="${MAD_NNODES:-${SLURM_ARRAY_TASK_COUNT:-}}"
        [ -z "$n" ] && [ -n "${SLURM_ARRAY_TASK_MAX:-}" ] && n=$(( SLURM_ARRAY_TASK_MAX + 1 ))
        # A task's own SLURM_NNODES is 1; above 1 it was set by a wrapper that knows the
        # run's size (madengine's Spur backend exports it for every task).
        [ -z "$n" ] && [ "${SLURM_NNODES:-1}" -gt 1 ] 2>/dev/null && n="$SLURM_NNODES"
        self="$(hostname)"
    fi
    if [ -z "$job" ] || [ -z "$rank" ] || [ -z "$n" ]; then
        echo "dispatch: ${mode} needs its run id, this task's rank and the node count" \
             "(spur-array: SLURM_ARRAY_JOB_ID, SLURM_ARRAY_TASK_ID, MAD_NNODES or the array size;" \
             "k8s-indexed: MAD_K8S_JOB_ID, JOB_COMPLETION_INDEX, MAD_K8S_NNODES)" >&2
        return 1
    fi
    _DISPATCH_DIR="${MAD_RENDEZVOUS_DIR:-${LOG_PATH:?dispatch: LOG_PATH must be set (shared storage)}/rendezvous/${job}}"
    _DISPATCH_RANK="$rank"
    _DISPATCH_NNODES="$n"
    _DISPATCH_SELF="$self"
    mkdir -p "$_DISPATCH_DIR" || return 1
    _dispatch_publish "host_rank$rank" "$self"
    _dispatch_publish "addr_rank$rank" "$(hostname -I)"
    _dispatch_wait_all host_rank "$n" "${MAD_RENDEZVOUS_JOIN_TIMEOUT:-${MAD_RENDEZVOUS_TIMEOUT:-3600}}" || return 1
    local r h hosts="" seen=" "
    for r in $(seq 0 $((n - 1))); do
        h="$(cat "$_DISPATCH_DIR/host_rank$r")"
        if [[ "$seen" == *" $h "* ]]; then
            _dispatch_abort "two ranks on node ${h}: the tasks did not run together (a later task started" \
                "on the node an earlier one left); every rank must be its own node"
            return 1
        fi
        seen+="$h "
        hosts="${hosts:+$hosts,}$h"
    done
    # The whole run, as one allocation would describe it.
    export SLURM_JOB_ID="$job" SLURM_PROCID="$rank" SLURM_NNODES="$n" SLURM_JOB_NODELIST="$hosts"
    export _DISPATCH_DIR _DISPATCH_RANK _DISPATCH_NNODES _DISPATCH_SELF
    echo "[dispatch] ${mode}: rank ${rank} of ${n} on ${self}; nodes ${hosts}; rendezvous ${_DISPATCH_DIR}"
}

_dispatch_publish() {  # NAME VALUE: written whole, so a reader never sees half of it
    printf '%s\n' "$2" > "$_DISPATCH_DIR/.$1.$$" && mv -f "$_DISPATCH_DIR/.$1.$$" "$_DISPATCH_DIR/$1"
}

# _dispatch_abort REASON...: end the run for every task. The first reason stays.
_dispatch_abort() {
    [ -e "$_DISPATCH_DIR/abort" ] || _dispatch_publish abort "rank ${_DISPATCH_RANK:-?} ($(hostname)): $*"
    echo "dispatch: $(cat "$_DISPATCH_DIR/abort")" >&2
}

_dispatch_wait_all() {  # PREFIX N TIMEOUT_S (0: no limit): every PREFIX0..PREFIX(N-1)
    local prefix="$1" n="$2" timeout="$3" start=$SECONDS r missing
    while :; do
        if [ -e "$_DISPATCH_DIR/abort" ]; then
            echo "dispatch: the run was ended by $(cat "$_DISPATCH_DIR/abort")" >&2
            return 1
        fi
        missing=""
        for r in $(seq 0 $((n - 1))); do
            [ -e "$_DISPATCH_DIR/${prefix}$r" ] || missing="$missing $r"
        done
        [ -z "$missing" ] && return 0
        if [ "$timeout" -gt 0 ] && [ $((SECONDS - start)) -ge "$timeout" ]; then
            _dispatch_abort "rank(s)${missing} did not report ${prefix}* within ${timeout}s (${_DISPATCH_DIR})"
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
