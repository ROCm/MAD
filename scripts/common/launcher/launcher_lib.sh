# What every multi-node launcher (scripts/<launcher>/run_*.slurm) does around its own
# settings, once: pick its nodes, find their addresses, run the node program on each
# through the dispatcher, and give the job's verdict.
#
#   launcher_select_nodes N [JOB_NAME]  the first N nodes of the allocation, in the
#                                       scheduler's order; the SLURM_* view narrowed
#                                       to them (SELECTED_NODES, SELECTED_NODELIST)
#   launcher_node_ips [SUBNET_PREFIX]   each selected node's address (IPS, IPADDRS),
#                                       MASTER_NODE / MASTER_ADDR, and ROUTER_ADDR when
#                                       ROUTER_NODE=1; with a prefix, the address in it
#   launcher_run_nodes NODE_SCRIPT [CLEANUP_COMMAND]
#                                       record the resolved config, run NODE_SCRIPT on
#                                       every selected node, then CLEANUP_COMMAND; the
#                                       nodes' status in LAUNCHER_RC
#   launcher_verdict [REQUIRE_CSV]      non-leaders exit with their own node's status;
#                                       the leader publishes perf.csv and decides
#
# Needs cluster.sh (cluster_expand_nodelist, cluster_job_verdict), dispatch.sh and
# config/layers.sh sourced, and LOG_PATH, SLURM_JOB_ID, MODEL_NAME set.

# launcher_select_nodes N [JOB_NAME]
# Nodes are kept in the scheduler's order: the ranks srun (or the array index, or the
# pod index) gives the nodes follow it, and IPADDRS must too. (The SGLang launcher
# used to sort them by name, which puts node10 before node2.)
launcher_select_nodes() {
    local n="$1" job_name="${2:-}" full avail
    full="$(cluster_expand_nodelist "$SLURM_JOB_NODELIST")"
    avail="$(printf '%s\n' "$full" | grep -c .)"
    echo "Allocation: ${avail} node(s): $(printf '%s\n' "$full" | paste -sd,)"
    # The node count comes from the topology, not the allocation, so a short allocation
    # used to be truncated into a silently wrong topology; it is refused instead.
    if [ "$avail" -lt "$n" ]; then
        echo "Error: topology needs ${n} nodes but the allocation has ${avail}." >&2
        echo "  Submit with --nodes=${n} (madengine: the card's slurm.nodes)." >&2
        exit 1
    fi
    SELECTED_NODES="$(printf '%s\n' "$full" | head -n "$n")"
    SELECTED_NODELIST="$(printf '%s\n' "$SELECTED_NODES" | paste -sd,)"
    export SLURM_NNODES="$n" SLURM_NTASKS="$n" SLURM_JOB_NUM_NODES="$n" SLURM_NPROCS="$n"
    export SLURM_JOB_NODELIST="$SELECTED_NODELIST" SLURM_NODELIST="$SELECTED_NODELIST"
    export SLURM_TASKS_PER_NODE="1(x$n)" SLURM_NTASKS_PER_NODE=1
    export SLURM_JOBID="${SLURM_JOBID:-$SLURM_JOB_ID}"
    export SLURM_JOB_CPUS_PER_NODE="${SLURM_JOB_CPUS_PER_NODE:-}" SLURM_SUBMIT_HOST="${SLURM_SUBMIT_HOST:-}"
    export SLURM_CLUSTER_NAME="${SLURM_CLUSTER_NAME:-}" SLURM_JOB_PARTITION="${SLURM_JOB_PARTITION:-}"
    export SLURM_JOB_ACCOUNT="${SLURM_JOB_ACCOUNT:-}"
    [ -n "$job_name" ] && export SLURM_JOB_NAME="${SLURM_JOB_NAME:-$job_name}"
    echo "Selected ${n} node(s): ${SELECTED_NODELIST}"
}

# launcher_node_ips [SUBNET_PREFIX]
# A node can have several addresses; the first is used, or with SUBNET_PREFIX (a literal
# prefix such as "10.158.") the first one in it when there is one.
launcher_node_ips() {
    local prefix="${1:-}" re="" node raw ip
    [ -n "$prefix" ] && re="^$(printf '%s' "$prefix" | sed 's/[.[\*^$]/\\&/g')"
    IPS=()
    for node in $SELECTED_NODES; do
        raw="$(dispatch_node_addresses "$node")"
        ip=""
        # (|| true: no address in the subnet is not an error, and callers may run with
        # set -e -o pipefail.)
        if [ -n "$re" ]; then ip="$(printf '%s\n' $raw | grep -E "$re" | head -1 || true)"; fi
        if [ -z "$ip" ]; then ip="$(printf '%s\n' $raw | head -1 || true)"; fi
        if [ -z "$ip" ]; then
            echo "Error: no address for node '${node}' (got: '${raw}')" >&2
            exit 1
        fi
        IPS+=("$ip")
    done
    MASTER_NODE="$(printf '%s\n' "$SELECTED_NODES" | head -n 1)"
    MASTER_ADDR="${IPS[0]}"
    IPADDRS="$(IFS=,; echo "${IPS[*]}")"
    echo "Node addresses: ${IPADDRS} (master ${MASTER_NODE} ${MASTER_ADDR})"
    # The router's address when it has a node of its own (the last one); unset otherwise,
    # and the role scripts use MASTER_ADDR.
    if [ "${ROUTER_NODE:-0}" = "1" ]; then
        export ROUTER_ADDR="${IPS[-1]}"
        echo "Router node: ${ROUTER_ADDR}"
    fi
}

# launcher_run_nodes NODE_SCRIPT [CLEANUP_COMMAND]
launcher_run_nodes() {
    local script="$1" cleanup="${2:-}"
    dispatch_is_leader && config_write "${LOG_PATH}/${SLURM_JOB_ID}/config.resolved.json"
    LAUNCHER_RC=0
    dispatch_run node "$SELECTED_NODELIST" bash "$script" || LAUNCHER_RC=$?
    if [ -n "$cleanup" ]; then
        dispatch_run cleanup "$SELECTED_NODELIST" bash -c "$cleanup"
    fi
    return 0
}

# launcher_verdict [REQUIRE_CSV]
# Under a per-node dispatch (spur-array, k8s-indexed) every node's task gets here;
# rank 0 makes the verdict for the run, and the others end with their own node's
# status. The verdict (cluster_job_verdict): the benchmark writes
# /run_logs/$SLURM_JOB_ID/perf.csv in the container (= $LOG_PATH on the host); it is
# published as perf_<MODEL_NAME>.csv in the job directory, where madengine reads a
# card's multiple_results, before any failure is reported, so the cells that passed
# still reach madengine. The job fails when a node's container exited non-zero, when
# there are no parseable results (REQUIRE_CSV=1, the default: madengine would
# otherwise record "0 successful, 0 failed" as a pass), or when any row is FAILURE.
launcher_verdict() {
    local require="${1:-1}"
    dispatch_is_leader || exit "${LAUNCHER_RC:-1}"
    cluster_job_verdict "${LAUNCHER_RC:-1}" "${LOG_PATH}/${SLURM_JOB_ID}/perf.csv" \
        "$(pwd)/perf_${MODEL_NAME}.csv" "$require"
}
