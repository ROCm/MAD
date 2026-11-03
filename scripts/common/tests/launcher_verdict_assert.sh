#!/bin/bash
# Offline check of how a launcher ends a job, and of the colocated launcher end to end.
#   - cluster_job_verdict (cluster.sh): the CSV is published first, then the job fails
#     on a node's non-zero exit, on missing results, or on any FAILURE row.
#   - run_multinode.slurm, run against stand-in srun / docker / hostname / fuser: the
#     container's exit status reaches the job, the teardown runs on a cancel during the
#     run, the barrier port is per job, the PP default is the node count, and the MoRI
#     and fabric env reach the container.
# No cluster, no GPUs, no docker.
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0
_ok()  { printf "  PASS  %s\n" "$1"; pass=$((pass+1)); }
_bad() { printf "  FAIL  %s\n" "$1"; fail=$((fail+1)); }
_is()  { [ "$1" = "$2" ] && _ok "$3" || { _bad "$3"; printf "        got:  %s\n        want: %s\n" "$1" "$2"; }; }
_has() { case "$1" in *"$2"*) _ok "$3" ;; *) _bad "$3"; printf "        missing: %s\n" "$2" ;; esac; }
_hasnot() { case "$1" in *"$2"*) _bad "$3"; printf "        unexpected: %s\n" "$2" ;; *) _ok "$3" ;; esac; }

CSV_OK='model,performance,metric,status\nm,1.0,tok/s,SUCCESS\n'
CSV_FAIL='model,performance,metric,status\nm,1.0,tok/s,SUCCESS\nm,0,tok/s,FAILURE\n'

echo "=== cluster_job_verdict ==="
_verdict() { # srun_rc csv(or empty) [require]
    local d="$TMP/v.$RANDOM"; mkdir -p "$d/logs"
    [ -n "$2" ] && printf "%b" "$2" > "$d/logs/perf.csv"
    env -i PATH="$PATH" HOME="$TMP" bash -c '. "$0" >/dev/null 2>&1; cluster_job_verdict "$1" "$2" "$3" ${4:+"$4"} >/dev/null 2>&1; echo "rc=$?"' \
        "$DIR/common/cluster.sh" "$1" "$d/logs/perf.csv" "$d/perf_M.csv" "${3:-}"
    [ -f "$d/perf_M.csv" ] && echo published || echo unpublished
}
_is "$(_verdict 0 "$CSV_OK" | tr '\n' ' ')"   "rc=0 published "   "all rows SUCCESS: passes, CSV published"
_is "$(_verdict 0 "$CSV_FAIL" | tr '\n' ' ')" "rc=1 published "   "a FAILURE row fails the job, CSV still published"
_is "$(_verdict 137 "$CSV_OK" | tr '\n' ' ')" "rc=137 published " "a node's exit status fails the job, CSV still published"
_is "$(_verdict 0 "" | tr '\n' ' ')"          "rc=1 unpublished " "no results fails the job"
_is "$(_verdict 0 "" 0 | tr '\n' ' ')"        "rc=0 unpublished " "no results passes when none were required"

echo ""
echo "=== cluster.sh leaves PP_SIZE to the colocated launcher ==="
_is "$(env -i PATH="$PATH" HOME="$TMP" bash -c '. "$0" >/dev/null 2>&1; echo "${PP_SIZE-unset}"' "$DIR/common/cluster.sh")" \
    "unset" "cluster.sh does not default PP_SIZE"

echo ""
echo "=== run_multinode.slurm, end to end with stand-ins ==="
B="$TMP/bin"; mkdir -p "$B" "$TMP/model"
cat > "$B/srun" <<'EOF'
#!/bin/bash
# Runs the bash -c body once per node of --nodelist, as SLURM_PROCID 0..n-1.
nl=""; body=""
while [ $# -gt 0 ]; do
    case "$1" in --nodelist=*) nl="${1#*=}" ;; bash) shift; [ "$1" = -c ] && shift; body="$1"; break ;; esac
    shift
done
i=0; rc=0
for n in ${nl//,/ }; do SLURM_PROCID=$i bash -c "$body" || rc=$?; i=$((i+1)); done
exit $rc
EOF
cat > "$B/docker" <<'EOF'
#!/bin/bash
echo "docker $*" >> "$STUB_LOG"
if [ "$1" = run ]; then
    mkdir -p "$LOG_PATH/$SLURM_JOB_ID"
    [ -n "${STUB_CSV:-}" ] && printf "%b" "$STUB_CSV" > "$LOG_PATH/$SLURM_JOB_ID/perf.csv"
    [ -n "${STUB_SLEEP:-}" ] && sleep "$STUB_SLEEP"
    exit "${STUB_DOCKER_RC:-0}"
fi
exit 0
EOF
printf '#!/bin/bash\n[ "$1" = -I ] && echo 10.0.0.1 || echo node\n' > "$B/hostname"
printf '#!/bin/bash\necho "fuser $*" >> "$STUB_LOG"\n' > "$B/fuser"
# `scontrol show hostnames LIST`, for a launcher that expands the nodelist with it.
printf '#!/bin/bash\n[ "$1 $2" = "show hostnames" ] && tr , "\\n" <<<"$3"\n' > "$B/scontrol"
chmod +x "$B"/*

_colo() { # [VAR=value ...] -> runs the launcher; prints its exit status; stand-in log in $TMP/run/stub.log
    rm -rf "$TMP/run"; mkdir -p "$TMP/run/logs"
    ( cd "$TMP/run" && env -i PATH="$B:$PATH" HOME="$TMP" USER=t STUB_LOG="$TMP/run/stub.log" \
        LOG_PATH="$TMP/run/logs" SLURM_JOB_ID=4242 SLURM_NNODES=2 SLURM_JOB_NODELIST=nodeA,nodeB \
        DOCKER_IMAGE_NAME=img MODEL_NAME=M MODEL_PATH="$TMP/model" MAD_GPU_ARCH=gfx950 \
        CLUSTER_SYSFS_IB="$TMP/none" STUB_CSV="$CSV_OK" "$@" \
        bash "$DIR/vllm_multinode/run_multinode.slurm" > "$TMP/run/out.txt" 2>&1; echo $? )
}
_stub() { cat "$TMP/run/stub.log" 2>/dev/null; }

_is "$(_colo)" "0" "2 nodes, no PP_SIZE given: runs (PP defaults to the node count)"
_has "$(_stub)" "-e PP_SIZE=2 " "PP_SIZE=2 reaches the container"
_has "$(_stub)" "-e CONTAINER_BARRIER_PORT=24242 " "barrier port derived from the job id"
_hasnot "$(_stub)" "fuser" "nothing is killed on a fixed port"
_has "$(_stub)" "set -o pipefail" "the container command keeps the server's exit status through tee"
_has "$(cat "$TMP/run/perf_M.csv" 2>/dev/null)" "SUCCESS" "perf CSV published to the job directory"
_hasnot "$(_stub)" "PYTORCH_HIP_ALLOC_CONF" "no MoRI backend: no MoRI platform env"
_hasnot "$(_stub)" "-e NCCL_IB_GID_INDEX=" "unknown fabric: cluster.sh fallbacks are not forwarded"

_is "$(_colo STUB_DOCKER_RC=3)" "3" "a container that exits 3 fails the job with 3"
_has "$(cat "$TMP/run/perf_M.csv" 2>/dev/null)" "SUCCESS" "...and its CSV is still published"
_is "$(_colo STUB_CSV="$CSV_FAIL")" "1" "a FAILURE row fails the job"
_is "$(_colo STUB_CSV=)" "1" "no results fails the job"

_colo ALL2ALL_BACKEND=mori_low_latency PYTORCH_ALLOC_CONF=custom NCCL_SOCKET_IFNAME=ens3 >/dev/null
_has "$(_stub)" "-e PYTORCH_HIP_ALLOC_CONF=expandable_segments:False " "MoRI backend: moriio.env reaches the container"
_has "$(_stub)" "-e PYTORCH_ALLOC_CONF=custom " "...a submit-time value wins over the file"
_has "$(_stub)" "-e MORI_GPU_ARCHS=gfx950 " "...MORI_GPU_ARCHS follows the detected GPU"
_has "$(_stub)" "-e NCCL_SOCKET_IFNAME=ens3 " "a passed fabric setting reaches the container"

# A cancel during the run: the teardown is installed before srun, so it runs.
rm -rf "$TMP/run"; mkdir -p "$TMP/run/logs"
( cd "$TMP/run" && exec env -i PATH="$B:$PATH" HOME="$TMP" USER=t STUB_LOG="$TMP/run/stub.log" \
    LOG_PATH="$TMP/run/logs" SLURM_JOB_ID=4242 SLURM_NNODES=2 SLURM_JOB_NODELIST=nodeA,nodeB \
    DOCKER_IMAGE_NAME=img MODEL_NAME=M MODEL_PATH="$TMP/model" MAD_GPU_ARCH=gfx950 \
    CLUSTER_SYSFS_IB="$TMP/none" STUB_SLEEP=30 \
    bash "$DIR/vllm_multinode/run_multinode.slurm" > "$TMP/run/out.txt" 2>&1 ) &
_lp=$!
for _i in $(seq 1 50); do grep -q "docker run" "$TMP/run/stub.log" 2>/dev/null && break; sleep 0.2; done
kill -TERM "$_lp"; wait "$_lp"; _rc=$?
_is "$_rc" "143" "TERM during the run ends the job with 143"
_has "$(_stub)" "docker stop colocated_M_4242" "TERM during the run stops the containers"
pkill -f "sleep 30" 2>/dev/null

echo ""
echo "=== the NIAH wrapper leaves max_tokens to benchmark_niah.py ==="
_hasnot "$(grep -v '^ *#' "$DIR/vllm_dissag/benchmark_niah.sh")" "NIAH_MAXTOK=" "benchmark_niah.sh does not override NIAH_MAXTOK"
_has "$(grep '^MAXTOK' "$DIR/vllm_dissag/benchmark_niah.py")" '"8192"' "benchmark_niah.py defaults to 8192"

echo "======================================================"
echo "  launcher_verdict_assert: ${pass} passed, ${fail} failed"
echo "======================================================"
[[ "$fail" == "0" ]]
