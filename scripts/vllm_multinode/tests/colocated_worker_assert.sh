#!/bin/bash
# Offline check of serve_colocated.sh's worker / head coordination, with a stand-in
# `vllm`: a worker whose server dies before the head's shutdown signal fails and says
# so, the head stops at once on that instead of waiting out its readiness timeout,
# and a worker the head shuts down still exits 0.
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$DIR/serve_colocated.sh"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0
_is()  { [ "$1" = "$2" ] && { printf "  PASS  %s\n" "$3"; pass=$((pass+1)); } || { printf "  FAIL  %s\n        got:  %s\n        want: %s\n" "$3" "$1" "$2"; fail=$((fail+1)); }; }
_has() { case "$1" in *"$2"*) printf "  PASS  %s\n" "$3"; pass=$((pass+1)) ;; *) printf "  FAIL  %s\n        missing: %s\n" "$3" "$2"; fail=$((fail+1)) ;; esac; }

B="$TMP/bin"; mkdir -p "$B" "$TMP/model" "$TMP/shared"
# vllm: VLLM_BEHAVIOUR=die exits 3 after a moment; =serve runs until killed; =ready
# says it is ready, then runs until killed.
cat > "$B/vllm" <<'EOF'
#!/bin/bash
echo "stand-in vllm $*"
case "${VLLM_BEHAVIOUR:-serve}" in
  die) sleep 1; echo "RuntimeError: worker lost its group"; exit 3 ;;
  ready) echo "Application startup complete."; exec sleep 300 ;;
  *)   exec sleep 300 ;;
esac
EOF
chmod +x "$B/vllm"

_run() { # NODE_RANK [VAR=value ...]: serve_colocated.sh as that node, logs under $TMP/logs
    local rank="$1"; shift
    env -i PATH="$B:$PATH" HOME="$TMP" _RUN_LOGS="$TMP/logs" SLURM_JOB_ID=5 NODE_RANK="$rank" NNODES=2 \
        MODEL_PATH="$TMP/model" MODEL_NAME=M MASTER_ADDR=127.0.0.1 IPADDRS=127.0.0.1,127.0.0.2 \
        SHARED_DIR="$TMP/shared" TP_SIZE=8 PP_SIZE=2 GPUS_PER_NODE=8 LOG_WAIT_TIMEOUT_SECONDS=60 "$@" \
        bash "$SCRIPT"
}

echo "=== a worker whose server dies ==="
out="$(_run 1 VLLM_BEHAVIOUR=die 2>&1)"; rc=$?
_is "$rc" "1" "fails instead of exiting 0"
_has "$out" "exited (status 3) before the shutdown signal" "says why"
_has "$out" "RuntimeError: worker lost its group" "...with its server's log tail"
_has "$(cat "$TMP/logs/5/.worker_failed_NODE1" 2>/dev/null)" "worker 1" "leaves a marker the head reads"

echo ""
echo "=== the head, with that marker present ==="
_t0=$SECONDS
out="$(_run 0 2>&1)"; rc=$?
_is "$rc" "1" "the head fails..."
_is "$([ $((SECONDS - _t0)) -lt 30 ] && echo at-once || echo waited)" "at-once" "...at once, not after its readiness timeout"
_has "$out" "worker 1" "...naming the worker"
_is "$([ -f "$TMP/logs/5/.shutdown" ] && echo raised || echo missing)" "raised" "...and raises the shutdown signal for the others"

echo ""
echo "=== a worker the head shuts down ==="
rm -rf "$TMP/logs"; mkdir -p "$TMP/logs/5"
( sleep 3; touch "$TMP/logs/5/.shutdown" ) &
out="$(_run 1 2>&1)"; rc=$?
wait
_is "$rc" "0" "exits 0"
_has "$out" "got shutdown signal" "after the head's signal"
_is "$(ls "$TMP/logs/5"/.worker_failed_NODE* 2>/dev/null | wc -l | tr -d ' ')" "0" "and leaves no failure marker"

echo ""
echo "=== the head's status is its benchmark's ==="
mkdir -p "$TMP/bench"
printf '#!/bin/bash\necho "stand-in benchmark"; exit "${BENCH_RC:-0}"\n' > "$TMP/bench/bench.sh"
rm -rf "$TMP/logs"; mkdir -p "$TMP/logs/5"
out="$(_run 0 VLLM_BEHAVIOUR=ready SHARED_DIR="$TMP/bench" BENCHMARK_SCRIPT_FILE=bench.sh BENCH_RC=7 2>&1)"; rc=$?
_is "$rc" "7" "a benchmark that fails: the head exits with its status"
_has "$out" "benchmark exited 7" "...and says so"
rm -rf "$TMP/logs"; mkdir -p "$TMP/logs/5"
out="$(_run 0 VLLM_BEHAVIOUR=ready SHARED_DIR="$TMP/bench" BENCHMARK_SCRIPT_FILE=bench.sh BENCH_RC=0 2>&1)"; rc=$?
_is "$rc" "0" "a benchmark that succeeds: the head exits 0"

echo ""
echo "=== a container barrier that fails ==="
mkdir -p "$TMP/barrier"
printf 'import sys\nprint("stand-in barrier: peer unreachable")\nsys.exit(2)\n' > "$TMP/barrier/socket_barrier.py"
rm -rf "$TMP/logs"; mkdir -p "$TMP/logs/5"
out="$(_run 1 SHARED_DIR="$TMP/barrier" 2>&1)"; rc=$?
_is "$rc" "1" "fails the node instead of starting its server"
_has "$out" "container barrier failed" "...and says so"
_is "$(grep -c 'stand-in vllm' <<<"$out")" "0" "...no server was started"

echo ""
echo "======================================================"
echo "  colocated_worker_assert: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
