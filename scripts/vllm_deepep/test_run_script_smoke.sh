#!/bin/bash
# Runs run_vllm_deepep.sh as a script, with a stub in place of vLLM.
#
# The other suite (../sglang_disagg/test_shutdown_and_barrier.sh) extracts functions and tests
# them in isolation. That is not enough here: both defects this file exists for were in top-level
# code that no function test reaches, and both would have failed every run of this workload --
# including runs with profiling off.
#
#   1. The sampler path was resolved with `$(cd ../sglang_disagg && pwd)`. madengine copies only a
#      model's own script directory into the container, so that directory is not there, `cd`
#      returns non-zero, and under `set -euo pipefail` the whole workload died on that line.
#   2. Artifacts defaulted under /run_logs, which this workload -- unlike sglang_disagg -- does
#      not mount. They would have gone to the container's writable layer and vanished with it.
#
# Two stubs. The first exits immediately, so the script takes its "server died during startup"
# path -- the launch, the readiness probe and the exit trap in a couple of seconds. The second
# answers the readiness probe and prints a parseable result block, so the benchmark loop, the CSV
# and the profiled pass are reached as well; without it, "PROFILE_ENABLE=1 captures nothing"
# could survive review, which is exactly what happened.
#
# What the live stub can and cannot establish, stated so the result is not over-read: it shows
# the script doing its half of the trace channel -- configuring the server and asking the
# benchmark to capture. It cannot show vLLM writing a trace. That needs the real wheel on real
# GPUs and is still outstanding.
#
# No GPUs and no model needed. RDMA counters are collected for real where /sys/class/infiniband
# exists and must degrade cleanly where it does not; both are asserted.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${SCRIPT_DIR:-${HERE}}"

_pass=0
_fail=0
ok()  { echo "  ok   - $1"; _pass=$(( _pass + 1 )); }
bad() { echo "  FAIL - $1" >&2; _fail=$(( _fail + 1 )); }

T="$(mktemp -d)"
trap 'rm -rf "${T}"' EXIT
mkdir -p "${T}/run_directory" "${T}/model"

# Exactly what madengine does: the model's own script directory, dereferenced, into the run
# directory. Nothing else comes along -- no sibling workload, no repository root.
cp -LR --preserve=all "${SCRIPT_DIR}/." "${T}/run_directory/" 2>/dev/null

echo "== the shared sampler survives madengine's per-model copy =="
if [[ -f "${T}/run_directory/rdma_counters.sh" && ! -L "${T}/run_directory/rdma_counters.sh" ]]; then
    ok "rdma_counters.sh arrived as a real file"
else
    bad "rdma_counters.sh missing from the copied directory (symlink not dereferenced?)"
fi

printf '#!/bin/bash\nsleep 2\nexit 7\n' > "${T}/stub_vllm"
chmod +x "${T}/stub_vllm"

# A stub that behaves enough like vLLM to drive the whole script: it answers the readiness probe,
# records the argv it was given, and prints a result block the parser accepts. The dying stub
# above only ever reaches the launch, so the benchmark loop, the CSV and the profiled pass were
# never covered -- which is how "PROFILE_ENABLE=1 captures nothing" survived review.
cat > "${T}/stub_vllm_live" <<'STUB'
#!/bin/bash
ARGV_DIR="${STUB_ARGV_DIR:?}"
case "$1" in
  serve)
    if [[ "${2:-}" == "--help" ]]; then
      echo "usage: vllm serve [options]"
      # Whether this wheel advertises the flag is the thing under test.
      [[ "${STUB_PROFILER_CONFIG:-0}" == "1" ]] && echo "  --profiler-config PROFILER_CONFIG"
      exit 0
    fi
    printf '%s\n' "$@" > "${ARGV_DIR}/serve.argv"
    port=18000
    for ((i = 1; i <= $#; i++)); do
      [[ "${!i}" == "--port" ]] && { j=$((i + 1)); port="${!j}"; }
    done
    # Always 200, so the script's `curl -sf .../health` succeeds.
    exec python3 -c "
import http.server, sys
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self): self.send_response(200); self.end_headers(); self.wfile.write(b'ok')
    def do_POST(self): self.do_GET()
    def log_message(self, *a): pass
http.server.HTTPServer(('127.0.0.1', int(sys.argv[1])), H).serve_forever()
" "$port"
    ;;
  bench)
    printf '%s\n' "$@" >> "${ARGV_DIR}/bench.argv"
    # Failures only on the profiled pass. Reporting them on a warm-up would abort the run before
    # the profiled pass exists -- correctly, which is why that path cannot be used to test this.
    _failed=0
    for _a in "$@"; do [[ "$_a" == "--profile" ]] && _failed="${STUB_FAILED:-0}"; done
    echo "============ Serving Benchmark Result ============"
    echo "Failed requests:                         ${_failed}"
    echo "Total token throughput (tok/s):          1234.56"
    echo "Median TPOT (ms):                        34.50"
    echo "P99 TTFT (ms):                           456.70"
    exit 0
    ;;
esac
exit 2
STUB
chmod +x "${T}/stub_vllm_live"

run_live() {
    rm -rf "${T}/argv"; mkdir -p "${T}/argv"
    ( cd "${T}/run_directory" && \
      PROFILE_ENABLE=1 RDMA_COUNTERS=0 PROFILE_DIR="${T}/profile_live" \
      STUB_ARGV_DIR="${T}/argv" STUB_PROFILER_CONFIG="$1" STUB_FAILED="$2" \
      WARMUP_RUNS=1 MEASURED_RUNS=1 \
      MODEL_PATH="${T}/model" VLLM="${T}/stub_vllm_live" PYTHON=python3 PORT="$3" \
      timeout 200 bash ./run_vllm_deepep.sh 2>&1 )
}

run_script() {
    ( cd "${T}/run_directory" && \
      PROFILE_ENABLE="$1" RDMA_COUNTERS="$2" PROFILE_DIR="${T}/profile" \
      RDMA_COUNTERS_INTERVAL=1 RDMA_COUNTERS_START_TIMEOUT_S=15 \
      MODEL_PATH="${T}/model" VLLM="${T}/stub_vllm" PORT="$3" \
      timeout 200 bash ./run_vllm_deepep.sh 2>&1 )
}

echo "== it reaches the server launch with profiling off =="
out="$(run_script 0 0 18094)"
# "server died during startup" is the stub being noticed. Anything earlier means the script
# aborted in its own prologue, which is defect 1.
case "$out" in
    *"server died during startup"*) ok "prologue completed and the server was launched" ;;
    *) bad "never reached the launch: $(printf '%s' "$out" | tail -3 | tr '\n' ' ')" ;;
esac

echo "== each invocation gets its own directory under PROFILE_DIR =="
out="$(run_script 1 1 18093)"
# compgen, not `[[ -d ... ]]`: bash does not expand a glob in the operand of -d, so that test
# asks about a directory literally named with an asterisk and always fails.
if compgen -G "${T}/profile/*/rdma" > /dev/null 2>&1; then
    ok "profile directory created under PROFILE_DIR"
else
    bad "PROFILE_DIR was not honoured"
fi

# A persistent mount is reused by every run of every model card. Sharing one directory left old
# RCCL logs and traces in place while the counter CSV was overwritten, so the artifacts described
# more than one run under a single label -- and an old trace could answer "were traces captured"
# for a run that captured none.
out2="$(run_script 1 1 18092)"
_runs="$(find "${T}/profile" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)"
if (( _runs >= 2 )); then
    ok "a second invocation did not reuse the first one's directory (${_runs} present)"
else
    bad "two invocations shared one directory (${_runs} present)"
fi

echo "== a destination that will not survive the run is called out =="
# The staged model directory is deleted after the run unless --keep-model-dir is given, and an
# unbound path lives only in the container's writable layer. Either way the run would otherwise
# report artifacts it cannot produce. ${T} here is on a real filesystem but not a mount point,
# which is the same shape as an unbound path in the container.
case "$out" in
    *"is not a mount point in this container"*) ok "warned that the destination is not persistent" ;;
    *) bad "collected into a non-persistent path without saying so" ;;
esac

echo "== the counter channel reports what actually happened =="
csv="$(ls "${T}/profile"/*/rdma/*.csv 2>/dev/null | head -1)"
if [[ -d /sys/class/infiniband ]] && compgen -G "/sys/class/infiniband/*" > /dev/null 2>&1; then
    rows="$(wc -l < "${csv}" 2>/dev/null || echo 0)"
    devices="$(awk -F, 'NR>1{print $3}' "${csv}" 2>/dev/null | sort -u | wc -l)"
    if (( rows > 1 )) && (( devices > 0 )); then
        ok "sampled ${rows} rows across ${devices} adapter(s)"
    else
        bad "adapters present but the sampler produced ${rows} row(s)"
    fi
else
    # A sampler that exits having found nothing must not be reported as collecting counters: the
    # file-first version of this check printed "RDMA counters: <path>" for a sampler already gone.
    case "$out" in
        *"found nothing to sample"*) ok "no adapters here, and the log says so" ;;
        *"RDMA counters:"*) bad "claimed counters were being collected with no adapters present" ;;
        *) bad "no adapters, and no statement either way" ;;
    esac
fi

echo "== the trace channel: the script does its half =="
# What a stub can establish and what it cannot. It can show the script configures the server and
# asks the benchmark to capture -- the two things that were missing when PROFILE_ENABLE=1 produced
# an empty directory. It cannot show vLLM writing a trace; that needs the real wheel on real GPUs.
out="$(run_live 1 0 18091)"
if grep -q -- "--profiler-config" "${T}/argv/serve.argv" 2>/dev/null; then
    ok "the server was configured through --profiler-config when the wheel advertises it"
else
    bad "wheel advertises --profiler-config and the server was not given it"
fi
if grep -q -- "--profile" "${T}/argv/bench.argv" 2>/dev/null; then
    ok "the benchmark was asked to capture"
else
    bad "no --profile reached the benchmark, so nothing would start a capture"
fi
# Exactly once: the measured runs must stay comparable, so only the extra pass profiles.
_n="$(grep -c -- "^--profile$" "${T}/argv/bench.argv" 2>/dev/null || echo 0)"
if [[ "${_n}" == "1" ]]; then
    ok "exactly one pass profiled, so the recorded runs stay unprofiled"
else
    bad "${_n} passes profiled; the recorded runs must not be"
fi

echo "== a wheel without --profiler-config is not handed the flag =="
# Passing it to a wheel that does not know it fails the launch outright, which would turn a
# profiling request into a workload that cannot start at all.
out="$(run_live 0 0 18090)"
if grep -q -- "--profiler-config" "${T}/argv/serve.argv" 2>/dev/null; then
    bad "handed --profiler-config to a wheel that does not advertise it"
else
    ok "flag withheld from a wheel that does not advertise it"
fi

echo "== failed requests during the profiled pass are reported =="
# `vllm bench serve` exits 0 while reporting failures, so a trace directory fills and passes the
# capture check while holding a capture of a workload that partly did not happen.
out="$(run_live 1 7 18089)"
case "$out" in
    *"profiled run reported 7 failed request"*) ok "the profiled pass named its failed requests" ;;
    *) bad "failed requests during the profiled pass went unreported" ;;
esac

echo
echo "passed ${_pass}, failed ${_fail}"
(( _fail == 0 ))
