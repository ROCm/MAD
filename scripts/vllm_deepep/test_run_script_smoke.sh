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
# The stub exits immediately, so the script takes its "server died during startup" path. That is
# deliberate: it reaches the launch, the readiness probe and the exit trap in a couple of seconds,
# which is every line this change touches.
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

echo
echo "passed ${_pass}, failed ${_fail}"
(( _fail == 0 ))
