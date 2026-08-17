#!/bin/bash
# preflight_nodes.sh <node> [node...]
# Gate a scored run. Two INDEPENDENT checks that catch different failures:
#   STATIC  (Finding 19b) - our own config: util*capacity + MoRI heap must leave >=20 GiB
#   DYNAMIC (Finding 21b) - somebody else's job: node must be actually idle
# Exit 0 = safe to launch. Non-zero = do not launch.
#
# Sampling discipline (learned by getting it wrong three times in one day):
#  - VRAM is sampled N times with a settle delay; a single sample catches teardown
#    transients and manufactures phantom co-tenants. We take the MINIMUM.
#  - Ownership requires a KFD PID, not just a high VRAM number or a running container.
#    "Container is up" is NOT evidence it holds GPU memory (mori_zqz was `sleep infinity`).
#  - pgrep patterns are anchored so the check cannot match its own command line.
set -u
CAP_BYTES=206141652992           # 191.98 GiB per MI308X
CAP_GIB=191.98
SAMPLES=${SAMPLES:-3}
SETTLE=${SETTLE:-5}
VRAM_PCT_MAX=${VRAM_PCT_MAX:-5}  # clean node reads 0.14%
HEADROOM_MIN=${HEADROOM_MIN:-20}
rc=0

echo "== STATIC budget check =="
UTIL=${GPU_MEMORY_UTILIZATION:-0.80}
HEAP_B=${MORI_SHMEM_HEAP_SIZE:-34359738368}
awk -v u="$UTIL" -v h="$HEAP_B" -v c="$CAP_GIB" -v m="$HEADROOM_MIN" 'BEGIN{
  hg=h/1073741824; pool=u*c; tot=pool+hg; head=c-tot;
  printf "  util %.4f -> pool %.2f + heap %.2f = %.2f of %.2f GiB, headroom %.2f GiB -> %s\n",
    u,pool,hg,tot,c,head,(head<m?"FAIL: will OOM in warmup":"ok");
  exit (head<m)}' || rc=1

echo "== DYNAMIC idle check (${SAMPLES} samples, ${SETTLE}s apart; reporting MIN) =="
for N in "$@"; do
  # min-of-N max-per-GPU used bytes, so a teardown transient cannot trip the gate
  MINPCT=$(ssh -n "$N" "for i in \$(seq 1 $SAMPLES); do
      rocm-smi --showmeminfo vram 2>/dev/null | grep -i 'used memory' \
        | awk '{print \$NF}' | sort -rn | head -1
      [ \$i -lt $SAMPLES ] && sleep $SETTLE
    done | sort -n | head -1" 2>/dev/null)
  # ownership: a real foreign job has a KFD PID. Absence of PIDs => memory is not held.
  KFD=$(ssh -n "$N" "rocm-smi --showpids 2>/dev/null | grep -cE '^[0-9]+' " 2>/dev/null)
  KFD=${KFD:-0}
  # anchored so it cannot self-match the wrapping bash -c
  BENCH=$(ssh -n "$N" "pgrep -af '[v]llm bench serve' | wc -l" 2>/dev/null)
  BENCH=${BENCH:-0}
  awk -v n="$N" -v b="${MINPCT:-0}" -v c="$CAP_BYTES" -v k="$KFD" -v s="$BENCH" -v lim="$VRAM_PCT_MAX" 'BEGIN{
    p=(c>0)?b/c*100:0; bad=0;
    if (p>lim && k>0) bad=1;                      # high VRAM AND an owning pid = real tenant
    if (s>0) bad=1;                               # stray load generator
    printf "  %-11s minVRAM %6.2f%%  kfd_pids=%s  bench=%s -> %s\n", n,p,k,s,
      (bad?"ABORT":(p>lim?"WARN high VRAM but no KFD pid (settling?) - recheck":"OK"));
    exit bad}' || rc=1
done
[ $rc -eq 0 ] && echo "PREFLIGHT PASS" || echo "PREFLIGHT FAIL"
exit $rc
