#!/bin/bash
# Dry-run parity gate for the unified two-axis launcher.
# =============================================================================
# Drives the REAL vllm_disagg.sh (DRY_RUN=1) for every role cell and diffs the
# assembled `vllm serve` argv against COMMITTED GOLDEN fixtures (tests/golden/),
# which were captured from the legacy standalone launchers before they were
# deleted. No GPUs required. Exits non-zero on any diff.
#
# Golden cells (the 3 connector/mode combos with a legacy equivalent):
#   rixl   + WIDE_EP=0          == vllm_disagg_server.sh        (prefill, decode)
#   moriio + WIDE_EP=1 (mori)   == vllm_disagg_mori_ep.sh       (4 role cells)
#   rixl   + WIDE_EP=1 (deepep) == vllm_disagg_server_deepep.sh (4 role cells)
#
# moriio + WIDE_EP=0 (new cell) has no legacy → smoke-checked (well-formed argv).
#
# NOTE: the mori_* goldens INTENTIONALLY diverge from MAD-develop's legacy mori
# launcher in ONE field: --all2all-backend. Legacy emitted the bare "mori" alias,
# which the v1.2.0 MoRI-EP image REJECTS; the connector now emits per-role
# mori_high_throughput / mori_low_latency (PR#324 behavior). Goldens were refreshed
# to the image-correct values; everything else stays byte-identical.
#
# To regenerate goldens (only if a legacy-equivalent change is intentional):
#   see tests/golden/README or git history for gen_golden.sh.
# =============================================================================
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
G="$DIR/tests/golden"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/out" /run_logs/PARITY 2>/dev/null || true

export MODEL_PATH=/models/DeepSeek-V3 MODEL_NAME=DeepSeek-V3
export NIXL_COOKBOOK_PATH="$DIR" SLURM_JOB_ID=PARITY GPUS_PER_NODE=8
export MASTER_ADDR=10.0.0.1 IPADDRS=10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4
export xP=2 yD=2
fail=0
norm_hostip() { sed -E 's/"kv_ip": "[^"]*"/"kv_ip": "HOSTIP"/'; }

# new_cell connector wide_ep ep_backend node_rank [model model_path] -> stdout argv
new_cell() {
    local _m="${5:-$MODEL_NAME}" _mp="${6:-$MODEL_PATH}"
    ( export CONNECTOR="$1" WIDE_EP="$2" EP_BACKEND="$3" NODE_RANK="$4" \
             MODEL_NAME="$_m" MODEL_PATH="$_mp" DRY_RUN=1
      bash "$DIR/vllm_disagg.sh" 2>/dev/null ) \
      | awk '/^===DRYRUN/{f=1;next} /^===END===/{f=0} f' | sed '1{/^vllm$/d}' | norm_hostip
}

cmp_golden() {  # label connector wide_ep ep_backend node_rank golden_file [model model_path]
    local label="$1"; shift
    new_cell "$1" "$2" "$3" "$4" "${6:-}" "${7:-}" > "$TMP/out/n.txt"
    if diff -u "$G/$5" "$TMP/out/n.txt" >/dev/null; then echo "  OK   $label";
    else echo "  DIFF $label"; diff -u "$G/$5" "$TMP/out/n.txt"; fail=1; fi
}

# rixl+TP uses a DENSE model (DeepSeek is wideEP-only now). Llama-70B is the
# representative dense TP model.
_L=amd-Llama-3.3-70B-Instruct-FP8-KV; _LP=/models/Llama-3.3-70B
echo "== rixl + WIDE_EP=0 dense (golden: vllm_disagg_server.sh) =="
cmp_golden "rixl prefill" rixl 0 deepep 0 rixl_prefill.txt "$_L" "$_LP"
cmp_golden "rixl decode"  rixl 0 deepep 2 rixl_decode.txt  "$_L" "$_LP"

echo "== moriio + WIDE_EP=1 mori (golden: vllm_disagg_mori_ep.sh) =="
cmp_golden "moriio prefill master" moriio 1 mori 0 mori_prefill_master.txt
cmp_golden "moriio prefill child"  moriio 1 mori 1 mori_prefill_child.txt
cmp_golden "moriio decode master"  moriio 1 mori 2 mori_decode_master.txt
cmp_golden "moriio decode child"   moriio 1 mori 3 mori_decode_child.txt

echo "== rixl + WIDE_EP=1 deepep (golden: vllm_disagg_server_deepep.sh) =="
cmp_golden "deepep prefill master" rixl 1 deepep 0 deepep_prefill_master.txt
cmp_golden "deepep prefill child"  rixl 1 deepep 1 deepep_prefill_child.txt
cmp_golden "deepep decode master"  rixl 1 deepep 2 deepep_decode_master.txt
cmp_golden "deepep decode child"   rixl 1 deepep 3 deepep_decode_child.txt

echo "== moriio + WIDE_EP=0 (NEW cell — smoke only, no legacy) =="
new_cell moriio 0 "" 0 > "$TMP/out/miotp.txt"
if grep -q -- '--tensor-parallel-size' "$TMP/out/miotp.txt" \
   && grep -q 'MoRIIOConnector' "$TMP/out/miotp.txt" \
   && ! grep -q -- '--enable-expert-parallel' "$TMP/out/miotp.txt"; then
    echo "  OK   moriio+TP smoke (--tensor-parallel-size + MoRIIOConnector, no --enable-expert-parallel)"
else
    echo "  FAIL moriio+TP smoke"; cat "$TMP/out/miotp.txt"; fail=1
fi

echo "== validation: invalid cross-pairings must abort =="
chk_reject() {  # label connector wide_ep ep_backend
    if ( export CONNECTOR="$2" WIDE_EP="$3" EP_BACKEND="$4" NODE_RANK=0 DRY_RUN=1
         bash "$DIR/vllm_disagg.sh" ) >/dev/null 2>&1; then
        echo "  FAIL $1 (should have aborted)"; fail=1
    else echo "  OK   $1 (rejected)"; fi
}
chk_reject "moriio+deepep rejected" moriio 1 deepep
chk_reject "rixl+mori rejected"     rixl   1 mori

echo "================================"
[ "$fail" -eq 0 ] && echo "ALL PARITY CELLS BYTE-IDENTICAL ✅" || echo "PARITY FAILED ❌"
exit $fail
