#!/bin/bash
# Offline check of cluster_check_recipe (scripts/common/cluster.sh): the launchers' gate that
# resolves a recipe's declared roles (models.yaml roles:) against the allocation's GPU and the
# capability catalog (compat/rocm.yaml) before anything expensive happens.
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
YAML="$DIR/vllm_dissag/models.yaml"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0
_is()  { [ "$1" = "$2" ] && { printf "  PASS  %s\n" "$3"; pass=$((pass+1)); } || { printf "  FAIL  %s\n        got:  %s\n        want: %s\n" "$3" "$1" "$2"; fail=$((fail+1)); }; }
_has() { case "$1" in *"$2"*) printf "  PASS  %s\n" "$3"; pass=$((pass+1)) ;; *) printf "  FAIL  %s\n        missing: %s\n" "$3" "$2"; fail=$((fail+1)) ;; esac; }

# gate MODEL [ROLES] with extra env: prints the gate's output, then "rc=N" and the recorded note.
gate() {
    local model="$1" roles="${2:-}"; shift 2 2>/dev/null || shift
    env -i PATH="$PATH" HOME="$TMP" MAD_SCRIPTS_DIR="$DIR" "$@" bash -c '
        . "$0/common/config/layers.sh"; . "$0/common/cluster.sh" >/dev/null 2>&1; config_begin
        cluster_check_recipe "$1" "$2" "$3" 2>&1; echo "rc=$?"
        echo "note=${_CONFIG_NOTE[MAD_RECIPE_ROLES]-<none>}"' "$DIR" "$YAML" "$model" "$roles"
}

echo "=== resolves ==="
out="$(gate Kimi-K3-MXFP4 "" MAD_GPU_ARCH=gfx942 MAD_ENGINE_BUILD=vllm@vllm-kimi-k3)"
_has "$out" "rc=0" "the MI300X recipe on gfx942 with its build"
_has "$out" "mxfp4 weights are converted to int4_g32" "the conversion is reported"
_has "$out" $'note=recipe-check\tresolves on gfx942 with vllm@vllm-kimi-k3: prefill weights=int4_g32' "and recorded for config.resolved.json"
out="$(gate Kimi-K3 "" MAD_GPU_ARCH=gfx942)"
_has "$out" "resolves with vllm@vllm-kimi-k3" "no build named: the catalog build that resolves is named"
out="$(gate Kimi-K3 server MAD_GPU_ARCH=gfx942)"
_has "$out" "server weights=int4_g32" "the colocated launcher's server role"

out="$(gate Kimi-K3-MXFP4 "" MAD_GPU_ARCH=gfx942 MAD_PIN_SET=vllm-kimi-k3)"
_has "$out" "resolves on gfx942 with vllm@vllm-kimi-k3" "MAD_PIN_SET names the image's build (as madengine sets it)"

echo ""
echo "=== refuses ==="
out="$(gate Kimi-K3-MXFP4 "" MAD_GPU_ARCH=gfx942 MAD_PIN_SET=no-such-pins)"
_has "$out" "rc=1" "a pin set the catalog has no entry for"
_has "$out" "has no entry in the capability catalog" "...saying so"
out="$(gate Kimi-K3-MXFP4 "" MAD_GPU_ARCH=gfx950 MAD_ENGINE_BUILD=vllm@vllm-kimi-k3)"
_has "$out" "rc=1" "the MI300X recipe on gfx950"
_has "$out" "which gfx950 does not need" "naming the conversion its flags make"
_has "$out" "note=<none>" "nothing recorded"
out="$(gate Kimi-K3-MXFP4-MI355X "" MAD_GPU_ARCH=gfx942 MAD_ENGINE_BUILD=vllm@vllm-kimi-k3)"
_has "$out" "rc=1" "the MI355X recipe on gfx942"
out="$(gate Kimi-K3-MXFP4-MI355X "" MAD_GPU_ARCH=gfx950 MAD_ENGINE_BUILD=vllm@vllm-mori121)"
_has "$out" "rc=1" "an engine build the catalog does not list for the GPU"

echo ""
echo "=== does not gate ==="
_has "$(gate Kimi-K3-MXFP4 "" MAD_GPU_ARCH=gfx950 RECIPE_CHECK=0)" "rc=0" "RECIPE_CHECK=0"
_has "$(gate Kimi-K3-MXFP4 "")" "rc=0" "GPU not detected"
_has "$(gate DeepSeek-V3 "" MAD_GPU_ARCH=gfx950)" "rc=0" "a recipe that declares no roles"
_has "$(gate No-Such-Model "" MAD_GPU_ARCH=gfx942)" "rc=0" "a model with no recipe (the launcher's own check)"
out="$(gate Kimi-K3-MXFP4 "" MAD_GPU_ARCH=gfx950 MAD_CATALOG="$TMP/none.yaml")"
_has "$out" "rc=0" "no catalog"
_has "$out" "NOT being checked" "...said loudly"

echo ""
echo "======================================================"
echo "  recipe_check_assert: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
