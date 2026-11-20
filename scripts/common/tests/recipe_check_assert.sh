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

# [GATE_YAML=recipes] gate MODEL [ROLES] with extra env (GATE_ENGINE, GATE_FLAGS): prints the
# gate's output, then "rc=N" and the recorded note.
cat > "$TMP/gate.sh" <<'GATE'
. "$1/common/config/layers.sh"; . "$1/common/cluster.sh" >/dev/null 2>&1; config_begin
if [ -n "${GATE_FLAGS+x}" ]; then cluster_check_recipe "$2" "$3" "${GATE_ENGINE:-vllm}" "$4" "$GATE_FLAGS" 2>&1
else cluster_check_recipe "$2" "$3" "${GATE_ENGINE:-vllm}" "$4" 2>&1; fi
echo "rc=$?"
echo "note=${_CONFIG_NOTE[MAD_RECIPE_ROLES]-<none>}"
GATE
gate() {
    local model="$1" roles="${2:-}"; shift 2 2>/dev/null || shift
    env -i PATH="$PATH" HOME="$TMP" MAD_SCRIPTS_DIR="$DIR" "$@" \
        bash "$TMP/gate.sh" "$DIR" "${GATE_YAML:-$YAML}" "$model" "$roles"
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
# An image built from a named spec: its vLLM build states its capabilities in the matrix (a fixture
# catalog: the real one, plus a spec "stack-x" selecting such a build).
mkdir -p "$TMP/cat/images"
python3 - "$DIR/common" "$TMP/cat" <<'PY'
import copy, sys, yaml
sys.path.insert(0, sys.argv[1])
import catalog
c = catalog.load(catalog.COMPAT)
c.pop(catalog.IMAGES_KEY, None)
c["components"].setdefault("vllm", {})["1.0+k3@x"] = {"source": {"repo": "r", "commit": "c"},
                                                     "capabilities": copy.deepcopy(c["kernels"]["vllm@vllm-kimi-k3"])}
yaml.safe_dump(c, open(sys.argv[2] + "/catalog.yaml", "w"))
yaml.safe_dump({"components": {"vllm": "1.0+k3@x"}}, open(sys.argv[2] + "/images/stack-x.yaml", "w"))
PY
out="$(gate Kimi-K3-MXFP4 "" MAD_GPU_ARCH=gfx942 MAD_CATALOG="$TMP/cat/catalog.yaml" MAD_IMAGE_COMPONENTS=mori=x,vllm=1.0+k3@x)"
_has "$out" "resolves on gfx942 with vllm@stack-x" "MAD_IMAGE_COMPONENTS (an image built from a spec) names its engine build"

echo ""
echo "=== refuses ==="
out="$(gate Kimi-K3-MXFP4 "" MAD_GPU_ARCH=gfx942 MAD_PIN_SET=no-such-pins)"
_has "$out" "rc=1" "a pin set the catalog has no entry for"
_has "$out" "has no entry in the capability catalog" "...saying so"
out="$(gate Kimi-K3-MXFP4-MI355X "" MAD_GPU_ARCH=gfx942 MAD_ENGINE_BUILD=vllm@vllm-kimi-k3)"
_has "$out" "rc=1" "a recipe pinned to gfx950 (alias), on gfx942"
_has "$out" "written for gfx950 (GPU_ARCHS), not gfx942" "naming its declared accelerators"
_has "$out" "note=<none>" "nothing recorded"
out="$(gate Kimi-K3-MXFP4 "" MAD_GPU_ARCH=gfx950 MAD_ENGINE_BUILD=vllm@vllm-kimi-k3)"
_has "$out" "resolves on gfx950" "a recipe with a gfx950 variant runs it there"
_has "$out" "weights=mxfp4" "...natively (no int4 conversion)"
out="$(gate Kimi-K3-MXFP4-MI355X "" MAD_GPU_ARCH=gfx942 MAD_ENGINE_BUILD=vllm@vllm-kimi-k3)"
_has "$out" "rc=1" "the MI355X recipe on gfx942"
out="$(gate Kimi-K3-MXFP4-MI355X "" MAD_GPU_ARCH=gfx950 MAD_ENGINE_BUILD=vllm@vllm-mori121)"
_has "$out" "rc=1" "an engine build the catalog does not list for the GPU"

INT4="--max-num-seqs 8 --quantization-config '{\"moe\":{\"weight\":\"int4_per_group_32\"}}'"
_has "$(gate Kimi-K3 server MAD_GPU_ARCH=gfx950 GATE_FLAGS="$INT4")" "rc=1" "colocated: the card's flags are compared (int4 conversion on gfx950)"
_has "$(gate Kimi-K3 server MAD_GPU_ARCH=gfx942 GATE_FLAGS="$INT4")" "rc=0" "...and agree on gfx942"
SG="$DIR/sglang_disagg/models.yaml"
_has "$(GATE_YAML="$SG" gate DeepSeek-V3 "" MAD_GPU_ARCH=gfx950 GATE_ENGINE=sglang)" "resolves with sglang@sglang-disagg" "the launcher's engine picks the catalog build"
_has "$(GATE_YAML="$SG" gate DeepSeek-V3 "" MAD_GPU_ARCH=gfx942 GATE_ENGINE=sglang MAD_PIN_SET=vllm-kimi-k3)" "is not a sglang build" "a pin set of another engine is refused"

echo ""
echo "=== does not gate ==="
_has "$(gate Kimi-K3-MXFP4 "" MAD_GPU_ARCH=gfx950 RECIPE_CHECK=0)" "rc=0" "RECIPE_CHECK=0"
_has "$(gate Kimi-K3-MXFP4 "")" "rc=0" "GPU not detected"
_has "$(gate gpt-oss-120b "" MAD_GPU_ARCH=gfx950)" "rc=0" "a recipe that declares no roles"
_has "$(gate No-Such-Model "" MAD_GPU_ARCH=gfx942)" "rc=0" "a model with no recipe (the launcher's own check)"
out="$(gate Kimi-K3-MXFP4 "" MAD_GPU_ARCH=gfx950 MAD_CATALOG="$TMP/none.yaml")"
_has "$out" "rc=0" "no catalog"
_has "$out" "NOT being checked" "...said loudly"

echo ""
echo "=== GPU memory for recipes that size memory from it ==="
KFD="$TMP/kfd"; mkdir -p "$KFD/0" "$KFD/1/mem_banks/0" "$KFD/1/mem_banks/1"
printf 'gfx_target_version 0\n' > "$KFD/0/properties"                       # the CPU node
printf 'gfx_target_version 90402\n' > "$KFD/1/properties"
printf 'heap_type 1\nsize_in_bytes 206141652992\n' > "$KFD/1/mem_banks/0/properties"
printf 'heap_type 2\nsize_in_bytes 4096\n' > "$KFD/1/mem_banks/1/properties"
cat > "$TMP/mem.yaml" <<'YAML'
Rel: {memory: {KV: {fraction: 0.1}}}
Abs: {env: {KV: "1"}, memory: {KV: {fraction: 0.1}}}
YAML
cat > "$TMP/mem.sh" <<'MEM'
. "$1/common/cluster.sh" >/dev/null 2>&1
cluster_gpu_memory_for_recipe "$2" "$3" 2>&1; echo "rc=$? bytes=${MAD_GPU_MEMORY_BYTES:-}"
MEM
mem() { env -i PATH="$PATH" HOME="$TMP" MAD_SCRIPTS_DIR="$DIR" "${@:2}" bash "$TMP/mem.sh" "$DIR" "$TMP/mem.yaml" "$1"; }
_has "$(mem Rel CLUSTER_SYSFS_KFD="$KFD")" "rc=0 bytes=206141652992" "probed: the GPU's largest memory bank"
_has "$(mem Rel MAD_GPU_MEMORY_BYTES=5)" "rc=0 bytes=5" "set explicitly: not probed"
_has "$(mem Abs CLUSTER_SYSFS_KFD="$TMP/none")" "rc=0 bytes=" "an explicit value in the recipe: nothing to probe"
_has "$(mem Rel CLUSTER_SYSFS_KFD="$TMP/none")" "rc=1" "unreadable GPU memory is refused"

echo ""
echo "======================================================"
echo "  recipe_check_assert: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
