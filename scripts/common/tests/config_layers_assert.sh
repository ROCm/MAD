#!/bin/bash
# Offline check of scripts/common/config/layers.sh: the one order of configuration
# layers, site < connector .env < recipe < topology spec < submit time, and the
# per-job record of where every value came from.
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIB="$DIR/common/config/layers.sh"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0
_is()  { [ "$1" = "$2" ] && { printf "  PASS  %s\n" "$3"; pass=$((pass+1)); } || { printf "  FAIL  %s\n        got:  %s\n        want: %s\n" "$3" "$1" "$2"; fail=$((fail+1)); }; }
_has() { case "$1" in *"$2"*) printf "  PASS  %s\n" "$3"; pass=$((pass+1)) ;; *) printf "  FAIL  %s\n        missing: %s\n" "$3" "$2"; fail=$((fail+1)) ;; esac; }

cat > "$TMP/models.yaml" <<'EOF'
M:
  env:
    SUBMITTED: recipe
    SPEC: recipe
    SITE: recipe
    SITE_SAME: same
    CONN: recipe
    CONN_SAME: same
    ONLY_RECIPE: recipe
    A_BOOL: true
    HF_TOKEN: recipe-secret
EOF
printf 'CONN=connector\nCONN_SAME=same\nCONN_ONLY=connector\n' > "$TMP/conn.env"

# A launcher in miniature: submit-time names, then each layer in the launchers' order.
run() {
    env -i PATH="$PATH" HOME="$TMP" SUBMITTED=submitted SPEC=submitted-but-spec "$@" bash -c '
        . "$0"
        config_begin
        export SPEC=spec;                         config_layer topology
        export SITE=site SITE_SAME=same SITE_ONLY=site; config_layer site
        config_apply_recipe "$1/models.yaml" M "$1/conn.env"
        for k in SUBMITTED SPEC SITE SITE_SAME CONN CONN_SAME ONLY_RECIPE A_BOOL SITE_ONLY; do
            echo "$k=${!k-<unset>}"
        done
        [ -n "${CONN_ARGS:-}" ] || CONN_ARGS="-e CONN=${CONN:-connector} -e CONN_SAME=${CONN_SAME:-same}"
        [[ -z "${CONN_ONLY:-}" ]] && config_note CONN_ONLY connector connector-env
        echo "ARGS $CONN_ARGS"
        echo "RECIPE_SUBMITTED=$(config_recipe SUBMITTED) RECIPE_NONE=[$(config_recipe NOT_IN_RECIPE)]"
        config_write "$1/out/config.resolved.json" >/dev/null
    ' "$LIB" "$TMP"
}
out="$(run)"

echo "=== which layer wins ==="
_has "$out" "SUBMITTED=submitted"   "submit time beats the recipe"
_has "$out" "SPEC=spec"             "the topology spec beats the recipe (and the submitted value it replaced)"
_has "$out" "SITE=recipe"           "the recipe beats a site default"
_has "$out" "CONN=recipe"           "the recipe beats a connector .env value (exported over it)"
_has "$out" "ARGS -e CONN=recipe -e CONN_SAME=same" "so the connector -e arguments carry the recipe's value"
_has "$out" "CONN_SAME=<unset>"     "a recipe value equal to the connector .env is not exported (no new -e)"
_has "$out" "ONLY_RECIPE=<unset>"   "a recipe name no lower layer sets is left to the container, as before"
_has "$out" "SITE_SAME=same"        "a site default equal to the recipe is unchanged"
_has "$out" "SITE_ONLY=site"        "a site default the recipe does not name is unchanged"
_has "$out" "[config] recipe SITE=recipe over site (site)" "an override is reported"
_has "$out" "RECIPE_SUBMITTED=recipe RECIPE_NONE=[]" "config_recipe: the recipe's own value, whichever layer won; empty when it sets none"
_has "$out" "[config] SUBMITTED=submitted (submit) over the recipe's recipe" "a submitted value over the recipe is reported"

echo ""
echo "=== the resolved record ==="
J="$TMP/out/config.resolved.json"
_src() { python3 -c 'import json,sys; e=json.load(open(sys.argv[1]))["env"].get(sys.argv[2]); print(e and e["source"]+":"+e["value"])' "$J" "$1"; }
_is "$(_src SUBMITTED)"   "submit:submitted"   "submitted"
_is "$(_src SPEC)"        "topology:spec"      "topology"
_is "$(_src SITE)"        "recipe:recipe"      "recipe over site"
_is "$(_src SITE_SAME)"   "recipe:same"        "a site value the recipe agrees with is credited to the recipe"
_is "$(_src SITE_ONLY)"   "site:site"          "site"
_is "$(_src ONLY_RECIPE)" "recipe:recipe"      "a name the container takes from the recipe is recorded"
_is "$(_src A_BOOL)"      "recipe:True"        "values as the container's reader converts them (str())"
_is "$(_src CONN_ONLY)"   "connector-env:connector" "a connector .env value passed only as -e is recorded"
_is "$(_src HF_TOKEN)"    "recipe:<set>"       "a secret-looking name's value is not written"
_is "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["order"])' "$J")" \
    "['container-default', 'site', 'connector-env', 'recipe', 'topology', 'workload', 'submit']" "the order is stated in the record"

echo ""
echo "=== madengine's sources (MAD_ENV_SOURCES) ==="
out="$(run MAD_ENV_SOURCES="SUBMITTED=card,SPEC=additional-context,NOT_SET=card,bad" A=1)"
_has "$out" "SUBMITTED=submitted" "a sourced submitted value still beats the recipe"
_is "$(_src SUBMITTED)" "submit:card:submitted" "the record names madengine's source"
_is "$(_src SPEC)" "topology:spec" "a later layer that changed the value still owns it"
_is "$(_src SITE)" "recipe:recipe" "word splitting of later layers is unaffected"
_is "$(_src A)" "submit:1" "a submitted name madengine did not list stays plain submit"

echo ""
echo "=== the workload file (MAD_CONFIG), read by MAD itself ==="
mkdir -p "$TMP/wl/sub"
cat > "$TMP/wl/base.yaml" <<'EOF'
version: 2
model: {env: {FROM_BASE: base, OVERRIDDEN: base, TP_SIZE: "4"}}
EOF
cat > "$TMP/wl/sub/card.yaml" <<'EOF'
version: 2
defaults: [../base.yaml]
model: {env: {OVERRIDDEN: card, TAG: "tp${model.env.TP_SIZE}", SUBMITTED: workload, BENCHMARK_SCRIPT: niah}}
benchmark:
  niah: {env: {WORDS: "50"}}
  sweep: {env: {WORDS: "sweep"}}
EOF
wl() {
    env -i PATH="$PATH" HOME="$TMP" SUBMITTED=submitted "$@" bash -c '
        . "$0"; config_begin
        config_apply_workload "$1/wl" || { echo "EXIT=1"; exit 0; }
        for k in FROM_BASE OVERRIDDEN TAG SUBMITTED WORDS; do echo "$k=${!k-<unset>}"; done
        config_write "$1/wl/out.json" >/dev/null
    ' "$LIB" "$TMP" 2>&1
}
out="$(wl MAD_CONFIG=sub/card.yaml)"
_has "$out" "FROM_BASE=base"   "a value inherited through defaults:"
_has "$out" "OVERRIDDEN=card"  "the file beats its base"
_has "$out" "TAG=tp4"          "interpolation resolved after the merge"
_has "$out" "SUBMITTED=submitted" "a submitted value beats the workload file"
_has "$out" "WORDS=50"         "the file's own benchmark kind selected"
_is "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["env"]["FROM_BASE"]["source"])' "$TMP/wl/out.json")" \
    "workload" "the record names the workload layer"
_has "$(wl MAD_CONFIG=sub/card.yaml BENCHMARK_SCRIPT=sweep)" "WORDS=sweep" "the run's benchmark kind selects its entries"
_has "$(wl)" "FROM_BASE=<unset>" "no MAD_CONFIG and no mad-config.yaml: nothing applied"
printf 'M:\n  env: {OVERRIDDEN: recipe, ONLY_RECIPE: recipe}\n' > "$TMP/wl/models.yaml"
_rw="$(env -i PATH="$PATH" HOME="$TMP" MAD_CONFIG=sub/card.yaml bash -c '
    . "$0"; config_begin; config_apply_workload "$1/wl"; config_apply_recipe "$1/wl/models.yaml" M
    echo "OVERRIDDEN=$OVERRIDDEN"' "$LIB" "$TMP" 2>&1)"
_has "$_rw" "OVERRIDDEN=card" "a workload value beats the recipe, as it does when madengine submits it"
cp "$TMP/wl/sub/card.yaml" "$TMP/wl/mad-config.yaml"; sed -i 's#\.\./base#base#' "$TMP/wl/mad-config.yaml"
_has "$(wl)" "OVERRIDDEN=card" "a mad-config.yaml beside the launcher is read, as madengine does"
rm -f "$TMP/wl/mad-config.yaml"
_has "$(wl MAD_CONFIG=missing.yaml)" "EXIT=1" "a MAD_CONFIG naming no file stops the run"
printf 'version: 2\ndefaults: [nowhere.yaml]\n' > "$TMP/wl/bad.yaml"
_has "$(wl MAD_CONFIG=bad.yaml)" "EXIT=1" "an unusable workload file stops the run"

echo ""
echo "======================================================"
echo "  config_layers_assert: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
