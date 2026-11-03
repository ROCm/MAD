# The configuration layers of a launcher run, in one order for every launcher:
#
#     site (cluster.sh)  <  connector .env  <  recipe (models.yaml env:)
#                        <  topology spec (MAD_TOPOLOGY)  <  submit time
#
# "Submit time" is everything already in the environment when the launcher starts:
# the card's env_vars, madengine's layered config and --additional-context, or your
# own `export` before sbatch. Those names are recorded first (config_begin), so the
# later layers can tell a value someone chose from a default some layer filled in.
# madengine passes where each of them came from as MAD_ENV_SOURCES (NAME=label,...);
# the record then says e.g. "submit:card" or "submit:additional-context".
#
# Before this, the recipe was applied only inside the container and only to names
# that were not set there yet. Every name the launcher forwards with `docker run -e`
# was set by then -- by cluster.sh or a connector .env when nobody else had -- so a
# recipe value for such a name never took effect. config_apply_recipe applies the
# recipe on the batch node instead, over site and connector defaults and under
# anything submitted or stated by the topology spec.
#
# Every name the run's environment holds, with the layer that set it, is written
# per job (config_write) for reproducibility; values of secret-looking names are
# replaced by "<set>".
#
# Usage, in a launcher:
#     . "${MAD_SCRIPTS_DIR}/common/config/layers.sh"
#     config_begin                              # first, before anything sets a name
#     . .../topology/apply.sh L; config_layer topology
#     . .../cluster.sh;          config_layer site
#     config_apply_recipe "$MODELS_YAML" "$MODEL_NAME"   # before the connector .env is read
#     ...
#     config_write "$LOG_PATH/$SLURM_JOB_ID/config.resolved.json"

# Names set when the launcher started, and the layer that last set each name.
config_begin() {
    declare -gA _CONFIG_SRC=() _CONFIG_VAL=() _CONFIG_NOTE=()
    local n p
    for n in $(compgen -e); do _CONFIG_SRC[$n]=submit; done
    # madengine says where each submitted value came from (card, additional-context,
    # mad-config, docker-env-vars, distributed, madengine): MAD_ENV_SOURCES=NAME=label,...
    local -a parts=()
    IFS=, read -ra parts <<<"${MAD_ENV_SOURCES:-}"
    for p in "${parts[@]}"; do
        n="${p%%=*}"
        [[ -n "$n" && "$p" == *=* && -v "_CONFIG_SRC[$n]" ]] && _CONFIG_SRC[$n]="submit:${p#*=}"
    done
    _config_snapshot
}

_config_snapshot() {
    local n
    _CONFIG_VAL=()
    for n in $(compgen -e); do _CONFIG_VAL[$n]="${!n}"; done
}

# config_layer LABEL: every exported name that appeared or changed since the last
# call was set by LABEL.
config_layer() {
    local n
    for n in $(compgen -e); do
        if [[ ! -v "_CONFIG_VAL[$n]" || "${_CONFIG_VAL[$n]}" != "${!n}" ]]; then
            _CONFIG_SRC[$n]="$1"
        fi
    done
    _config_snapshot
}

# config_source NAME: the layer that set NAME (empty when no layer did).
config_source() { [[ -v "_CONFIG_SRC[$1]" ]] && printf '%s\n' "${_CONFIG_SRC[$1]}"; return 0; }

# config_apply_recipe MODELS_YAML MODEL_NAME [DEFAULTS_FILE...]: the recipe's env: block
# over every lower layer, under what was submitted or set by the topology spec.
# Where a lower layer set the name here (site, launcher) or a DEFAULTS_FILE (a connector
# .env) would set it, to another value, the recipe's value is exported over it. Every
# other recipe name is left to the container, which applies the recipe to the names it
# was not given -- the same value -- so `docker run` changes only where the recipe now
# takes effect over a default.
# Values are converted as the in-container reader converts them (str()).
config_apply_recipe() {
    local yaml="$1" model="$2" line k v f
    local -A defaulted=()
    shift 2
    [[ -n "$model" && -f "$yaml" ]] || return 0
    for f in "$@"; do
        [[ -f "$f" ]] || continue
        while IFS= read -r line; do
            [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] && defaulted[${line%%=*}]="${line#*=}"
        done < "$f"
    done
    config_layer launcher
    while IFS= read -r -d '' line; do
        k="${line%%=*}"; v="${line#*=}"
        case "$(config_source "$k")" in
            submit|submit:*|topology)
                [[ "${!k}" != "$v" ]] && echo "[config] ${k}=${!k} ($(config_source "$k")) over the recipe's ${v}"
                continue ;;
        esac
        if [[ -v "$k" && "${!k}" != "$v" ]]; then
            echo "[config] recipe ${k}=${v} over ${!k} ($(config_source "$k"))"
        elif [[ ! -v "$k" && -v "defaulted[$k]" && "${defaulted[$k]}" != "$v" ]]; then
            echo "[config] recipe ${k}=${v} over ${defaulted[$k]} (connector .env)"
        else
            if [[ -v "$k" ]]; then _CONFIG_SRC[$k]=recipe          # the recipe agrees
            else config_note "$k" "$v" recipe; fi                 # the container sets it from the recipe
            continue
        fi
        export "$k=$v"
        _CONFIG_SRC[$k]=recipe
    done < <(MODELS_YAML="$yaml" MODEL_NAME="$model" python3 -c '
import os, sys, yaml
m = yaml.safe_load(open(os.environ["MODELS_YAML"])) or {}
for k, v in ((m.get(os.environ["MODEL_NAME"]) or {}).get("env") or {}).items():
    sys.stdout.write(f"{k}={v}\0")
')
    _config_snapshot
}

# config_note NAME VALUE LABEL: a value the launcher passes to the containers without
# exporting it here (e.g. `-e NAME=VALUE` built from a connector .env), for config_write.
config_note() { [[ -v "_CONFIG_NOTE[$1]" ]] || _CONFIG_NOTE[$1]="$3"$'\t'"$2"; }   # the first (higher) layer to note a name keeps it

# config_write FILE: the run's environment as {name: {"value", "source"}}, JSON.
config_write() {
    local f="$1" n
    config_layer launcher
    mkdir -p "$(dirname "$f")" || return 0
    {
        for n in "${!_CONFIG_SRC[@]}"; do printf 'E\t%s\t%s\0' "$n" "${_CONFIG_SRC[$n]}"; done
        for n in "${!_CONFIG_NOTE[@]}"; do printf 'N\t%s\t%s\0' "$n" "${_CONFIG_NOTE[$n]}"; done
    } | python3 -c '
import json, os, re, sys
src, notes = {}, {}
for rec in sys.stdin.read().split("\0"):
    if rec.startswith("E\t"):
        _, n, label = rec.split("\t", 2); src[n] = label
    elif rec.startswith("N\t"):
        _, n, label, value = rec.split("\t", 3); notes[n] = (label, value)
secret = re.compile(r"TOKEN|SECRET|PASSW|CREDENTIAL|(^|_)KEY$|API_?KEY|AUTH", re.I)
env = {n: {"value": v, "source": src.get(n, "launcher")} for n, v in os.environ.items() if n in src}
for n, (label, value) in notes.items():
    if n not in env:
        env[n] = {"value": value, "source": label}
for n in env:
    if secret.search(n):
        env[n]["value"] = "<set>"
json.dump({"order": ["container-default", "site", "connector-env", "recipe", "topology", "submit"],
           "note": "the launcher\x27s environment; only the names it forwards reach the containers",
           "env": env},
          open(sys.argv[1], "w"), indent=1, sort_keys=True)
' "$f" && echo "[config] resolved environment: $f"
}
