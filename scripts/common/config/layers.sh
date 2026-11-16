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

# The recipe loader (aliases, per-accelerator variants): scripts/common/recipe/models.py.
_CONFIG_RECIPE_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/../recipe" && pwd)"

# Names set when the launcher started, and the layer that last set each name.
config_begin() {
    declare -gA _CONFIG_SRC=() _CONFIG_VAL=() _CONFIG_NOTE=() _CONFIG_RECIPE=()
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
        _CONFIG_RECIPE[$k]="$v"
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
    done < <(MODELS_YAML="$yaml" MODEL_NAME="$model" PYTHONPATH="${_CONFIG_RECIPE_LIB}${PYTHONPATH:+:$PYTHONPATH}" python3 -c '
import os, sys, yaml
import models  # scripts/common/recipe/models.py: aliases and per-accelerator variants
m = yaml.safe_load(open(os.environ["MODELS_YAML"])) or {}
try:
    r, best = models.for_job(m, os.environ["MODEL_NAME"], os.environ.get("MAD_GPU_ARCH") or None)
    r = models.with_memory(r, models.hbm_from_env())
except ValueError as e:
    sys.exit(f"[config] {os.environ['MODEL_NAME']}: {e}")
if best:
    print(f"[config] best-known configuration for {os.environ['MODEL_NAME']} on {best.get('platform')} "
          f"({best.get('date', 'undated')}, {best.get('throughput')} tok/s/GPU): over the recipe; "
          "MAD_BEST_CONFIG=0 runs the recipe as written", file=sys.stderr)
    sys.stdout.write(f"MAD_BEST_KNOWN={best.get('platform')}:{best.get('date', '')}\0")
for k, v in ((r or {}).get("env") or {}).items():
    sys.stdout.write(f"{k}={v}\0")
')
    _config_snapshot
}

# config_recipe_file MODELS_YAML MODEL: when MODEL's recipe is an alias or has per-accelerator
# variants (scripts/common/recipe/models.py), write the recipe as this job uses it to
# $LOG_PATH/$SLURM_JOB_ID/recipe.yaml and export MAD_RECIPE_FILE, its path in the containers
# (/run_logs/...), so the role scripts -- which read a recipe by name, without aliases -- run
# exactly the recipe the launcher applied. A recipe that uses neither: nothing changes.
config_recipe_file() {
    local yaml="$1" model="$2" dir tmp
    [[ -n "$model" && -f "$yaml" && -n "${LOG_PATH:-}" && -n "${SLURM_JOB_ID:-}" ]] || return 0
    if [[ -n "${MAD_RECIPE_FILE:-}" ]]; then                 # a recipe file passed through config wins
        echo "[config] ${model}: the recipe file given, ${MAD_RECIPE_FILE}"; return 0
    fi
    PYTHONPATH="${_CONFIG_RECIPE_LIB}" python3 -m models "$yaml" "$model" --uses-variants || return 0
    dir="${LOG_PATH}/${SLURM_JOB_ID}"
    mkdir -p "$dir" || return 1
    # Written whole, then renamed: every node of a job array writes the same file.
    tmp="$(mktemp "${dir}/.recipe.XXXXXX")" || return 1
    PYTHONPATH="${_CONFIG_RECIPE_LIB}" python3 -m models "$yaml" "$model" --write "$tmp" || { rm -f "$tmp"; return 1; }
    mv -f "$tmp" "${dir}/recipe.yaml"
    export MAD_RECIPE_FILE="/run_logs/${SLURM_JOB_ID}/recipe.yaml"
    config_note MAD_RECIPE_FILE "$MAD_RECIPE_FILE" launcher
    echo "[config] ${model}: the recipe as this job uses it -> ${dir}/recipe.yaml"
}

# config_recipe KEY: the recipe's value for KEY (after config_apply_recipe), or empty --
# for a launcher step that needs a recipe value on the batch node whichever layer won.
config_recipe() { [[ -v "_CONFIG_RECIPE[$1]" ]] && printf '%s\n' "${_CONFIG_RECIPE[$1]}"; return 0; }

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
