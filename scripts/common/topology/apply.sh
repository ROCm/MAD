#!/bin/bash
# Sourced at the top of a launcher, before its own defaults:  . apply.sh LAUNCHER
# With MAD_TOPOLOGY naming a spec file (JSON; a relative path is taken from
# scripts/), sets the launcher's settings from it -- xP / yD / WIDE_EP / ... for a
# disagg launcher, TP_SIZE / PP_SIZE / ... for the colocated one (topology.py
# legacy-env) -- so the run is the one the file describes, and every node then
# resolves its part from the same file. A setting also given directly must agree;
# a spec the launcher cannot run is refused with what it can run.
if [ -n "${MAD_TOPOLOGY:-}" ]; then
    _topo_env="$(python3 "$(dirname "${BASH_SOURCE[0]}")/topology.py" legacy-env --launcher "$1")" \
        || { echo "ERROR: MAD_TOPOLOGY=${MAD_TOPOLOGY} cannot be run by $1 (above)" >&2; exit 1; }
    while IFS='=' read -r _k _v; do
        [ -n "$_k" ] && export "$_k=$_v"
    done <<<"$_topo_env"
    echo "[topology] MAD_TOPOLOGY=${MAD_TOPOLOGY}: $(tr '\n' ' ' <<<"$_topo_env")"
    unset _topo_env _k _v
fi
