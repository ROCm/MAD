#!/bin/bash
# Adapting the recipes' sglang flags to the sglang that is installed. Sourced by the
# disagg launchers (sglang_disagg_mori_io_ep.sh, sglang_disagg_server.sh).
#
# The recipes were tuned on sglang 0.5.12. Later releases changed the command line:
#   - --prefill-round-robin-balance was removed: 0.5.20 refuses to start with
#     "unrecognized arguments".
#   - --cuda-graph-bs was split into --cuda-graph-bs-decode / --cuda-graph-bs-prefill:
#     0.5.20 refuses with "ambiguous option". The old flag sized the decode graphs, so
#     its meaning is carried by --cuda-graph-bs-decode.
# What a release accepts is read from its own --help (argparse lists every option),
# once per launcher. Looking for the flag's name in server_args.py was not enough:
# 0.5.20 still mentions --prefill-round-robin-balance there but does not accept it.
# If --help cannot be read, flags pass unchanged, as before.

# sglang's option list, read once.
_sgl_options() {
    if [ -z "${_SGL_HELP+x}" ]; then
        _SGL_HELP="$(python3 -m sglang.launch_server --help 2>/dev/null)"
    fi
    printf '%s' "$_SGL_HELP"
}

# True when the installed sglang accepts option $1 exactly (not as a prefix of a longer
# option), or when its option list cannot be read.
sgl_has_option() {
    local opts; opts="$(_sgl_options)"
    [ -z "$opts" ] && return 0
    grep -qE -- "(^|[[:space:][])$1([]=,[:space:]]|$)" <<<"$opts"
}

# The given flag string with renamed options mapped to the installed sglang's names.
sgl_compat_flags() {
    local flags="$1"
    if grep -qE -- '(^|[[:space:]])--cuda-graph-bs([[:space:]]|$)' <<<"$flags" \
        && ! sgl_has_option --cuda-graph-bs && sgl_has_option --cuda-graph-bs-decode; then
        flags="$(sed -E 's/(^|[[:space:]])--cuda-graph-bs([[:space:]]|$)/\1--cuda-graph-bs-decode\2/g' <<<"$flags")"
        echo "[sglang] this sglang splits --cuda-graph-bs; passing the recipe's sizes as --cuda-graph-bs-decode" >&2
    fi
    printf '%s' "$flags"
}

# --prefill-round-robin-balance when the installed sglang accepts it, else empty.
sgl_prr_flag() {
    if sgl_has_option --prefill-round-robin-balance; then
        printf '%s' "--prefill-round-robin-balance"
    else
        echo "[sglang] this sglang has no --prefill-round-robin-balance; launching without it" >&2
    fi
}
