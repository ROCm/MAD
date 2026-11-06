#!/bin/bash
# Offline check of sglang_compat.sh, which adapts the recipes' flags to the installed
# sglang, read from its own --help: --prefill-round-robin-balance only where it is
# accepted (0.5.12 yes; 0.5.20 refuses to start on it, although its server_args.py
# still mentions the name), --cuda-graph-bs mapped to --cuda-graph-bs-decode where it
# was split (0.5.20: "ambiguous option"), and everything unchanged when sglang cannot be
# read. Also that a server rejecting its command line is caught as fatal at once. Uses
# fake sglang installs whose launch_server is a real argparse parser. No GPUs.
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0
_is() { [ "$1" = "$2" ] && { printf "  PASS  %s\n" "$3"; pass=$((pass+1)); } || { printf "  FAIL  %s (got [%s], want [%s])\n" "$3" "$1" "$2"; fail=$((fail+1)); }; }

_sglang() { # NAME "argparse add_argument lines" "server_args.py text"
    mkdir -p "$TMP/$1/sglang/srt"; : > "$TMP/$1/sglang/__init__.py"
    printf '%s\n' "$3" > "$TMP/$1/sglang/srt/server_args.py"
    printf 'import argparse\np = argparse.ArgumentParser(prog="sglang serve")\n%s\np.parse_args()\n' "$2" \
        > "$TMP/$1/sglang/launch_server.py"
}
_sglang v0512 'p.add_argument("--prefill-round-robin-balance", action="store_true")
p.add_argument("--cuda-graph-bs", type=int, nargs="+")
p.add_argument("--disable-radix-cache", action="store_true")' 'prefill_round_robin_balance: bool = False'
_sglang v0520 'p.add_argument("--cuda-graph-bs-decode", type=int, nargs="+")
p.add_argument("--cuda-graph-bs-prefill", type=int, nargs="+")
p.add_argument("--disable-radix-cache", action="store_true")' '# removed: --prefill-round-robin-balance (use --load-balance-method)'
mkdir -p "$TMP/none"

_run() { # <install> <bash snippet using the compat functions>
    env -i PATH="$PATH" HOME="$HOME" PYTHONPATH="$TMP/$1" PYTHONNOUSERSITE=1 bash -c "
        source '$DIR/sglang_compat.sh'; $2" 2>/dev/null
}
F='--disable-radix-cache --cuda-graph-bs 8 16 32 64 128 256 512'

echo "=== sglang 0.5.12-like ==="
_is "$(_run v0512 'sgl_prr_flag')" "--prefill-round-robin-balance" "accepted: the flag is passed"
_is "$(_run v0512 "sgl_compat_flags '$F'")" "$F" "--cuda-graph-bs accepted: flags unchanged"
echo "=== sglang 0.5.20-like ==="
_is "$(_run v0520 'sgl_prr_flag')" "" "not accepted (though server_args.py names it): the flag is left out"
_is "$(_run v0520 "sgl_compat_flags '$F'")" "--disable-radix-cache --cuda-graph-bs-decode 8 16 32 64 128 256 512" \
    "--cuda-graph-bs split: the recipe's sizes go to --cuda-graph-bs-decode"
_is "$(_run v0520 "sgl_compat_flags '--cuda-graph-bs-decode 4 8'")" "--cuda-graph-bs-decode 4 8" "an already-new flag is left alone"
_is "$(_run v0520 'sgl_has_option --cuda-graph-bs && echo yes || echo no')" "no" "--cuda-graph-bs-decode does not count as --cuda-graph-bs"
_is "$(_run v0520 "sgl_compat_flags '--cuda-graph-bs \$(seq 1 8)'")" '--cuda-graph-bs-decode $(seq 1 8)' "a recipe's \$(seq ...) is kept for the launcher to expand"
echo "=== no sglang readable ==="
_is "$(_run none 'sgl_prr_flag')" "--prefill-round-robin-balance" "unreadable: the flag is passed, as before"
_is "$(_run none "sgl_compat_flags '$F'")" "$F" "unreadable: flags unchanged"

echo "=== launchers ==="
M="$DIR/sglang_disagg_mori_io_ep.sh"
_is "$(grep -c -- '--prefill-round-robin-balance \\$' "$M")" "0" "no launch command hardcodes the flag"
_is "$(grep -c 'source "${SCRIPT_DIR}/sglang_compat.sh"' "$M")" "1" "the MoRI launcher uses sglang_compat.sh"
_is "$(grep -c 'sglang_compat.sh"' "$DIR/sglang_disagg_server.sh")" "1" "the Mooncake launcher uses sglang_compat.sh"
RE="$(sed -n "s/^_FATAL_SERVER_LOG_RE='\(.*\)'$/\1/p" "$M")"
for line in 'sglang serve: error: unrecognized arguments: --prefill-round-robin-balance' \
            'sglang serve: error: ambiguous option: --cuda-graph-bs could match --cuda-graph-bs-decode, --cuda-graph-bs-prefill'; do
    _is "$(grep -cE "$RE" <<<"$line")" "1" "fatal at once: ${line:0:60}..."
done
_is "$(grep -cE "$RE" <<<'INFO: The server is fired up and ready to roll!')" "0" "a healthy line is not fatal"

echo "======================================================"
echo "  prr_flag_assert: ${pass} passed, ${fail} failed"
echo "======================================================"
[[ "$fail" == "0" ]]
