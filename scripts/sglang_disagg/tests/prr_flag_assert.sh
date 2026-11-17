#!/bin/bash
# Offline check of models.yaml `compat`, which render.py applies to the recipe's flags
# from the installed sglang's own --help: --prefill-round-robin-balance only where it
# is accepted (0.5.12 yes; 0.5.20 refuses to start on it, although its server_args.py
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

_argv() { # <install> <model>: the decode node's sglang arguments, one line
    env -i PATH="$PATH" HOME="$HOME" PYTHONPATH="$TMP/$1" PYTHONNOUSERSITE=1 \
        MODEL_NAME="$2" IPADDRS=10.0.0.1,10.0.0.2 bash -c '
        eval "$(python3 "$0/render.py" shell --rank 1)" && echo "${SERVER_ARGV[*]}"' "$DIR" 2>/dev/null
}
_has() { grep -qF -- "$2" <<<" $1 " && echo yes || echo no; }
L=Llama-3.1-8B-Instruct; B=Llama-3.1-405B-Instruct-FP8-KV

echo "=== sglang 0.5.12-like ==="
A="$(_argv v0512 $L)"
_is "$(_has "$A" ' --prefill-round-robin-balance ')" yes "accepted: the flag is passed"
_is "$(_has "$A" ' --cuda-graph-bs 8 16 32 64 128 256 512 ')" yes "--cuda-graph-bs accepted: unchanged"
echo "=== sglang 0.5.20-like ==="
A="$(_argv v0520 $L)"
_is "$(_has "$A" ' --prefill-round-robin-balance ')" no "not accepted (though server_args.py names it): the flag is left out"
_is "$(_has "$A" ' --cuda-graph-bs-decode 8 16 32 64 128 256 512 ')" yes \
    "--cuda-graph-bs split: the recipe's sizes go to --cuda-graph-bs-decode"
_is "$(_has "$A" ' --cuda-graph-bs ')" no "the old name is gone"
_is "$(_has "$(_argv v0520 $B)" ' --cuda-graph-bs-decode 1 2 3 4 5 6 7 8 ')" yes "a range (1..8) is expanded and renamed"
echo "=== no sglang readable ==="
A="$(_argv none $L)"
_is "$(_has "$A" ' --prefill-round-robin-balance ')" yes "unreadable: the flag is passed, as before"
_is "$(_has "$A" ' --cuda-graph-bs 8 16 32 64 128 256 512 ')" yes "unreadable: flags unchanged"

echo "=== launchers ==="
M="$DIR/sglang_disagg_mori_io_ep.sh"
_is "$(grep -c -- '--prefill-round-robin-balance' "$M")" "0" "no launch command hardcodes the flag"
_is "$(grep -c 'shell --rank "${NODE_RANK}"' "$M")" "1" "the role script takes its commands from render.py"
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
