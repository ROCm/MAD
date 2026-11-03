#!/bin/bash
# mori_ep_env.sh applies connectors/mori.env as `${KEY:-VALUE}`: every value in the
# file, under anything the container was given, and the derived values after it.
set -u
D="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0
_is() { [ "$1" = "$2" ] && { printf "  PASS  %s\n" "$3"; pass=$((pass+1)); } || { printf "  FAIL  %s\n        got:  %s\n        want: %s\n" "$3" "$1" "$2"; fail=$((fail+1)); }; }
mkdir "$TMP/bin"; printf '#!/bin/sh\necho "default via 10.0.0.254 dev ens9"\n' > "$TMP/bin/ip"; chmod +x "$TMP/bin/ip"
_env() { env -i PATH="$TMP/bin:/usr/bin:/bin" "$@" bash -c '. "$0" >/dev/null 2>&1; env' "$D/mori_ep_env.sh"; }
_get() { grep "^$1=" <<<"$2" | cut -d= -f2-; }

out="$(_env)"
missing=""
while IFS= read -r l; do
    [[ "$l" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || continue
    [ "$(_get "${l%%=*}" "$out")" = "${l#*=}" ] || missing+=" ${l%%=*}"
done < "$D/connectors/mori.env"
_is "$missing" "" "every connectors/mori.env value is exported"
_is "$(_get MORI_IB_GID_INDEX "$out")" "3" "MoRI's GID follows NCCL's default"
_is "$(_get NCCL_SOCKET_IFNAME "$out")" "ens9" "the socket interface is the default route's"
out="$(_env NCCL_IB_GID_INDEX=1 NCCL_DEBUG=WARN)"
_is "$(_get NCCL_DEBUG "$out")" "WARN" "a value the container was given wins"
_is "$(_get MORI_IB_GID_INDEX "$out")" "1" "and the derived MoRI GID follows it"
_is "$(_get NCCL_DEBUG "$(_env NCCL_DEBUG=)")" "INFO" "an empty value takes the default, as \${KEY:-VALUE}"

echo ""
echo "======================================================"
echo "  mori_env_assert: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
