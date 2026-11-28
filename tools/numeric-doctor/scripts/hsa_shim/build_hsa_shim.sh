#!/bin/bash
# Build the HSA code-object interception shim (scripts/hsa_shim/hsa_patch_shim.c).
#
# Must be built INSIDE the container it will run in: it is LD_PRELOADed into a process
# whose glibc and ROCm come from that image, and a shim built on the host against a
# different libc is exactly the sort of failure that shows up as a silent no-op.
#
# Usage:   bash build_hsa_shim.sh [OUT_SO]
# Prints the path to the built .so on stdout (and nothing else on stdout), so callers
# can capture it.  Diagnostics go to stderr.
#
# Rebuilds only when the source is newer than the object, so a sweep that calls this
# once per trial pays for one compile.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
SRC="$HERE/hsa_patch_shim.c"
OUT=${1:-${NUMERIC_DOCTOR_HSA_SHIM_OUT:-/tmp/numeric_doctor_hsa_shim.so}}

[ -r "$SRC" ] || { echo "missing $SRC" >&2; exit 1; }

if [ -f "$OUT" ] && [ "$OUT" -nt "$SRC" ]; then
  echo "$OUT"
  exit 0
fi

CC=${CC:-}
if [ -z "$CC" ]; then
  for c in gcc cc clang /opt/rocm/llvm/bin/clang; do
    if command -v "$c" >/dev/null 2>&1; then CC=$c; break; fi
  done
fi
[ -n "$CC" ] || { echo "no C compiler found (tried gcc, cc, clang)" >&2; exit 1; }

mkdir -p -- "$(dirname -- "$OUT")"
tmp="$OUT.$$.tmp"
# -ldl for dlsym(RTLD_NEXT).  No ROCm headers needed: the shim spells out the one
# ABI it depends on, so this builds on a node with a compiler and nothing else.
"$CC" -shared -fPIC -O2 -Wall -Wextra -o "$tmp" "$SRC" -ldl >&2
mv -f "$tmp" "$OUT"
echo "built $OUT with $CC" >&2
echo "$OUT"
