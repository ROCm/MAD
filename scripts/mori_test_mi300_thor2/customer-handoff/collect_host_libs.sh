#!/bin/bash
# collect_host_libs.sh — gather the 5 host RDMA libraries the Dockerfile bakes in.
#
# Run this on a compute node that is ALREADY on the bnxt 235.2.86.0 driver.
# It copies the real .so files into ./libs/ with the exact filenames the Dockerfile expects,
# and verifies the bnxt provider is the 235 build + libibverbs is the v34 ABI.
#
# Usage:  bash collect_host_libs.sh        # writes ./libs/*
set -euo pipefail
OUT="$(dirname "$0")/libs"
mkdir -p "$OUT"

say(){ echo "[collect-libs] $*"; }
die(){ echo "[collect-libs] ERROR: $*" >&2; exit 1; }

# --- resolve each lib to its REAL file, copy with the versioned name the Dockerfile COPYs ---
copy_real(){  # $1 = soname to resolve, $2 = search dir
  local soname="$1" dir="$2" real
  real="$(readlink -f "$dir/$soname" 2>/dev/null || true)"
  [ -n "$real" ] && [ -f "$real" ] || die "cannot resolve $dir/$soname — is rdma-core installed?"
  cp -f "$real" "$OUT/$(basename "$real")"
  say "copied $(basename "$real")  (from $soname)"
}

# 4 distro libs (rdma-core + libnl): libibverbs1 / librdmacm1 / libnl-3-200 / libnl-route-3-200
copy_real libibverbs.so.1     /usr/lib/x86_64-linux-gnu
copy_real librdmacm.so.1      /usr/lib/x86_64-linux-gnu
copy_real libnl-3.so.200      /usr/lib/x86_64-linux-gnu
copy_real libnl-route-3.so.200 /usr/lib/x86_64-linux-gnu

# 1 driver-provided lib: the bnxt 235 RoCE provider (from the 235 driver install, NOT a distro pkg)
BNXT="$(ls /usr/local/lib/x86_64-linux-gnu/libbnxt_re-rdmav34.so /usr/local/lib/libbnxt_re-rdmav34.so 2>/dev/null | head -1 || true)"
[ -n "$BNXT" ] || die "libbnxt_re-rdmav34.so not found — install the bnxt 235.2.86.0 driver first (see ../driver-235.2.86.0/)."
cp -fL "$BNXT" "$OUT/libbnxt_re-rdmav34.so"
say "copied libbnxt_re-rdmav34.so  (from $BNXT)"

# --- sanity checks ---
V="$(strings "$OUT/$(basename "$(readlink -f /usr/lib/x86_64-linux-gnu/libibverbs.so.1)")" 2>/dev/null | grep -m1 IBVERBS_PRIVATE || true)"
[ "$V" = "IBVERBS_PRIVATE_34" ] || die "host libibverbs is '$V', expected IBVERBS_PRIVATE_34 (v34). Wrong host stack."
SZ="$(stat -c %s "$OUT/libbnxt_re-rdmav34.so" 2>/dev/null || echo 0)"
[ "$SZ" -gt 400000 ] || die "libbnxt_re-rdmav34.so is only $SZ bytes — expected ~539696 (235 build). Is this host on the 235 driver?"

echo ""
say "OK — ./libs/ ready for 'docker build':"
ls -la "$OUT" | grep -vE '\.gitkeep|^total|^d'
echo ""
say "Host driver check:  $(modinfo bnxt_re 2>/dev/null | awk '/^version:/{print "bnxt_re="$2}')  (expect 235.2.86.0)"
