#!/bin/bash
# debug-numeric step 8: serve a patched aiter .co -- build an overlay of this image's stock aiter tree
# with the patched file in place, and print the exports that make aiter load it.
#
#   eval "$(bash aiter_overlay.sh <patched.co> <rel path, e.g. gfx950/mla/<name>.co> <overlay dir>)"
#
# amd-aiter 0.1.26+ rewrites AITER_ASM_DIR from AITER_META_DIR at import (aiter/jit/core.py), and
# honours AITER_META_DIR only if it holds csrc/: when the stock root is <meta>/hsa the overlay
# mirrors the whole meta dir (the patched hsa/ farm, links to the rest) and both are exported.
# Older aiter reads AITER_ASM_DIR.  Paths come from the aiter of THIS image, so the package runs
# anywhere the reproducer does.  Built once per overlay dir, under a lock: parallel arms sourcing
# it at once are safe.  Check a run loaded it: aiter's `LoadKernel: ... hsaco: <overlay>/...`
# (0.1.26+, AITER_LOG_LEVEL=INFO) or `[aiter] hipModuleLoad: <overlay>/...` line.
set -eu
CO=$(cd "$(dirname "${1:?patched .co}")" && pwd -P)/$(basename "$1"); REL=${2:?rel path under the asm root}
D=${3:?overlay dir}; mkdir -p "$D"; D=$(cd "$D" && pwd -P)
STOCK=${AITER_STOCK_ASM_DIR:-$(env -u AITER_ASM_DIR -u AITER_META_DIR python3 -c \
  'import aiter.jit.core as c; print(c.AITER_ASM_DIR)' 2>/dev/null | tail -n 1)}
[ -d "$STOCK" ] || STOCK=$(python3 -c 'import os, aiter_meta; print(os.path.join(os.path.dirname(aiter_meta.__file__), "hsa"))')
STOCK=$(cd "$STOCK" && pwd -P)
[ -f "$STOCK/$REL" ] || { echo "aiter_overlay: $STOCK/$REL is not in this image's aiter" >&2; exit 2; }
META=$(dirname "$STOCK")
if [ "$(basename "$STOCK")" = hsa ] && [ -d "$META/csrc" ]; then FARM=$D/aiter_meta/hsa; else FARM=$D/aiter_asm; META=; fi
until mkdir "$D/.lock" 2>/dev/null; do sleep 1; done
trap 'rmdir "$D/.lock"' EXIT
if ! cmp -s "$CO" "$FARM/$REL"; then
  rm -rf "$FARM" && mkdir -p "$FARM" && cp -as "$STOCK/." "$FARM/"
  rm -f "$FARM/$REL" && cp "$CO" "$FARM/$REL"
  if [ -n "$META" ]; then
    for e in "$META"/*; do [ "$(basename "$e")" = hsa ] || ln -sfn "$e" "$D/aiter_meta/"; done
  fi
fi
echo "export AITER_ASM_DIR=$FARM/"
[ -n "$META" ] && echo "export AITER_META_DIR=$D/aiter_meta"
echo "# aiter overlay: $REL patched from $CO; stock $STOCK" >&2
