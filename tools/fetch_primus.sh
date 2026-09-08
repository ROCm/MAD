#!/usr/bin/env bash
###############################################################################
#
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################

# Check Primus out into scripts/Primus. The JAX backends need it before anything else
# happens: scripts/jax-{maxtext,maxdiffusion}/get_models_json.py glob its example configs
# to enumerate models, and both primus_* dockerfiles COPY the tree into the image. Without
# it, discovery reports zero models instead of a missing prerequisite.
#
# Run on the host, from anywhere:
#   tools/fetch_primus.sh
#
# Idempotent, so it is safe in CI or a Makefile. Override PRIMUS_URL, PRIMUS_REF, or
# PRIMUS_DIR for a fork, another branch or commit, or a different location.
set -uo pipefail

PRIMUS_URL="${PRIMUS_URL:-https://github.com/AMD-AGI/Primus}"
PRIMUS_REF="${PRIMUS_REF:-main}"

MAD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRIMUS_DIR="${PRIMUS_DIR:-$MAD_ROOT/scripts/Primus}"

log() { echo "[fetch-primus] $*"; }
die() { echo "[fetch-primus] ERROR: $*" >&2; exit 1; }

command -v git >/dev/null || die "git not found on PATH."

if git -C "$PRIMUS_DIR" rev-parse --git-dir >/dev/null 2>&1; then
  log "already checked out at $PRIMUS_DIR"
elif [[ -d "$PRIMUS_DIR" ]] && [[ -z "$(ls -A "$PRIMUS_DIR" 2>/dev/null)" ]]; then
  # Empty dir left by an uninitialized git submodule; remove so clone succeeds.
  rmdir "$PRIMUS_DIR"
  log "cloning $PRIMUS_URL ($PRIMUS_REF) into $PRIMUS_DIR"
  git clone --branch "$PRIMUS_REF" "$PRIMUS_URL" "$PRIMUS_DIR" \
    || die "clone failed. For a private repo, check that your git credentials can read $PRIMUS_URL."
elif [[ -e "$PRIMUS_DIR" ]]; then
  die "$PRIMUS_DIR exists but is not a git checkout. Move it aside and re-run."
else
  log "cloning $PRIMUS_URL ($PRIMUS_REF) into $PRIMUS_DIR"
  # Deliberately not --recursive, and no submodules are initialized afterwards. Both
  # primus_* images take their framework from the base image (/workspace/maxtext,
  # /workspace/maxdiffusion) and pin MAXTEXT_PATH / MAXDIFFUSION_PATH to it. A submodule
  # checkout would only add an unpatched second copy of the same commit to the build context.
  git clone --branch "$PRIMUS_REF" "$PRIMUS_URL" "$PRIMUS_DIR" \
    || die "clone failed. For a private repo, check that your git credentials can read $PRIMUS_URL."
fi

maxtext_configs=$(find "$PRIMUS_DIR/examples/maxtext/configs" -name '*.yaml' 2>/dev/null | wc -l)
maxdiff_configs=$(find "$PRIMUS_DIR/examples/maxdiffusion/configs" -name '*.yaml' 2>/dev/null | wc -l)
log "$(git -C "$PRIMUS_DIR" rev-parse --short HEAD) on $(git -C "$PRIMUS_DIR" rev-parse --abbrev-ref HEAD)"
# Config files, not discovered models: discovery filters the multi-node-only ones out.
log "config files found: $maxtext_configs maxtext, $maxdiff_configs maxdiffusion"
[[ "$maxtext_configs" -gt 0 ]] || die "no maxtext configs found; is $PRIMUS_REF the right ref?"
